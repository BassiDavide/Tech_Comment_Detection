import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def compute_token_metrics_from_logits(token_logits, token_labels, attention_mask=None, threshold: float = 0.5):
    """
    Compute micro/macro precision/recall/F1 for multi-label token predictions.
    """
    probs = 1.0 / (1.0 + np.exp(-np.array(token_logits)))
    preds = (probs > threshold).astype(int)

    labels = np.array(token_labels).astype(int)
    if attention_mask is None:
        mask = np.ones(labels.shape[:2], dtype=bool)
    else:
        mask = np.array(attention_mask).astype(bool)

    preds = preds[mask]
    labels = labels[mask]

    if preds.size == 0:
        return {
            "token_precision": 0.0,
            "token_recall": 0.0,
            "token_f1_micro": 0.0,
            "token_f1_macro": 0.0,
        }

    token_precision = precision_score(labels, preds, average="micro", zero_division=0)
    token_recall = recall_score(labels, preds, average="micro", zero_division=0)
    token_f1_micro = f1_score(labels, preds, average="micro", zero_division=0)

    per_label_f1 = f1_score(labels, preds, average=None, zero_division=0)
    support = labels.sum(axis=0)
    if np.any(support > 0):
        token_f1_macro = float(per_label_f1[support > 0].mean())
    else:
        token_f1_macro = 0.0

    return {
        "token_precision": token_precision,
        "token_recall": token_recall,
        "token_f1_micro": token_f1_micro,
        "token_f1_macro": token_f1_macro,
    }


def compute_token_per_label_metrics(token_logits, token_labels, attention_mask, label_list=None, threshold: float = 0.5):
    """
    Compute precision/recall/F1 for each token-level label.
    """
    probs = 1.0 / (1.0 + np.exp(-np.array(token_logits)))
    preds = (probs > threshold).astype(int)

    labels = np.array(token_labels).astype(int)
    mask = np.array(attention_mask).astype(bool)

    preds = preds[mask]
    labels = labels[mask]

    if label_list is None:
        label_list = [f"label_{idx}" for idx in range(labels.shape[1])]

    precision_per_label = precision_score(labels, preds, average=None, zero_division=0)
    recall_per_label = recall_score(labels, preds, average=None, zero_division=0)
    f1_per_label = f1_score(labels, preds, average=None, zero_division=0)
    support_per_label = labels.sum(axis=0)

    metrics = {}
    for idx, label_name in enumerate(label_list):
        metrics[label_name] = {
            "precision": float(precision_per_label[idx]),
            "recall": float(recall_per_label[idx]),
            "f1": float(f1_per_label[idx]),
            "support": int(support_per_label[idx]),
        }

    return metrics


def compute_span_metrics(eval_pred, threshold: float = 0.5):
    """
    Compute span-level metrics from Trainer EvalPrediction.
    """
    token_logits = eval_pred.predictions
    label_ids = eval_pred.label_ids
    if isinstance(label_ids, (list, tuple)) and len(label_ids) == 2:
        token_labels, attention_mask = label_ids
    else:
        token_labels = label_ids
        attention_mask = None
    return compute_token_metrics_from_logits(
        token_logits,
        token_labels,
        attention_mask=attention_mask,
        threshold=threshold,
    )


def compute_pos_weight_from_dataset(dataset, num_labels: int, field: str = "token_labels"):
    """
    Compute per-label positive weights for BCEWithLogitsLoss based on class imbalance.
    Includes smoothing, clamping, and normalization to stabilize gradients.
    """
    pos_counts = torch.zeros(num_labels)
    total_counts = 0

    for example in dataset:
        labels = example[field]
        if isinstance(labels, torch.Tensor):
            labels = labels.detach()
        else:
            labels = torch.tensor(labels)
        pos_counts += labels.sum(dim=0)
        total_counts += labels.numel() / num_labels

    neg_counts = total_counts - pos_counts
    pos_weight = neg_counts / (pos_counts + 1e-8)

    pos_weight = (neg_counts + 1.0) / (pos_counts + 5.0)
    pos_weight = torch.clamp(pos_weight, max=300.0)
    pos_weight = pos_weight / pos_weight.mean() * 2.0

    print(
        f"[compute_pos_weight_from_dataset] Mean pos_weight: {pos_weight.mean():.2f}, "
        f"Min: {pos_weight.min():.2f}, Max: {pos_weight.max():.2f}"
    )

    return pos_weight


def print_training_summary(metrics, step, mode="train"):
    """Pretty print training metrics."""
    print(f"\n{'='*60}")
    print(f"{mode.upper()} Metrics at Step {step}")
    print(f"{'='*60}")

    if "loss" in metrics:
        print(f"Total Loss: {metrics['loss']:.4f}")
    if "token_loss" in metrics:
        print(f"  Token Loss: {metrics['token_loss']:.4f}")

    print(f"\nSpan Detection:")
    if "token_f1_micro" in metrics:
        print(f"  F1 Micro: {metrics['token_f1_micro']:.4f}")
    if "token_f1_macro" in metrics:
        print(f"  F1 Macro: {metrics['token_f1_macro']:.4f}")
    if "token_precision" in metrics:
        print(f"  Precision: {metrics['token_precision']:.4f}")
    if "token_recall" in metrics:
        print(f"  Recall: {metrics['token_recall']:.4f}")

    print(f"{'='*60}\n")
