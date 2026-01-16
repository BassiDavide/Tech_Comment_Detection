import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

def compute_comment_metrics(predictions, labels, threshold=0.3):
    """
    Compute metrics for comment-level multi-label classification
    
    Args:
        predictions: Predicted logits (batch_size, num_labels)
        labels: True labels (batch_size, num_labels)
        threshold: Threshold for binary classification
        
    Returns:
        Dictionary of metrics
    """
    # Convert logits to binary predictions
    probs = torch.sigmoid(torch.tensor(predictions))
    preds = (probs > threshold).int().numpy()
    labels = np.array(labels)

    # Calculate metrics
    f1_micro = f1_score(labels, preds, average='micro', zero_division=0)

    # Ignore labels that never appear in the gold data when computing macro averages
    support = labels.sum(axis=0)
    per_label_f1 = f1_score(labels, preds, average=None, zero_division=0)
    if np.any(support > 0):
        f1_macro = float(per_label_f1[support > 0].mean())
    else:
        f1_macro = 0.0

    precision_micro = precision_score(labels, preds, average='micro', zero_division=0)
    recall_micro = recall_score(labels, preds, average='micro', zero_division=0)
    
    return {
        'comment_f1_micro': f1_micro,
        'comment_f1_macro': f1_macro,
        'comment_precision': precision_micro,
        'comment_recall': recall_micro,
    }


def compute_comment_per_label_metrics(predictions, labels, threshold=0.3, label_list=None):
    """
    Compute precision/recall/F1 for each individual comment-level label.
    """
    probs = torch.sigmoid(torch.tensor(predictions))
    preds = (probs > threshold).int().numpy()
    labels = np.array(labels)

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

def compute_token_metrics_from_logits(token_logits, token_labels, attention_mask, threshold: float = 0.5):
    """
    Compute micro precision/recall/F1 for multi-label token predictions.
    """
    probs = 1.0 / (1.0 + np.exp(-np.array(token_logits)))
    preds = (probs > threshold).astype(int)

    mask = np.array(attention_mask).astype(bool)
    preds = preds[mask]
    labels = np.array(token_labels).astype(int)[mask]

    if preds.size == 0:
        return {'token_precision': 0.0, 'token_recall': 0.0, 'token_f1': 0.0}

    token_precision = precision_score(labels, preds, average='micro', zero_division=0)
    token_recall = recall_score(labels, preds, average='micro', zero_division=0)
    token_f1 = f1_score(labels, preds, average='micro', zero_division=0)

    return {
        "token_precision": token_precision,
        "token_recall": token_recall,
        "token_f1": token_f1,
    }

def compute_multitask_metrics(eval_pred, threshold=0.3):
    """
    Compute metrics for both tasks in multi-task learning
    
    Args:
        eval_pred: Tuple of (predictions, labels) from Trainer
        threshold: Threshold for comment classification
        
    Returns:
        Dictionary with metrics for both tasks
    """
    (comment_logits, token_logits, *_), (comment_labels, token_labels, attention_mask) = eval_pred
    
    # Compute metrics for both tasks
    comment_metrics = compute_comment_metrics(comment_logits, comment_labels, threshold)
    token_metrics = compute_token_metrics_from_logits(token_logits, token_labels, attention_mask, threshold=0.5)
    
    # Combine metrics
    all_metrics = {**comment_metrics, **token_metrics}
    
    return all_metrics


import torch

def compute_pos_weight_from_dataset(dataset, num_labels: int, field: str = "token_labels"):
    """
    Compute per-label positive weights for BCEWithLogitsLoss based on class imbalance.
    Includes smoothing, clamping, and normalization to stabilize gradients.
    """
    pos_counts = torch.zeros(num_labels)
    total_counts = 0

    # Count occurrences
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

    # ---- 🧩 NEW ADDITIONS ----
    # 1. Smooth extremely rare classes
    pos_weight = (neg_counts + 1.0) / (pos_counts + 5.0)

    # 2. Clamp to avoid exploding gradients
    pos_weight = torch.clamp(pos_weight, max=300.0)

    # 3. Normalize around mean ≈ 2.0 for stability
    pos_weight = pos_weight / pos_weight.mean() * 2.0

    # Optional print for debugging
    print(f"[compute_pos_weight_from_dataset] Mean pos_weight: {pos_weight.mean():.2f}, "
          f"Min: {pos_weight.min():.2f}, Max: {pos_weight.max():.2f}")

    return pos_weight


class MultitaskTrainer:
    """Custom training logic for multi-task model (if needed)"""
    
    @staticmethod
    def compute_loss(model, inputs, return_outputs=False):
        """
        Custom loss computation for multi-task learning
        This is compatible with HuggingFace Trainer
        """
        comment_labels = inputs.pop("comment_labels")
        token_labels = inputs.pop("token_labels")
        
        outputs = model(**inputs, comment_labels=comment_labels, token_labels=token_labels)
        
        loss = outputs['loss']
        
        return (loss, outputs) if return_outputs else loss

def print_training_summary(metrics, step, mode="train"):
    """Pretty print training metrics"""
    print(f"\n{'='*60}")
    print(f"{mode.upper()} Metrics at Step {step}")
    print(f"{'='*60}")
    
    if 'loss' in metrics:
        print(f"Total Loss: {metrics['loss']:.4f}")
    if 'comment_loss' in metrics:
        print(f"  Comment Loss: {metrics['comment_loss']:.4f}")
    if 'token_loss' in metrics:
        print(f"  Token Loss: {metrics['token_loss']:.4f}")
    
    print(f"\nComment Classification:")
    if 'comment_f1_micro' in metrics:
        print(f"  F1 Micro: {metrics['comment_f1_micro']:.4f}")
    if 'comment_f1_macro' in metrics:
        print(f"  F1 Macro: {metrics['comment_f1_macro']:.4f}")
    if 'comment_precision' in metrics:
        print(f"  Precision: {metrics['comment_precision']:.4f}")
    if 'comment_recall' in metrics:
        print(f"  Recall: {metrics['comment_recall']:.4f}")
    
    print(f"\nToken Cue Detection:")
    if 'token_precision' in metrics:
        print(f"  Precision: {metrics['token_precision']:.4f}")
    if 'token_recall' in metrics:
        print(f"  Recall: {metrics['token_recall']:.4f}")
    if 'token_f1' in metrics:
        print(f"  F1: {metrics['token_f1']:.4f}")
    
    print(f"{'='*60}\n")
