import os
import json
import math
import traceback
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import optuna
import torch
from optuna.trial import Trial
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    EvalPrediction,
    set_seed,
)

from data_processing import prepare_datasets
from utils import compute_comment_metrics

# Shim for Python 3.8: ensure packages_distributions exists
import importlib.metadata as _im
if not hasattr(_im, "packages_distributions"):
    try:
        import importlib_metadata as _im_backport
        _im.packages_distributions = _im_backport.packages_distributions
    except Exception:
        _im.packages_distributions = lambda: {}

# Base configuration (mirrors the multitask script)
BASE_CONFIG = {
    "model_name": "/mnt/beegfs/home/davide.bassi/Comm_Tech/Nature/Base_Nature/comment_pre_ft_output",
    "train_file": "/mnt/beegfs/home/davide.bassi/Comm_Tech/Data/Comments/merged_splits/train.jsonl",
    "dev_file": "/mnt/beegfs/home/davide.bassi/Comm_Tech/Data/Comments/merged_splits/dev.jsonl",
    "test_file": "/mnt/beegfs/home/davide.bassi/Comm_Tech/Data/Comments/merged_splits/test.jsonl",
    "output_dir": "./CommentOnly_Nature_results",
    "logging_steps": 50,
    "eval_steps": 200,
    "save_steps": 200,
    "save_total_limit": 2,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "fp16": False,
}

LABEL_LIST = [
    "appeal to authority",
    "appeal to fear, prejudice",
    "appeal to hypocrisy (to quoque)",
    "appeal to pity",
    "appeal to popularity (bandwagon)",
    "appeal to time",
    "appeal to values/flag waving",
    "black-and-white fallacy",
    "causal oversimplification",
    "distraction",
    "exaggeration or minimisation",
    "intentional vagueness",
    "loaded language",
    "name calling",
    "reductio ad hitlerum",
    "repetition",
    "slogans",
    "smears/doubt",
    "thought-terminating cliché",
]


def freeze_bottom_layers(model, num_layers_to_freeze: int = 6):
    """
    Freeze lower encoder layers for stability.
    """
    encoder = None
    for attr in ["deberta", "encoder", "model", "base_model"]:
        if hasattr(model, attr):
            candidate = getattr(model, attr)
            if hasattr(candidate, "encoder"):
                encoder = candidate.encoder
                break
            if hasattr(candidate, "layer"):
                encoder = candidate
                break
    if encoder and hasattr(encoder, "layer"):
        target_layers = encoder.layer[:num_layers_to_freeze]
        for _, param in target_layers.named_parameters():
            param.requires_grad = False
        frozen_params = sum(1 for p in model.parameters() if not p.requires_grad)
        total_params = sum(1 for _ in model.parameters())
        percent = 100.0 * frozen_params / max(1, total_params)
        print(f"[Freeze] Frozen bottom {num_layers_to_freeze} layers ({percent:.1f}% params).")
    else:
        print("[Freeze] Warning: could not access encoder layers. No freezing applied.")


class CommentOnlyTrainer(Trainer):
    """
    Trainer that uses comment_labels only.
    """

    def compute_loss(self, model, inputs, return_outputs: bool = False, num_items_in_batch: Optional[int] = None):
        inputs.pop("token_labels", None)
        labels = inputs.pop("comment_labels", None)
        if labels is None:
            raise ValueError("Missing comment_labels in batch.")
        labels = labels.float()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


def objective(trial: Trial, tokenizer):
    """
    Optuna objective: multi-label comment classification.
    """
    max_length = trial.suggest_int("max_length", 256, 512, step=64)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 3e-5, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
    gradient_accumulation_steps = trial.suggest_categorical("gradient_accumulation_steps", [1, 2, 4])
    num_epochs = trial.suggest_int("num_epochs", 10, 24, step=2)
    dropout_prob = trial.suggest_float("dropout_prob", 0.05, 0.2, step=0.05)
    weight_decay = trial.suggest_float("weight_decay", 0.01, 0.05, step=0.01)
    freeze_layers = trial.suggest_int("freeze_layers", 0, 6, step=2)
    threshold = trial.suggest_float("threshold", 0.1, 0.3, step=0.05)

    trial_dir = f"{BASE_CONFIG['output_dir']}/trial_{trial.number}"
    os.makedirs(trial_dir, exist_ok=True)
    with open(f"{trial_dir}/trial_params.json", "w") as f:
        json.dump(trial.params, f, indent=2)

    print(f"\n{'='*80}\nTRIAL {trial.number}\n{'='*80}")
    for k, v in trial.params.items():
        print(f"  {k}: {v}")
    print(f"{'='*80}\n")

    train_dataset, dev_dataset, _, _ = prepare_datasets(
        BASE_CONFIG["train_file"],
        BASE_CONFIG["dev_file"],
        BASE_CONFIG["test_file"],
        LABEL_LIST,
        tokenizer,
        max_length=max_length,
    )

    config = AutoConfig.from_pretrained(BASE_CONFIG["model_name"])
    config.num_labels = len(LABEL_LIST)
    config.problem_type = "multi_label_classification"
    config.hidden_dropout_prob = dropout_prob
    config.attention_probs_dropout_prob = dropout_prob

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_CONFIG["model_name"],
        config=config,
        ignore_mismatched_sizes=True,
    )
    if freeze_layers > 0:
        freeze_bottom_layers(model, num_layers_to_freeze=freeze_layers)

    training_args = TrainingArguments(
        output_dir=trial_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=BASE_CONFIG["warmup_ratio"],
        weight_decay=weight_decay,
        logging_dir=f"{trial_dir}/logs",
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=BASE_CONFIG["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_comment_f1_macro",
        greater_is_better=True,
        fp16=BASE_CONFIG["fp16"],
        report_to="none",
        remove_unused_columns=True,
        label_names=["comment_labels"],
    )

    def compute_metrics_fn(eval_pred: EvalPrediction):
        logits, labels = eval_pred
        return compute_comment_metrics(logits, labels, threshold=threshold)

    trainer = CommentOnlyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics_fn,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.001,
            )
        ],
    )

    try:
        trainer.train()
        eval_results = trainer.evaluate()
        with open(f"{trial_dir}/eval_results.json", "w") as f:
            json.dump({k: float(v) if isinstance(v, np.floating) else v for k, v in eval_results.items()}, f, indent=2)
        return eval_results["eval_comment_f1_macro"]
    except Exception as e:
        traceback.print_exc()
        print(f"Trial {trial.number} failed: {e}")
        return 0.0


def run_optimization(n_trials: int = 20, n_jobs: int = 1):
    print("=" * 80)
    print("COMMENT-ONLY HYPERPARAMETER OPTIMIZATION WITH OPTUNA")
    print("=" * 80)
    print(f"\nOutput dir: {BASE_CONFIG['output_dir']}")
    print(f"Trials: {n_trials}, Jobs: {n_jobs}")
    print(f"Metric: eval_comment_f1_macro\n")

    os.makedirs(BASE_CONFIG["output_dir"], exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_CONFIG["model_name"])

    study = optuna.create_study(
        direction="maximize",
        study_name="comment_only_optimization",
        storage=f"sqlite:///{BASE_CONFIG['output_dir']}/optuna_study.db",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )

    study.optimize(
        lambda trial: objective(trial, tokenizer),
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True,
    )

    with open(f"{BASE_CONFIG['output_dir']}/best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)

    trials_df = study.trials_dataframe()
    trials_df.to_csv(f"{BASE_CONFIG['output_dir']}/all_trials.csv", index=False)

    print("\nBest trial:", study.best_trial.number)
    print("Best F1 Macro:", study.best_value)
    return study


def train_with_best_params(study=None, test_dataset=None, seed: int = 42, output_suffix: str = "final_model"):
    if study is None:
        study = optuna.load_study(
            study_name="comment_only_optimization",
            storage=f"sqlite:///{BASE_CONFIG['output_dir']}/optuna_study.db",
        )
    set_seed(seed)
    best_params = study.best_params

    tokenizer = AutoTokenizer.from_pretrained(BASE_CONFIG["model_name"])
    train_dataset, dev_dataset, prepared_test_dataset, _ = prepare_datasets(
        BASE_CONFIG["train_file"],
        BASE_CONFIG["dev_file"],
        BASE_CONFIG["test_file"],
        LABEL_LIST,
        tokenizer,
        max_length=best_params["max_length"],
    )
    if test_dataset is None:
        test_dataset = prepared_test_dataset

    config = AutoConfig.from_pretrained(BASE_CONFIG["model_name"])
    config.num_labels = len(LABEL_LIST)
    config.problem_type = "multi_label_classification"
    config.hidden_dropout_prob = best_params["dropout_prob"]
    config.attention_probs_dropout_prob = best_params["dropout_prob"]

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_CONFIG["model_name"],
        config=config,
        ignore_mismatched_sizes=True,
    )
    freeze_layers = best_params.get("freeze_layers", 0)
    if freeze_layers > 0:
        freeze_bottom_layers(model, num_layers_to_freeze=freeze_layers)

    final_output_dir = f"{BASE_CONFIG['output_dir']}/{output_suffix}"
    os.makedirs(final_output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=final_output_dir,
        num_train_epochs=best_params["num_epochs"],
        per_device_train_batch_size=best_params["batch_size"],
        per_device_eval_batch_size=best_params["batch_size"],
        gradient_accumulation_steps=best_params["gradient_accumulation_steps"],
        learning_rate=best_params["learning_rate"],
        warmup_ratio=BASE_CONFIG["warmup_ratio"],
        weight_decay=best_params.get("weight_decay", BASE_CONFIG["weight_decay"]),
        logging_dir=f"{final_output_dir}/logs",
        logging_steps=BASE_CONFIG["logging_steps"],
        eval_strategy="steps",
        eval_steps=BASE_CONFIG["eval_steps"],
        save_strategy="steps",
        save_steps=BASE_CONFIG["save_steps"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_comment_f1_macro",
        greater_is_better=True,
        fp16=BASE_CONFIG["fp16"],
        report_to="none",
        save_total_limit=BASE_CONFIG["save_total_limit"],
        remove_unused_columns=True,
        label_names=["comment_labels"],
    )

    def compute_metrics_fn(eval_pred: EvalPrediction):
        logits, labels = eval_pred
        return compute_comment_metrics(logits, labels, threshold=best_params["threshold"])

    trainer = CommentOnlyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics_fn,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.001,
            )
        ],
    )

    print("[Final] Training comment-only model...")
    trainer.train()

    dev_results = trainer.evaluate(dev_dataset)
    test_results = trainer.evaluate(test_dataset)

    with open(Path(final_output_dir) / "dev_results.json", "w") as f:
        json.dump({k: float(v) if isinstance(v, np.floating) else v for k, v in dev_results.items()}, f, indent=2)
    with open(Path(final_output_dir) / "test_results.json", "w") as f:
        json.dump({k: float(v) if isinstance(v, np.floating) else v for k, v in test_results.items()}, f, indent=2)
    with open(Path(final_output_dir) / "training_config.json", "w") as f:
        final_config = {**BASE_CONFIG, **best_params}
        json.dump(final_config, f, indent=2)

    tokenizer.save_pretrained(final_output_dir)
    trainer.save_model(final_output_dir)

    return trainer, test_results


def run_best_multiple_times(n_runs: int = 3, base_seed: int = 42):
    study = optuna.load_study(
        study_name="comment_only_optimization",
        storage=f"sqlite:///{BASE_CONFIG['output_dir']}/optuna_study.db",
    )
    run_metrics = []
    for i in range(n_runs):
        seed = base_seed + i
        suffix = f"final_model_run_{i}"
        _, test_results = train_with_best_params(
            study=study,
            test_dataset=None,
            seed=seed,
            output_suffix=suffix,
        )
        f1 = float(test_results.get("eval_comment_f1_macro", 0.0))
        run_metrics.append({"seed": seed, "eval_comment_f1_macro": f1})
        print(f"[Repeat Runs] Run {i} (seed={seed}) F1 Macro: {f1:.4f}")
    f1_values = [m["eval_comment_f1_macro"] for m in run_metrics]
    summary = {
        "runs": run_metrics,
        "mean_f1": float(np.mean(f1_values)),
        "std_f1": float(np.std(f1_values)),
        "n_runs": n_runs,
    }
    with open(f"{BASE_CONFIG['output_dir']}/final_runs_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("[Repeat Runs] Summary:", summary)
    return summary


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Comment-only multi-label finetuning with Optuna")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--n_jobs", type=int, default=1, help="Parallel jobs for Optuna")
    parser.add_argument("--train_final", action="store_true", help="Train final model after search")
    parser.add_argument("--only_final", action="store_true", help="Skip search and train final model from existing study")
    parser.add_argument("--repeat_best", type=int, default=0, help="Repeat best setting N times (train+eval) with different seeds")
    args = parser.parse_args()

    if args.only_final:
        if args.repeat_best > 0:
            run_best_multiple_times(n_runs=args.repeat_best)
        else:
            train_with_best_params()
    else:
        study = run_optimization(n_trials=args.n_trials, n_jobs=args.n_jobs)
        if args.train_final:
            train_with_best_params(study)
        if args.repeat_best > 0:
            run_best_multiple_times(n_runs=args.repeat_best)


if __name__ == "__main__":
    main()
