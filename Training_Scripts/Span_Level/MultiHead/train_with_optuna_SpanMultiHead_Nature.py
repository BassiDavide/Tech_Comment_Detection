import os
import json
import torch
import numpy as np
import optuna
import traceback
import math
from optuna.trial import Trial
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    Trainer,
    EvalPrediction,
    TrainerCallback,
    EarlyStoppingCallback,
    Adafactor,
    get_cosine_schedule_with_warmup,
    set_seed,
)

from data_processing import prepare_datasets
from model import MultiHeadTokenDeberta
from utils import (
    compute_token_metrics,
    compute_token_metrics_from_logits,
    compute_token_per_label_metrics,
    print_training_summary,
    compute_pos_weight_from_dataset,
)

# Shim for Python 3.8: ensure packages_distributions exists
import importlib.metadata as _im
if not hasattr(_im, "packages_distributions"):
    try:
        import importlib_metadata as _im_backport
        _im.packages_distributions = _im_backport.packages_distributions
    except Exception:
        _im.packages_distributions = lambda: {}


BASE_CONFIG = {
    "model_name": "/mnt/beegfs/home/davide.bassi/Comm_Tech/Domain_Adaptation/deberta-youtube-adapted-large",
    "num_token_labels": 20,
    "train_file": "/mnt/beegfs/home/davide.bassi/Comm_Tech/Data/Comments/Def_splits/train.jsonl",
    "dev_file": "/mnt/beegfs/home/davide.bassi/Comm_Tech/Data/Comments/Def_splits/dev.jsonl",
    "test_file": "/mnt/beegfs/home/davide.bassi/Comm_Tech/Data/Comments/Def_splits/test.jsonl",
    "output_dir": "/mnt/beegfs/home/davide.bassi/Comm_Tech/Nature/2Heads_MultiToken/Span_MultiHead/Span_MultiHead_optuna_results",
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "fp16": torch.cuda.is_available(),
    "logging_steps": 50,
    "eval_steps": 200,
    "save_steps": 200,
    "save_total_limit": 1,
}

TECHNIQUE_LABELS = [
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

LABEL_LIST = TECHNIQUE_LABELS + ["none"]
NONE_LABEL_INDEX = len(LABEL_LIST) - 1

BASE_CONFIG["num_token_labels"] = len(LABEL_LIST)


class TokenTrainerWrapper(Trainer):
    """Custom Trainer to handle token-only outputs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_loss_components = {}
        self._last_grad_norm = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        token_labels = inputs.pop("token_labels", None)
        if token_labels is None:
            raise ValueError("Missing token_labels in batch.")
        if torch.isnan(token_labels).any():
            raise ValueError("Found NaNs in token_labels")
        device = model.device if hasattr(model, "device") else next(model.parameters()).device
        token_labels = token_labels.to(device)
        outputs = model(**inputs, token_labels=token_labels)
        loss = outputs.get("loss", None)

        if not return_outputs and loss is None:
            raise ValueError("Model must return a loss when labels are provided.")

        with torch.no_grad():
            loss_components = {}
            if loss is not None:
                loss_components["loss"] = float(loss.detach().cpu().item())
            token_loss = outputs.get("token_loss")
            if token_loss is not None:
                loss_components["token_loss"] = float(token_loss.detach().cpu().item())
            if loss_components:
                self._last_loss_components = loss_components

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs):
        loss = super().training_step(model, inputs)
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            if torch.isnan(param.grad).any():
                raise ValueError(f"NaN gradient detected in parameter '{name}' at step {self.state.global_step}")
            grad_norms.append(torch.norm(param.grad.detach().float(), 2))
        if grad_norms:
            stacked = torch.stack(grad_norms)
            self._last_grad_norm = float(torch.norm(stacked, 2).detach().cpu().item())
        else:
            self._last_grad_norm = None
        return loss

    def log(self, logs):
        merged_logs = dict(logs)
        if getattr(self, "_last_loss_components", None):
            for key, value in self._last_loss_components.items():
                merged_logs.setdefault(key, value)
        if getattr(self, "_last_grad_norm", None) is not None:
            merged_logs.setdefault("grad_norm", self._last_grad_norm)
        super().log(merged_logs)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        has_labels = inputs.get("token_labels") is not None
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            outputs = model(**inputs)
            token_logits = outputs["token_logits"]
            loss = outputs.get("loss") if has_labels else None

        if prediction_loss_only:
            return (loss, None, None)

        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            raise ValueError("attention_mask is required for prediction")
        logits = token_logits.detach().cpu()

        if has_labels:
            labels = (
                inputs["token_labels"].detach().cpu(),
                attention_mask.detach().cpu(),
            )
        else:
            labels = None

        return (loss, logits, labels)


class NaNGradientCallback(TrainerCallback):
    """Abort training as soon as any gradient contains NaNs."""

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                raise ValueError(f"NaN gradient detected in parameter '{name}' at step {state.global_step}")


def freeze_bottom_layers(model, num_layers_to_freeze: int = 6):
    """
    Handles both plain DeBERTa and wrapper models.
    """
    encoder = None
    for attr in ["deberta", "encoder", "model"]:
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
        frozen_params = 0
        total_params = 0
        for param in model.parameters():
            total_params += 1
            if not param.requires_grad:
                frozen_params += 1
        percent = 100.0 * frozen_params / max(1, total_params)
        print(f"[LayerFreeze] Frozen bottom {num_layers_to_freeze} layers.")
        print(f"[LayerFreeze] {frozen_params}/{total_params} parameters frozen ({percent:.1f}%).")
    else:
        print("[LayerFreeze] Warning: could not access encoder layers. No freezing applied.")


def unfreeze_all_layers(model):
    for param in model.parameters():
        param.requires_grad = True
    print("[LayerFreeze] All layers unfrozen for full fine-tuning.")


class UnfreezeCallback(TrainerCallback):
    """
    Unfreeze encoder layers after a warmup period and keep track of epochs.
    """

    def __init__(self, unfreeze_epoch: int = 2):
        self.unfreeze_epoch = unfreeze_epoch
        self.unfrozen = False

    def _set_current_epoch(self, model, state):
        if model is None:
            return
        epoch_idx = 0 if state.epoch is None else int(state.epoch)
        setattr(model, "current_epoch", epoch_idx)

    def on_train_begin(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        self._set_current_epoch(model, state)

    def on_epoch_begin(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        self._set_current_epoch(model, state)

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        self._set_current_epoch(model, state)
        if self.unfrozen or state.epoch is None:
            return
        if state.epoch >= self.unfreeze_epoch:
            unfreeze_all_layers(model)
            self.unfrozen = True
            print(f"[LayerFreeze] Layers unfrozen at epoch {self.unfreeze_epoch}.")


class DelayedEarlyStoppingCallback(EarlyStoppingCallback):
    """
    Early stopping that waits for a minimum number of epochs before activating.
    """

    def __init__(self, min_epochs: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.min_epochs = max(0, min_epochs)

    def on_evaluate(self, args, state, control, **kwargs):
        epoch = state.epoch or 0
        if epoch + 1 < self.min_epochs:
            return control
        return super().on_evaluate(args, state, control, **kwargs)


def objective(trial: Trial, tokenizer):
    """
    Optuna objective function for multi-head span detection.
    """
    set_seed(42)
    max_length = trial.suggest_int("max_length", 256, 512, step=64)
    head_dim = trial.suggest_categorical("head_dim", [48, 64, 96])
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
    gradient_accumulation_steps = trial.suggest_categorical("gradient_accumulation_steps", [1, 2, 4])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 3e-5, log=True)
    num_epochs = trial.suggest_int("num_epochs", 20, 30, step=2)
    threshold = trial.suggest_float("threshold", 0.3, 0.7, step=0.05)
    dropout_prob = trial.suggest_float("dropout_prob", 0.05, 0.2, step=0.05)
    weight_decay = trial.suggest_float("weight_decay", 0.01, 0.05, step=0.01)

    trial_dir = f"{BASE_CONFIG['output_dir']}/trial_{trial.number}"
    os.makedirs(trial_dir, exist_ok=True)

    with open(f"{trial_dir}/trial_params.json", "w") as f:
        json.dump(trial.params, f, indent=2)

    print(f"\n{'='*80}")
    print(f"TRIAL {trial.number}")
    print(f"{'='*80}")
    print("Parameters:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")
    print(f"{'='*80}\n")

    train_dataset, dev_dataset, _, _ = prepare_datasets(
        BASE_CONFIG["train_file"],
        BASE_CONFIG["dev_file"],
        BASE_CONFIG["test_file"],
        LABEL_LIST,
        tokenizer,
        max_length=max_length,
    )
    pos_weight = compute_pos_weight_from_dataset(
        train_dataset,
        BASE_CONFIG["num_token_labels"],
        field="token_labels",
    )

    config = AutoConfig.from_pretrained(BASE_CONFIG["model_name"])
    config.hidden_dropout_prob = dropout_prob
    config.attention_probs_dropout_prob = dropout_prob

    model = MultiHeadTokenDeberta(
        config=config,
        num_token_labels=BASE_CONFIG["num_token_labels"],
        head_dim=head_dim,
        pretrained_model_name=BASE_CONFIG["model_name"],
    )
    model.set_pos_weight(pos_weight)
    if model.pos_weight is not None:
        print("Token pos_weight:", model.pos_weight.detach().cpu())
    freeze_bottom_layers(model, num_layers_to_freeze=6)

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
        save_total_limit=BASE_CONFIG.get("save_total_limit", 2),
        load_best_model_at_end=True,
        metric_for_best_model="eval_token_f1_macro",
        greater_is_better=True,
        fp16=False,
        max_grad_norm=0.5,
        report_to="none",
        disable_tqdm=False,
        remove_unused_columns=False,
    )

    optimizer = Adafactor(
        model.parameters(),
        lr=learning_rate,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        weight_decay=weight_decay,
    )
    steps_per_epoch = max(
        1, math.ceil(len(train_dataset) / (batch_size * gradient_accumulation_steps))
    )
    num_training_steps = max(1, steps_per_epoch * num_epochs)
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    def compute_metrics_fn(eval_pred: EvalPrediction):
        return compute_token_metrics(eval_pred, threshold=threshold, ignore_label_indices=NONE_LABEL_INDEX)

    trainer = TokenTrainerWrapper(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics_fn,
        optimizers=(optimizer, scheduler),
        callbacks=[
            NaNGradientCallback(),
            DelayedEarlyStoppingCallback(
                min_epochs=max(5, num_epochs // 2),
                early_stopping_patience=3,
                early_stopping_threshold=0.001,
            ),
        ],
    )
    trainer.add_callback(UnfreezeCallback())

    try:
        trainer.train()
        best_checkpoint = getattr(trainer.state, "best_model_checkpoint", None)
        best_model_dir = Path(trial_dir) / "best_model"
        best_model_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(best_model_dir)
        tokenizer.save_pretrained(best_model_dir)
        if best_checkpoint:
            with open(Path(trial_dir) / "best_checkpoint.txt", "w") as f:
                f.write(str(best_checkpoint))
            print(f"Best checkpoint saved at: {best_checkpoint}")
        else:
            print("Warning: trainer.state.best_model_checkpoint missing; saved current model as best_model.")

        eval_results = trainer.evaluate()

        with open(f"{trial_dir}/eval_results.json", "w") as f:
            results_serializable = {
                k: float(v) if isinstance(v, np.floating) else v for k, v in eval_results.items()
            }
            json.dump(results_serializable, f, indent=2)

        return eval_results["eval_token_f1_macro"]

    except Exception as e:
        traceback.print_exc()
        print(f"Trial {trial.number} failed with error: {e}")
        return 0.0


def run_optimization(n_trials=20, n_jobs=1):
    """
    Run Optuna optimization for multi-head span detection.
    """
    print("=" * 80)
    print("MULTI-HEAD SPAN OPTIMIZATION WITH OPTUNA")
    print("=" * 80)
    print("\nBase Configuration:")
    for key, value in BASE_CONFIG.items():
        print(f"  {key}: {value}")
    print("\nOptimization Settings:")
    print(f"  Number of trials: {n_trials}")
    print(f"  Number of jobs: {n_jobs}")
    print("  Optimization metric: token_f1_macro (excluding none)")
    print("=" * 80 + "\n")

    os.makedirs(BASE_CONFIG["output_dir"], exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_CONFIG["model_name"])
    print("\nDatasets will be rebuilt inside each trial to reflect the sampled max_length.\n")

    study = optuna.create_study(
        direction="maximize",
        study_name="span_multihead_optimization",
        storage=f"sqlite:///{BASE_CONFIG['output_dir']}/optuna_study.db",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )

    print("\nStarting optimization...\n")
    study.optimize(
        lambda trial: objective(trial, tokenizer),
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True,
    )

    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETED")
    print("=" * 80)

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best F1 Macro: {study.best_value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    with open(f"{BASE_CONFIG['output_dir']}/best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)

    print(f"\n{'='*80}")
    print("TOP 5 TRIALS")
    print(f"{'='*80}")

    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values("value", ascending=False)
    for _, row in trials_df.head(5).iterrows():
        print(f"\nTrial {int(row['number'])}:")
        print(f"  F1 Macro: {row['value']:.4f}")
        print(f"  State: {row['state']}")
        param_cols = [col for col in trials_df.columns if col.startswith("params_")]
        for col in param_cols:
            param_name = col.replace("params_", "")
            print(f"  {param_name}: {row[col]}")

    trials_df.to_csv(f"{BASE_CONFIG['output_dir']}/all_trials.csv", index=False)

    try:
        import optuna.visualization as vis

        fig = vis.plot_optimization_history(study)
        fig.write_html(f"{BASE_CONFIG['output_dir']}/optimization_history.html")

        fig = vis.plot_param_importances(study)
        fig.write_html(f"{BASE_CONFIG['output_dir']}/param_importances.html")

        fig = vis.plot_parallel_coordinate(study)
        fig.write_html(f"{BASE_CONFIG['output_dir']}/parallel_coordinate.html")

        print(f"\nVisualization plots saved to {BASE_CONFIG['output_dir']}/")
    except ImportError:
        print("\nInstall plotly to generate visualization plots: pip install plotly")

    print(f"\n{'='*80}")
    print(f"Results saved to: {BASE_CONFIG['output_dir']}")
    print("  - best_params.json: Best hyperparameters")
    print("  - all_trials.csv: All trial results")
    print("  - optuna_study.db: Optuna study database")
    print("  - trial_*/: Individual trial outputs")
    print(f"{'='*80}\n")

    return study


def _read_best_from_trainer_state(trainer_state_path: Path):
    """
    Read best_model_checkpoint from a trainer_state.json if present and valid.
    """
    if not trainer_state_path.exists():
        return None
    try:
        with open(trainer_state_path) as f:
            trainer_state = json.load(f)
        best_checkpoint = trainer_state.get("best_model_checkpoint")
        if best_checkpoint:
            best_path = Path(best_checkpoint)
            if not best_path.is_absolute():
                best_path = trainer_state_path.parent / best_path
            if best_path.exists():
                return best_path
    except Exception as exc:
        print(f"[Checkpoint] Failed to read {trainer_state_path}: {exc}")
    return None


def get_best_checkpoint_path(trial_dir: Path) -> Path:
    """
    Prefer the explicitly saved best model, otherwise fall back to trainer_state,
    and finally to the last checkpoint.
    """
    best_model_dir = trial_dir / "best_model"
    if best_model_dir.exists():
        return best_model_dir
    best_from_root_state = _read_best_from_trainer_state(trial_dir / "trainer_state.json")
    if best_from_root_state:
        return best_from_root_state

    checkpoint_dirs = sorted(
        [path for path in trial_dir.iterdir() if path.is_dir() and path.name.startswith("checkpoint-")],
        key=lambda p: int(p.name.split("-")[1]),
    )
    if not checkpoint_dirs:
        raise FileNotFoundError(
            f"No checkpoints found under {trial_dir}. "
            "Ensure the Optuna trial saved its trainer state."
        )
    for ckpt in reversed(checkpoint_dirs):
        best_from_ckpt = _read_best_from_trainer_state(ckpt / "trainer_state.json")
        if best_from_ckpt:
            return best_from_ckpt
    print("[Checkpoint] Falling back to latest checkpoint.")
    return checkpoint_dirs[-1]


def tune_token_threshold_on_dev(trainer, dev_dataset, base_threshold: float, grid=None):
    """
    Find the token threshold that maximizes F1 macro on the dev set.
    """
    if grid is None:
        grid = [round(x, 2) for x in np.arange(0.30, 0.71, 0.02)]
    try:
        preds = trainer.predict(dev_dataset)
        token_logits = preds.predictions
        label_ids = preds.label_ids
        if isinstance(label_ids, (list, tuple)) and len(label_ids) == 2:
            token_labels, attention_mask = label_ids
        else:
            token_labels = label_ids
            attention_mask = None
    except Exception as exc:
        print(f"[Threshold] Dev predictions failed, fallback to base threshold {base_threshold}: {exc}")
        return base_threshold, None

    best_thr = base_threshold
    best_f1 = -1.0
    for thr in grid:
        metrics = compute_token_metrics_from_logits(
            token_logits,
            token_labels,
            attention_mask,
            threshold=thr,
            ignore_label_indices=NONE_LABEL_INDEX,
        )
        f1 = metrics.get("token_f1_macro", 0.0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return best_thr, best_f1


def evaluate_best_trial_on_test():
    """
    Evaluate the best Optuna trial checkpoint on the test set without retraining.
    """
    print("\n" + "=" * 80)
    print("EVALUATING BEST OPTUNA TRIAL ON TEST SET (MULTI-HEAD)")
    print("=" * 80)

    study = optuna.load_study(
        study_name="span_multihead_optimization",
        storage=f"sqlite:///{BASE_CONFIG['output_dir']}/optuna_study.db",
    )
    best_trial = study.best_trial

    print(f"\nBest trial: {best_trial.number}")
    print(f"Best F1 Macro (dev): {study.best_value:.4f}")

    trial_dir = Path(BASE_CONFIG["output_dir"]) / f"trial_{best_trial.number}"
    if not trial_dir.exists():
        raise FileNotFoundError(f"Expected directory {trial_dir} for best trial checkpoint.")

    best_checkpoint = get_best_checkpoint_path(trial_dir)
    print(f"Using checkpoint: {best_checkpoint}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_CONFIG["model_name"])
    max_length = best_trial.params.get("max_length", 512)
    train_dataset, dev_dataset, test_dataset, _ = prepare_datasets(
        BASE_CONFIG["train_file"],
        BASE_CONFIG["dev_file"],
        BASE_CONFIG["test_file"],
        LABEL_LIST,
        tokenizer,
        max_length=max_length,
    )
    pos_weight = compute_pos_weight_from_dataset(
        train_dataset,
        BASE_CONFIG["num_token_labels"],
        field="token_labels",
    )

    config = AutoConfig.from_pretrained(best_checkpoint)
    model = MultiHeadTokenDeberta.from_pretrained(
        best_checkpoint,
        config=config,
        num_token_labels=BASE_CONFIG["num_token_labels"],
        head_dim=best_trial.params.get("head_dim", 64),
    )
    model.set_pos_weight(pos_weight)

    eval_output_dir = trial_dir / "test_eval"
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    eval_args = TrainingArguments(
        output_dir=str(eval_output_dir),
        per_device_eval_batch_size=best_trial.params["batch_size"],
        report_to="none",
        remove_unused_columns=False,
    )

    threshold = best_trial.params["threshold"]
    trainer = TokenTrainerWrapper(
        model=model,
        args=eval_args,
        eval_dataset=test_dataset,
        compute_metrics=lambda pred: compute_token_metrics(
            pred,
            threshold=threshold,
            ignore_label_indices=NONE_LABEL_INDEX,
        ),
    )

    tuned_threshold, tuned_dev_f1 = tune_token_threshold_on_dev(
        trainer, dev_dataset, base_threshold=threshold
    )
    if tuned_dev_f1 is not None:
        print(f"Tuned threshold on dev: {tuned_threshold:.2f} (F1_macro={tuned_dev_f1:.3f})")
    else:
        print(f"Using base threshold {tuned_threshold:.2f}")
    threshold = tuned_threshold
    trainer.compute_metrics = lambda pred: compute_token_metrics(
        pred,
        threshold=threshold,
        ignore_label_indices=NONE_LABEL_INDEX,
    )

    print("\nRunning test evaluation...")
    test_results = trainer.evaluate(test_dataset)
    print_training_summary(test_results, f"Trial {best_trial.number}", mode="test")

    with open(eval_output_dir / "test_results.json", "w") as f:
        results_serializable = {
            k: float(v) if isinstance(v, np.floating) else v for k, v in test_results.items()
        }
        json.dump(results_serializable, f, indent=2)

    test_predictions = trainer.predict(test_dataset)
    token_logits = test_predictions.predictions
    label_ids = test_predictions.label_ids
    if isinstance(label_ids, (list, tuple)) and len(label_ids) == 2:
        token_labels, attention_mask = label_ids
    else:
        token_labels = label_ids
        attention_mask = None

    per_label_metrics = compute_token_per_label_metrics(
        token_logits,
        token_labels,
        attention_mask,
        label_list=LABEL_LIST,
        threshold=threshold,
        ignore_label_indices=NONE_LABEL_INDEX,
    )

    with open(eval_output_dir / "test_token_per_label.json", "w") as f:
        json.dump(per_label_metrics, f, indent=2)

    with open(eval_output_dir / "label_list.json", "w") as f:
        json.dump(LABEL_LIST, f, indent=2)

    print(f"\nTest evaluation artifacts saved to: {eval_output_dir}")
    print("=" * 80 + "\n")


def train_with_best_params(study=None, test_dataset=None, seed: int = 42, output_suffix: str = "final_model"):
    """
    Train final model with best parameters from optimization.
    """
    if study is None:
        study = optuna.load_study(
            study_name="span_multihead_optimization",
            storage=f"sqlite:///{BASE_CONFIG['output_dir']}/optuna_study.db",
        )
    set_seed(seed)
    best_params = study.best_params

    print("\n" + "=" * 80)
    print("TRAINING FINAL MULTI-HEAD MODEL WITH BEST PARAMETERS")
    print("=" * 80)
    print(f"\nBest parameters (F1 Macro: {study.best_value:.4f}):")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print("=" * 80 + "\n")

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
    pos_weight = compute_pos_weight_from_dataset(
        train_dataset,
        BASE_CONFIG["num_token_labels"],
        field="token_labels",
    )

    config = AutoConfig.from_pretrained(BASE_CONFIG["model_name"])
    config.hidden_dropout_prob = best_params["dropout_prob"]
    config.attention_probs_dropout_prob = best_params["dropout_prob"]

    model = MultiHeadTokenDeberta(
        config=config,
        num_token_labels=BASE_CONFIG["num_token_labels"],
        head_dim=best_params.get("head_dim", 64),
        pretrained_model_name=BASE_CONFIG["model_name"],
    )
    model.set_pos_weight(pos_weight)
    freeze_bottom_layers(model, num_layers_to_freeze=6)

    final_output_dir = f"{BASE_CONFIG['output_dir']}/{output_suffix}"
    final_weight_decay = best_params.get("weight_decay", BASE_CONFIG["weight_decay"])
    training_args = TrainingArguments(
        output_dir=final_output_dir,
        num_train_epochs=best_params["num_epochs"],
        per_device_train_batch_size=best_params["batch_size"],
        per_device_eval_batch_size=best_params["batch_size"],
        gradient_accumulation_steps=best_params["gradient_accumulation_steps"],
        learning_rate=best_params["learning_rate"],
        warmup_ratio=BASE_CONFIG["warmup_ratio"],
        weight_decay=final_weight_decay,
        logging_dir=f"{final_output_dir}/logs",
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_token_f1_macro",
        greater_is_better=True,
        fp16=best_params.get("fp16", False),
        report_to="none",
        save_total_limit=BASE_CONFIG.get("save_total_limit", 1),
        max_grad_norm=0.5,
        remove_unused_columns=False,
    )
    optimizer = Adafactor(
        model.parameters(),
        lr=best_params["learning_rate"],
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        weight_decay=final_weight_decay,
    )
    final_steps_per_epoch = max(
        1,
        math.ceil(
            len(train_dataset)
            / (best_params["batch_size"] * best_params["gradient_accumulation_steps"])
        ),
    )
    final_num_training_steps = max(1, final_steps_per_epoch * best_params["num_epochs"])
    final_num_warmup_steps = int(0.1 * final_num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=final_num_warmup_steps,
        num_training_steps=final_num_training_steps,
    )

    def compute_metrics_fn(eval_pred: EvalPrediction):
        return compute_token_metrics(
            eval_pred,
            threshold=best_params["threshold"],
            ignore_label_indices=NONE_LABEL_INDEX,
        )

    trainer = TokenTrainerWrapper(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics_fn,
        optimizers=(optimizer, scheduler),
        callbacks=[
            NaNGradientCallback(),
            DelayedEarlyStoppingCallback(
                min_epochs=max(5, best_params["num_epochs"] // 2),
                early_stopping_patience=3,
                early_stopping_threshold=0.001,
            ),
        ],
    )
    trainer.add_callback(UnfreezeCallback())

    print("Training final model...")
    trainer.train()
    tuned_threshold, tuned_dev_f1 = tune_token_threshold_on_dev(
        trainer, dev_dataset, base_threshold=best_params["threshold"]
    )
    if tuned_dev_f1 is not None:
        print(f"[Threshold] Tuned on dev: {tuned_threshold:.2f} (F1_macro={tuned_dev_f1:.3f})")
    else:
        print(f"[Threshold] Using base threshold {tuned_threshold:.2f}")
    trainer.compute_metrics = lambda pred: compute_token_metrics(
        pred,
        threshold=tuned_threshold,
        ignore_label_indices=NONE_LABEL_INDEX,
    )

    print("\nSaving final model...")
    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)

    with open(f"{final_output_dir}/label_list.json", "w") as f:
        json.dump(LABEL_LIST, f, indent=2)

    final_config = {**BASE_CONFIG, **best_params}
    final_config["fp16"] = best_params.get("fp16", False)
    with open(f"{final_output_dir}/training_config.json", "w") as f:
        json.dump(final_config, f, indent=2)

    print("\n" + "=" * 80)
    print("EVALUATING ON TEST SET")
    print("=" * 80)

    test_results = trainer.evaluate(test_dataset)
    print_training_summary(test_results, "Final", mode="test")

    with open(f"{final_output_dir}/test_results.json", "w") as f:
        results_serializable = {
            k: float(v) if isinstance(v, np.floating) else v for k, v in test_results.items()
        }
        json.dump(results_serializable, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Final model saved to: {final_output_dir}")
    print(f"{'='*80}\n")

    return trainer, test_results


def run_best_multiple_times(n_runs: int = 3, base_seed: int = 42):
    """
    Re-train/evaluate the best Optuna setting multiple times to report mean/std.
    """
    study = optuna.load_study(
        study_name="span_multihead_optimization",
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
        f1 = float(test_results.get("eval_token_f1_macro", 0.0))
        run_metrics.append({"seed": seed, "eval_token_f1_macro": f1})
        print(f"[Repeat Runs] Run {i} (seed={seed}) F1 Macro: {f1:.4f}")
    f1_values = [m["eval_token_f1_macro"] for m in run_metrics]
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-head span optimization for propaganda detection")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of optimization trials")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs")
    parser.add_argument("--train_final", action="store_true", help="Train final model with best params after optimization")
    parser.add_argument("--only_final", action="store_true", help="Skip optimization and only train final model")
    parser.add_argument("--eval_test_only", action="store_true", help="Evaluate best Optuna trial on test set")
    parser.add_argument("--repeat_best", type=int, default=0, help="Repeat best setting N times (train+eval) with different seeds")

    args = parser.parse_args()

    if args.only_final:
        if args.repeat_best > 0:
            run_best_multiple_times(n_runs=args.repeat_best)
        else:
            train_with_best_params()
    elif args.eval_test_only:
        evaluate_best_trial_on_test()
        if args.repeat_best > 0:
            run_best_multiple_times(n_runs=args.repeat_best)
    else:
        study = run_optimization(n_trials=args.n_trials, n_jobs=args.n_jobs)
        if args.train_final:
            train_with_best_params(study)
        evaluate_best_trial_on_test()
        if args.repeat_best > 0:
            run_best_multiple_times(n_runs=args.repeat_best)
