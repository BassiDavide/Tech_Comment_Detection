import os
import json
import torch
import pickle
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

from Training_Scripts.Multi_Token.data_processing import prepare_datasets
from model import MultitaskDeberta
from utils import (
    compute_multitask_metrics,
    print_training_summary,
    compute_comment_per_label_metrics,
    compute_pos_weight_from_dataset,
)

BASE_CONFIG = {
    #"model_name": "microsoft/deberta-v3-large",
    "model_name": "Domain_Adaptation/deberta-youtube-adapted-large",
    "num_comment_labels": 20,
    "num_token_labels": 20,
    
    # Data paths
    "train_file": "train.jsonl",
    "dev_file": "dev.jsonl",
    "test_file": "test.jsonl",

    # Output
    "output_dir": "./MultiToken_DEF_optuna_results",
    
    # Fixed training settings
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "fp16": torch.cuda.is_available(),
    "logging_steps": 50,
    "eval_steps": 200,
    "save_steps": 200,
    "save_total_limit": 2,
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

BASE_CONFIG["num_comment_labels"] = len(LABEL_LIST)
BASE_CONFIG["num_token_labels"] = len(LABEL_LIST)


class MultitaskTrainerWrapper(Trainer):
    """Custom Trainer to handle multi-task model outputs"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_loss_components = {}
        self._last_grad_norm = None
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Custom loss that explicitly passes labels to the model"""
        comment_labels = inputs.pop("comment_labels", None)
        token_labels = inputs.pop("token_labels", None)
        if comment_labels is None or token_labels is None:
            missing = [k for k in ("comment_labels", "token_labels") if inputs.get(k) is None]
            raise ValueError(f"Missing labels in batch: {missing}")
        if torch.isnan(comment_labels).any() or torch.isnan(token_labels).any():
            raise ValueError("Found NaNs in labels")
        device = model.device if hasattr(model, 'device') else next(model.parameters()).device
        comment_labels = comment_labels.to(device)
        token_labels = token_labels.to(device)
        outputs = model(**inputs, comment_labels=comment_labels, token_labels=token_labels)
        loss = outputs.get('loss', None)

        if not return_outputs and loss is None:
            raise ValueError("Model must return a loss when labels are provided.")

        with torch.no_grad():
            loss_components = {}
            if loss is not None:
                loss_components["loss"] = float(loss.detach().cpu().item())
            comment_loss = outputs.get("comment_loss")
            if comment_loss is not None:
                loss_components["comment_loss"] = float(comment_loss.detach().cpu().item())
            token_loss = outputs.get("token_loss")
            if token_loss is not None:
                loss_components["token_loss"] = float(token_loss.detach().cpu().item())
            if loss_components:
                self._last_loss_components = loss_components

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Run a training step and inspect gradients for NaNs right after backward."""
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
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
        """Custom prediction step to handle multi-task outputs"""
        has_labels = all(inputs.get(k) is not None for k in ["comment_labels", "token_labels"])
        inputs = self._prepare_inputs(inputs)
        
        with torch.no_grad():
            if has_labels:
                outputs = model(**inputs)
                loss = outputs['loss']
                comment_logits = outputs['comment_logits']
                token_logits = outputs['token_logits']
            else:
                loss = None
                outputs = model(**inputs)
                comment_logits = outputs['comment_logits']
                token_logits = outputs['token_logits']
        
        if prediction_loss_only:
            return (loss, None, None)
        
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            raise ValueError("attention_mask is required for prediction")
        logits = (
            comment_logits.detach().cpu(),
            token_logits.detach().cpu(),
        )
        
        if has_labels:
            labels = (
                inputs['comment_labels'].detach().cpu(),
                inputs['token_labels'].detach().cpu(),
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
    Handles both plain DeBERTa and multitask wrappers.
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
        print(f"✅ Frozen bottom {num_layers_to_freeze} layers.")
        print(f"🔒 {frozen_params}/{total_params} parameters frozen initially ({percent:.1f}%)")
    else:
        print("[LayerFreeze] Warning: could not access encoder layers. No freezing applied.")


def unfreeze_all_layers(model):
    for param in model.parameters():
        param.requires_grad = True
    print("[LayerFreeze] All layers unfrozen for full fine-tuning.")


class UnfreezeCallback(TrainerCallback):
    """
    Unfreeze encoder layers after a warmup period and keep track of epochs for pooling warmup.
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
    Optuna objective function to optimize
    """
    max_length = trial.suggest_int("max_length", 256, 512, step=64)
    alpha = trial.suggest_float("alpha", 4.0, 5.0, step=0.5)
    beta = trial.suggest_float("beta", 1.0, 2.0, step=0.1)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
    gradient_accumulation_steps = trial.suggest_categorical("gradient_accumulation_steps", [1, 2, 4])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 3e-5, log=True)
    num_epochs = trial.suggest_int("num_epochs", 20, 30, step=2)
    threshold = trial.suggest_float("threshold", 0.1, 0.3, step=0.05)
    dropout_prob = trial.suggest_float("dropout_prob", 0.05, 0.2, step=0.05)
    weight_decay = trial.suggest_float("weight_decay", 0.01, 0.05, step=0.01)
    token_pooling_ratio = trial.suggest_float("token_pooling_ratio", 0.2, 0.5, step=0.05)
    
    trial_dir = f"{BASE_CONFIG['output_dir']}/trial_{trial.number}"
    os.makedirs(trial_dir, exist_ok=True)
    
    with open(f"{trial_dir}/trial_params.json", 'w') as f:
        json.dump(trial.params, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"TRIAL {trial.number}")
    print(f"{'='*80}")
    print(f"Parameters:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")
    print(f"{'='*80}\n")

    train_dataset, dev_dataset, _, _ = prepare_datasets(
        BASE_CONFIG['train_file'],
        BASE_CONFIG['dev_file'],
        BASE_CONFIG['test_file'],
        LABEL_LIST,
        tokenizer,
        max_length=max_length,
    )
    pos_weight = compute_pos_weight_from_dataset(
        train_dataset,
        BASE_CONFIG['num_token_labels'],
        field="token_labels",
    )

    config = AutoConfig.from_pretrained(BASE_CONFIG['model_name'])
    config.hidden_dropout_prob = dropout_prob
    config.attention_probs_dropout_prob = dropout_prob
    config.token_loss_strategy = "balanced_per_label"
    config.token_loss_max_pos_weight = 5.0
    config.token_loss_eps = 1e-8
    
    model = MultitaskDeberta(
        config=config,
        num_comment_labels=BASE_CONFIG['num_comment_labels'],
        num_token_labels=BASE_CONFIG['num_token_labels'],
        alpha=alpha,
        beta=beta,
        pretrained_model_name=BASE_CONFIG['model_name'],
        token_pooling_ratio=token_pooling_ratio,
        pooling_warmup_epochs=2,
    )
    model.set_pos_weight(pos_weight)
    if model.pos_weight is not None:
        print("Token pos_weight:", model.pos_weight.detach().cpu())
    freeze_bottom_layers(model, num_layers_to_freeze=6)
    
    # Training arguments
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
        metric_for_best_model="eval_comment_f1_macro",
        greater_is_better=True,

        
        fp16=False,
        max_grad_norm=0.5,
        report_to="none",
        disable_tqdm=False,
        remove_unused_columns=False,
    )

    # Initialize Adafactor optimizer and cosine scheduler
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

    
    # Compute metrics function
    def compute_metrics_fn(eval_pred: EvalPrediction):
        return compute_multitask_metrics(eval_pred, threshold=threshold)
    
    # Initialize trainer
    trainer = MultitaskTrainerWrapper(
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
    
    # Train
    try:
        trainer.train()
        
        # Evaluate
        eval_results = trainer.evaluate()
        
        # Save results
        with open(f"{trial_dir}/eval_results.json", 'w') as f:
            results_serializable = {k: float(v) if isinstance(v, np.floating) else v 
                                   for k, v in eval_results.items()}
            json.dump(results_serializable, f, indent=2)
        
        # Return the metric we want to optimize
        return eval_results['eval_comment_f1_macro']
    
    except Exception as e:
        traceback.print_exc()
        print(f"Trial {trial.number} failed with error: {e}")
        return 0.0  # Return worst possible score


def run_optimization(n_trials=20, n_jobs=1):
    """
    Run Optuna optimization
    
    Args:
        n_trials: Number of trials to run
        n_jobs: Number of parallel jobs (set to 1 for GPU training)
    """
    print("="*80)
    print("MULTI-TASK HYPERPARAMETER OPTIMIZATION WITH OPTUNA")
    print("="*80)
    print(f"\nBase Configuration:")
    for key, value in BASE_CONFIG.items():
        print(f"  {key}: {value}")
    print(f"\nOptimization Settings:")
    print(f"  Number of trials: {n_trials}")
    print(f"  Number of jobs: {n_jobs}")
    print(f"  Optimization metric: comment_f1_macro")
    print("="*80 + "\n")
    
    # Create output directory
    os.makedirs(BASE_CONFIG['output_dir'], exist_ok=True)
    
    # Load tokenizer
    print("Loading tokenizer...")
    #tokenizer = DebertaV2TokenizerFast.from_pretrained(BASE_CONFIG['model_name'])
    tokenizer = AutoTokenizer.from_pretrained(BASE_CONFIG['model_name'])

    print("\nDatasets will be rebuilt inside each trial to reflect the sampled max_length.\n")
    
    # Create Optuna study
    study = optuna.create_study(
        direction="maximize",
        study_name="multitask_deberta_optimization",
        storage=f"sqlite:///{BASE_CONFIG['output_dir']}/optuna_study.db",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    )
    
    # Run optimization
    print("\nStarting optimization...\n")
    study.optimize(
        lambda trial: objective(trial, tokenizer),
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True
    )
    
    # Print results
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETED")
    print("="*80)
    
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best F1 Macro: {study.best_value:.4f}")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save best parameters
    with open(f"{BASE_CONFIG['output_dir']}/best_params.json", 'w') as f:
        json.dump(study.best_params, f, indent=2)
    
    # Print top 5 trials
    print(f"\n{'='*80}")
    print("TOP 5 TRIALS")
    print(f"{'='*80}")
    
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values('value', ascending=False)
    
    for idx, row in trials_df.head(5).iterrows():
        print(f"\nTrial {int(row['number'])}:")
        print(f"  F1 Macro: {row['value']:.4f}")
        print(f"  State: {row['state']}")
        param_cols = [col for col in trials_df.columns if col.startswith('params_')]
        for col in param_cols:
            param_name = col.replace('params_', '')
            print(f"  {param_name}: {row[col]}")
    
    # Save study results
    trials_df.to_csv(f"{BASE_CONFIG['output_dir']}/all_trials.csv", index=False)
    
    # Create visualization plots
    try:
        import optuna.visualization as vis
        
        # Optimization history
        fig = vis.plot_optimization_history(study)
        fig.write_html(f"{BASE_CONFIG['output_dir']}/optimization_history.html")
        
        # Parameter importance
        fig = vis.plot_param_importances(study)
        fig.write_html(f"{BASE_CONFIG['output_dir']}/param_importances.html")
        
        # Parallel coordinate plot
        fig = vis.plot_parallel_coordinate(study)
        fig.write_html(f"{BASE_CONFIG['output_dir']}/parallel_coordinate.html")
        
        print(f"\nVisualization plots saved to {BASE_CONFIG['output_dir']}/")
    except ImportError:
        print("\nInstall plotly to generate visualization plots: pip install plotly")
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {BASE_CONFIG['output_dir']}")
    print(f"  - best_params.json: Best hyperparameters")
    print(f"  - all_trials.csv: All trial results")
    print(f"  - optuna_study.db: Optuna study database")
    print(f"  - trial_*/: Individual trial outputs")
    print(f"{'='*80}\n")
    
    return study


def evaluate_best_trial_on_test():
    """
    Evaluate the best Optuna trial checkpoint on the test set without retraining.
    """
    print("\n" + "=" * 80)
    print("EVALUATING BEST OPTUNA TRIAL ON TEST SET")
    print("=" * 80)

    study = optuna.load_study(
        study_name="multitask_deberta_optimization",
        storage=f"sqlite:///{BASE_CONFIG['output_dir']}/optuna_study.db",
    )
    best_trial = study.best_trial

    print(f"\nBest trial: {best_trial.number}")
    print(f"Best F1 Macro (dev): {study.best_value:.4f}")

    trial_dir = Path(BASE_CONFIG['output_dir']) / f"trial_{best_trial.number}"
    if not trial_dir.exists():
        raise FileNotFoundError(f"Expected directory {trial_dir} for best trial checkpoint.")

    checkpoint_dirs = sorted(
        [path for path in trial_dir.iterdir() if path.is_dir() and path.name.startswith("checkpoint-")],
        key=lambda p: int(p.name.split("-")[1])
    )
    if not checkpoint_dirs:
        raise FileNotFoundError(
            f"No checkpoints found under {trial_dir}. "
            "Ensure the Optuna trial saved its trainer state."
        )
    best_checkpoint = checkpoint_dirs[-1]
    print(f"Using checkpoint: {best_checkpoint.name}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_CONFIG['model_name'])
    max_length = best_trial.params.get("max_length", 512)
    train_dataset, _, test_dataset, _ = prepare_datasets(
        BASE_CONFIG['train_file'],
        BASE_CONFIG['dev_file'],
        BASE_CONFIG['test_file'],
        LABEL_LIST,
        tokenizer,
        max_length=max_length
    )
    pos_weight = compute_pos_weight_from_dataset(
        train_dataset,
        BASE_CONFIG['num_token_labels'],
        field="token_labels",
    )

    config = AutoConfig.from_pretrained(best_checkpoint)
    model = MultitaskDeberta.from_pretrained(
        best_checkpoint,
        config=config,
        num_comment_labels=BASE_CONFIG['num_comment_labels'],
        num_token_labels=BASE_CONFIG['num_token_labels'],
        alpha=best_trial.params['alpha'],
        beta=best_trial.params['beta'],
        token_pooling_ratio=best_trial.params.get('token_pooling_ratio', 0.0),
        pooling_warmup_epochs=2,
    )
    model.set_pos_weight(pos_weight)
    if model.pos_weight is not None:
        print("Token pos_weight:", model.pos_weight.detach().cpu())

    eval_output_dir = trial_dir / "test_eval"
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    eval_args = TrainingArguments(
        output_dir=str(eval_output_dir),
        per_device_eval_batch_size=best_trial.params['batch_size'],
        report_to="none"
    )

    threshold = best_trial.params['threshold']

    trainer = MultitaskTrainerWrapper(
        model=model,
        args=eval_args,
        eval_dataset=test_dataset,
        compute_metrics=lambda pred: compute_multitask_metrics(pred, threshold=threshold),
    )

    print("\nRunning test evaluation...")
    test_results = trainer.evaluate(test_dataset)
    print_training_summary(test_results, f"Trial {best_trial.number}", mode="test")

    with open(eval_output_dir / "test_results.json", 'w') as f:
        results_serializable = {
            k: float(v) if isinstance(v, np.floating) else v
            for k, v in test_results.items()
        }
        json.dump(results_serializable, f, indent=2)

    print("Generating test predictions...")
    test_predictions = trainer.predict(test_dataset)

    prediction_tensors = test_predictions.predictions
    comment_logits = prediction_tensors[0]
    token_logits = prediction_tensors[1]
    comment_probs = torch.sigmoid(torch.from_numpy(comment_logits)).numpy()
    comment_preds = (comment_probs > threshold).astype(int)

    token_probs = torch.sigmoid(torch.from_numpy(token_logits)).numpy()
    token_preds = (token_probs > 0.5).astype(int)

    np.save(eval_output_dir / "test_comment_probs.npy", comment_probs)
    np.save(eval_output_dir / "test_comment_predictions.npy", comment_preds)
    np.save(eval_output_dir / "test_token_probs.npy", token_probs)
    np.save(eval_output_dir / "test_token_predictions.npy", token_preds)

    per_label_metrics = compute_comment_per_label_metrics(
        comment_logits,
        test_dataset.comment_labels,
        threshold=threshold,
        label_list=LABEL_LIST
    )

    with open(eval_output_dir / "test_comment_per_label.json", 'w') as f:
        json.dump(per_label_metrics, f, indent=2)

    print("\nTechnique-level comment metrics (sorted by F1):")
    for technique, stats in sorted(
        per_label_metrics.items(),
        key=lambda item: item[1]['f1'],
        reverse=True
    ):
        print(
            f"  {technique}: F1={stats['f1']:.3f}, "
            f"P={stats['precision']:.3f}, R={stats['recall']:.3f}, "
            f"support={stats['support']}"
        )

    with open(eval_output_dir / "label_list.json", 'w') as f:
        json.dump(LABEL_LIST, f, indent=2)

    print(f"\nTest evaluation artifacts saved to: {eval_output_dir}")
    print("=" * 80 + "\n")


def train_with_best_params(study=None, test_dataset=None, seed: int = 42, output_suffix: str = "final_model"):
    """
    Train final model with best parameters from optimization
    
    Args:
        study: Optuna study object (optional, will load from DB if None)
        test_dataset: Test dataset for final evaluation
    """
    if study is None:
        study = optuna.load_study(
            study_name="multitask_deberta_optimization",
            storage=f"sqlite:///{BASE_CONFIG['output_dir']}/optuna_study.db"
        )
    set_seed(seed)
    best_params = study.best_params
    
    print("\n" + "="*80)
    print("TRAINING FINAL MODEL WITH BEST PARAMETERS")
    print("="*80)
    print(f"\nBest parameters (F1 Macro: {study.best_value:.4f}):")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print("="*80 + "\n")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_CONFIG['model_name'])

    # Prepare datasets with the best max_length so both heads see aligned supervision
    train_dataset, dev_dataset, prepared_test_dataset, _ = prepare_datasets(
        BASE_CONFIG['train_file'],
        BASE_CONFIG['dev_file'],
        BASE_CONFIG['test_file'],
        LABEL_LIST,
        tokenizer,
        max_length=best_params['max_length']
    )
    if test_dataset is None:
        test_dataset = prepared_test_dataset
    pos_weight = compute_pos_weight_from_dataset(
        train_dataset,
        BASE_CONFIG['num_token_labels'],
        field="token_labels",
    )

    # Load model
    #config = DebertaV2Config.from_pretrained(BASE_CONFIG['model_name'])
    config = AutoConfig.from_pretrained(BASE_CONFIG['model_name'])
    config.hidden_dropout_prob = best_params['dropout_prob']
    config.attention_probs_dropout_prob = best_params['dropout_prob']
    config.token_loss_strategy = "balanced_per_label"
    config.token_loss_max_pos_weight = 5.0
    config.token_loss_eps = 1e-8
    
    model = MultitaskDeberta.from_pretrained(
        BASE_CONFIG['model_name'],
        config=config,
        num_comment_labels=BASE_CONFIG['num_comment_labels'],
        num_token_labels=BASE_CONFIG['num_token_labels'],
        alpha=best_params['alpha'],
        beta=best_params['beta'],
        token_pooling_ratio=best_params.get('token_pooling_ratio', 0.0),
        pooling_warmup_epochs=2,
    )
    model.set_pos_weight(pos_weight)
    if model.pos_weight is not None:
        print("Token pos_weight:", model.pos_weight.detach().cpu())
    freeze_bottom_layers(model, num_layers_to_freeze=6)
    
    # Training arguments
    final_output_dir = f"{BASE_CONFIG['output_dir']}/{output_suffix}"
    final_weight_decay = best_params.get('weight_decay', BASE_CONFIG['weight_decay'])
    training_args = TrainingArguments(
        output_dir=final_output_dir,
        num_train_epochs=best_params['num_epochs'],
        per_device_train_batch_size=best_params['batch_size'],
        per_device_eval_batch_size=best_params['batch_size'],
        gradient_accumulation_steps=best_params['gradient_accumulation_steps'],
        learning_rate=best_params['learning_rate'],
        warmup_ratio=BASE_CONFIG['warmup_ratio'],
        weight_decay=final_weight_decay,
        logging_dir=f"{final_output_dir}/logs",
        logging_steps=BASE_CONFIG['logging_steps'],
        eval_strategy="steps",
        eval_steps=BASE_CONFIG['eval_steps'],
        save_strategy="steps",
        save_steps=BASE_CONFIG['save_steps'],
        load_best_model_at_end=True,
        metric_for_best_model="eval_comment_f1_macro",
        greater_is_better=True,
        fp16=BASE_CONFIG['fp16'],
        report_to="none",
        save_total_limit=3,
        max_grad_norm=0.5,
    )
    optimizer = Adafactor(
        model.parameters(),
        lr=best_params['learning_rate'],
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        weight_decay=final_weight_decay,
    )
    final_steps_per_epoch = max(
        1,
        math.ceil(
            len(train_dataset)
            / (best_params['batch_size'] * best_params['gradient_accumulation_steps'])
        ),
    )
    final_num_training_steps = max(1, final_steps_per_epoch * best_params['num_epochs'])
    final_num_warmup_steps = int(0.1 * final_num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=final_num_warmup_steps,
        num_training_steps=final_num_training_steps,
    )
    
    def compute_metrics_fn(eval_pred: EvalPrediction):
        return compute_multitask_metrics(eval_pred, threshold=best_params['threshold'])
    
    trainer = MultitaskTrainerWrapper(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics_fn,
        optimizers=(optimizer, scheduler),
        callbacks=[
            NaNGradientCallback(),
            DelayedEarlyStoppingCallback(
                min_epochs=max(5, best_params['num_epochs'] // 2),
                early_stopping_patience=3,
                early_stopping_threshold=0.001,
            ),
        ],
    )
    trainer.add_callback(UnfreezeCallback())
    
    # Train
    print("Training final model...")
    trainer.train()
    
    # Save
    print("\nSaving final model...")
    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)

    # Save label list to final model directory
    with open(f"{final_output_dir}/label_list.json", 'w') as f:
        json.dump(LABEL_LIST, f, indent=2)

    # Save training configuration (use different name to avoid overwriting model config)
    final_config = {**BASE_CONFIG, **best_params}
    with open(f"{final_output_dir}/training_config.json", 'w') as f:
        json.dump(final_config, f, indent=2)
    
    # Evaluate on test set
    print("\n" + "="*80)
    print("EVALUATING ON TEST SET")
    print("="*80)
    
    test_results = trainer.evaluate(test_dataset)
    print_training_summary(test_results, "Final", mode="test")
    
    with open(f"{final_output_dir}/test_results.json", 'w') as f:
        results_serializable = {k: float(v) if isinstance(v, np.floating) else v 
                               for k, v in test_results.items()}
        json.dump(results_serializable, f, indent=2)
    
    # Get predictions
    test_predictions = trainer.predict(test_dataset)
    
    prediction_tensors = test_predictions.predictions
    comment_logits = prediction_tensors[0]
    comment_probs = torch.sigmoid(torch.tensor(comment_logits)).numpy()
    comment_preds = (comment_probs > best_params['threshold']).astype(int)
    
    token_logits = prediction_tensors[1]
    token_probs = torch.sigmoid(torch.tensor(token_logits)).numpy()
    token_preds = (token_probs > 0.5).astype(int)
    
    np.save(f"{final_output_dir}/test_comment_predictions.npy", comment_preds)
    np.save(f"{final_output_dir}/test_token_predictions.npy", token_preds)
    np.save(f"{final_output_dir}/test_token_probabilities.npy", token_probs)
    
    print(f"\n{'='*80}")
    print(f"Final model saved to: {final_output_dir}")
    print(f"{'='*80}\n")
    
    return trainer, test_results


def run_best_multiple_times(n_runs: int = 3, base_seed: int = 42):
    """
    Re-train/evaluate the best Optuna setting multiple times to report mean/std.
    """
    study = optuna.load_study(
        study_name="multitask_deberta_optimization",
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


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for multi-task propaganda detection")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of optimization trials")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs")
    parser.add_argument("--train_final", action="store_true", help="Train final model with best params after optimization")
    parser.add_argument("--only_final", action="store_true", help="Skip optimization and only train final model")
    parser.add_argument("--eval_test_only", action="store_true", help="Evaluate best Optuna trial on test set")
    parser.add_argument("--repeat_best", type=int, default=0, help="Repeat best setting N times (train+eval) with different seeds")
    
    args = parser.parse_args()
    
    if args.only_final:
        # Load existing study and train final model
        if args.repeat_best > 0:
            run_best_multiple_times(n_runs=args.repeat_best)
        else:
            train_with_best_params()
    elif args.eval_test_only:
        evaluate_best_trial_on_test()
        if args.repeat_best > 0:
            run_best_multiple_times(n_runs=args.repeat_best)
    else:
        # Run optimization
        study = run_optimization(n_trials=args.n_trials, n_jobs=args.n_jobs)
        
        # Optionally train final model
        if args.train_final:
            train_with_best_params(study)
            
        evaluate_best_trial_on_test()
        
        if args.repeat_best > 0:
            run_best_multiple_times(n_runs=args.repeat_best)
