import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PreTrainedModel
from typing import Optional, Dict, Any


class MultitaskDebertaBestOfBoth(PreTrainedModel):
    """
    Dual-head multitask DeBERTa that blends the stability of the legacy setup with the
    normalized losses and pooling clarity of the newer architecture.
    """

    config_class = AutoConfig

    def __init__(
        self,
        config,
        num_comment_labels: int = 20,
        num_token_labels: int = 20,
        alpha: float = 2.0,
        beta: float = 1.0,
        pretrained_model_name: Optional[str] = None,
        token_pooling_ratio: float = 0.3,
        dropout_prob: Optional[float] = None,
        pooling_warmup_epochs: int = 0,
        use_pooling_gate: bool = True,
    ):
        super().__init__(config)
        self.num_comment_labels = num_comment_labels
        self.num_token_labels = num_token_labels
        self.alpha = alpha
        self.beta = beta
        self.token_pooling_ratio = float(min(max(token_pooling_ratio, 0.0), 1.0))
        self.pooling_warmup_epochs = max(0, pooling_warmup_epochs)
        self.use_pooling_gate = use_pooling_gate
        self.current_epoch = 0
        self.use_focal_loss = getattr(config, "use_focal", False)
        self.focal_gamma = getattr(config, "focal_gamma", 2.0)
        self.focal_alpha = getattr(config, "focal_alpha", 0.25)

        hidden_dropout = dropout_prob if dropout_prob is not None else config.hidden_dropout_prob

        if pretrained_model_name:
            self.encoder = AutoModel.from_pretrained(pretrained_model_name, config=config)
        else:
            self.encoder = AutoModel.from_config(config)

        self.dropout = nn.Dropout(hidden_dropout)
        self.token_classifier = nn.Linear(config.hidden_size, num_token_labels)
        self.comment_classifier = nn.Linear(config.hidden_size, num_comment_labels)
        self.pooling_gate = nn.Linear(config.hidden_size, 1) if use_pooling_gate else None
        self.register_buffer("pos_weight", None)

        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        comment_labels: Optional[torch.Tensor] = None,
        token_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Dict[str, Any]:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs.last_hidden_state  # [B, L, H]
        cls_output = sequence_output[:, 0, :]  # [B, H]

        token_hidden = self.dropout(sequence_output)
        token_logits = self.token_classifier(token_hidden)  # [B, L, num_token_labels]

        mask = attention_mask.bool() if attention_mask is not None else torch.ones(
            sequence_output.size(0),
            sequence_output.size(1),
            dtype=torch.bool,
            device=sequence_output.device,
        )

        token_loss = None
        if token_labels is not None:
            labels = token_labels.float()
            pos_weight = self.pos_weight
            if pos_weight is not None and (
                pos_weight.device != token_logits.device
                or pos_weight.dtype != token_logits.dtype
            ):
                pos_weight = pos_weight.to(token_logits.device, dtype=token_logits.dtype)
                self.pos_weight = pos_weight
            if pos_weight is not None:
                bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
            else:
                bce = nn.BCEWithLogitsLoss(reduction="none")
            raw_token_loss = bce(token_logits, labels)
            if self.use_focal_loss:
                probs = torch.sigmoid(token_logits)
                focal_weight = (
                    self.focal_alpha * (1 - probs) ** self.focal_gamma * labels
                    + (1 - self.focal_alpha) * probs ** self.focal_gamma * (1 - labels)
                )
                raw_token_loss = focal_weight * raw_token_loss
            masked_token_loss = raw_token_loss * mask.unsqueeze(-1)
            denom = mask.sum().clamp(min=1).float() * token_logits.size(-1) + 1e-8
            token_loss = masked_token_loss.sum() / denom

        token_probs = torch.sigmoid(token_logits)
        pooling_probs = token_probs
        if (
            self.training
            and self.pooling_warmup_epochs > 0
            and self.current_epoch < self.pooling_warmup_epochs
        ):
            pooling_probs = pooling_probs.detach()

        token_importance = pooling_probs.mean(dim=-1)  # [B, L]
        mask_value = torch.finfo(token_importance.dtype).min
        token_importance = token_importance.masked_fill(~mask, mask_value)
        token_weights = torch.softmax(token_importance, dim=-1).unsqueeze(-1)  # [B, L, 1]

        weighted_sum = torch.sum(sequence_output * token_weights, dim=1)  # [B, H]

        if self.use_pooling_gate and self.pooling_gate is not None:
            gate = torch.sigmoid(self.pooling_gate(cls_output))  # [B, 1]
            pooled_output = gate * weighted_sum + (1.0 - gate) * cls_output
        else:
            ratio = self.token_pooling_ratio
            pooled_output = (1.0 - ratio) * cls_output + ratio * weighted_sum

        pooled_output = self.dropout(pooled_output)
        comment_logits = self.comment_classifier(pooled_output)

        comment_loss = None
        if comment_labels is not None:
            comment_loss = F.binary_cross_entropy_with_logits(
                comment_logits,
                comment_labels.float(),
            )

        total_loss = None
        if comment_loss is not None and token_loss is not None:
            # === Adaptive weighting between heads (stabilized) ===
            with torch.no_grad():
                temp = 0.5  # smoothing temperature
                comment_scale = (token_loss.detach() + temp) / (comment_loss.detach() + temp)
                token_scale = (comment_loss.detach() + temp) / (token_loss.detach() + temp)

            # Clip extreme ratios to avoid runaway weights
            comment_scale = comment_scale.clamp(0.5, 1.5)
            token_scale = token_scale.clamp(0.5, 1.5)

            if self.training and random.random() < 0.05:
                print(
                    f"[AdaptiveLoss] comment_scale={comment_scale.item():.3f}, "
                    f"token_scale={token_scale.item():.3f}"
                )

            # Weighted multitask loss
            total_loss = (
                self.alpha * comment_scale * comment_loss
                + self.beta * token_scale * token_loss
            )
        elif comment_loss is not None:
            total_loss = comment_loss
        elif token_loss is not None:
            total_loss = token_loss

        if not return_dict:
            output = (comment_logits, token_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return {
            "loss": total_loss,
            "comment_loss": comment_loss,
            "token_loss": token_loss,
            "comment_logits": comment_logits,
            "token_logits": token_logits,
            "hidden_states": getattr(outputs, "hidden_states", None),
            "attentions": getattr(outputs, "attentions", None),
        }

    def set_pos_weight(self, pos_weight: Optional[torch.Tensor]):
        if pos_weight is None:
            self.pos_weight = None
        else:
            device = self.token_classifier.weight.device
            dtype = self.token_classifier.weight.dtype
            self.pos_weight = pos_weight.to(device=device, dtype=dtype)

    def get_comment_predictions(self, comment_logits: torch.Tensor, threshold: float = 0.3):
        probs = torch.sigmoid(comment_logits)
        return (probs > threshold).int()

    def get_token_predictions(self, token_logits: torch.Tensor, threshold: float = 0.5):
        probs = torch.sigmoid(token_logits)
        return (probs > threshold).int()


class MultitaskDeberta(MultitaskDebertaBestOfBoth):
    """
    Backwards-compatible alias so existing training and inference scripts keep working.
    """

    pass


__all__ = ["MultitaskDebertaBestOfBoth", "MultitaskDeberta"]
