import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PreTrainedModel
from typing import Optional, Dict, Any


class TokenDeberta(PreTrainedModel):
    """
    Span-only DeBERTa with a token-level multi-label head.
    """

    config_class = AutoConfig

    def __init__(
        self,
        config,
        num_token_labels: int = 20,
        pretrained_model_name: Optional[str] = None,
        dropout_prob: Optional[float] = None,
    ):
        super().__init__(config)
        self.num_token_labels = num_token_labels
        self.use_focal_loss = getattr(config, "use_focal", False)
        self.focal_gamma = getattr(config, "focal_gamma", 2.0)
        self.focal_alpha = getattr(config, "focal_alpha", 0.25)
        self.current_epoch = 0

        hidden_dropout = dropout_prob if dropout_prob is not None else config.hidden_dropout_prob

        if pretrained_model_name:
            self.encoder = AutoModel.from_pretrained(pretrained_model_name, config=config)
        else:
            self.encoder = AutoModel.from_config(config)

        self.dropout = nn.Dropout(hidden_dropout)
        self.token_classifier = nn.Linear(config.hidden_size, num_token_labels)
        self.register_buffer("pos_weight", None)

        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        token_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Dict[str, Any]:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs.last_hidden_state  # [B, L, H]
        token_hidden = self.dropout(sequence_output)
        token_logits = self.token_classifier(token_hidden)  # [B, L, num_labels]

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

        if not return_dict:
            output = (token_logits,) + outputs[2:]
            return ((token_loss,) + output) if token_loss is not None else output

        return {
            "loss": token_loss,
            "token_loss": token_loss,
            "token_logits": token_logits,
            "logits": token_logits,
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

    def get_token_predictions(self, token_logits: torch.Tensor, threshold: float = 0.5):
        probs = torch.sigmoid(token_logits)
        return (probs > threshold).int()


__all__ = ["TokenDeberta"]
