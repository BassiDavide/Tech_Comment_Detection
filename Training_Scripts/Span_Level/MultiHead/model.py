import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PreTrainedModel
from typing import Optional, Dict, Any


class MultiHeadTokenDeberta(PreTrainedModel):
    """
    Token classifier with per-label heads and a shared projection.
    """

    config_class = AutoConfig

    def __init__(
        self,
        config,
        num_token_labels: int = 20,
        head_dim: int = 64,
        pretrained_model_name: Optional[str] = None,
        dropout_prob: Optional[float] = None,
    ):
        super().__init__(config)
        self.num_token_labels = num_token_labels
        self.head_dim = head_dim

        hidden_dropout = dropout_prob if dropout_prob is not None else config.hidden_dropout_prob

        if pretrained_model_name:
            self.encoder = AutoModel.from_pretrained(pretrained_model_name, config=config)
        else:
            self.encoder = AutoModel.from_config(config)

        self.proj = nn.Linear(config.hidden_size, head_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(hidden_dropout)
        self.heads = nn.ModuleList([nn.Linear(head_dim, 1) for _ in range(num_token_labels)])
        self.register_buffer("pos_weight", None)
        self.register_buffer("label_weights", None)

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
        hidden = self.dropout(self.act(self.proj(sequence_output)))

        logits_per_head = [head(hidden).squeeze(-1) for head in self.heads]
        token_logits = torch.stack(logits_per_head, dim=-1)  # [B, L, num_labels]

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
            if self.label_weights is not None:
                raw_token_loss = raw_token_loss * self.label_weights
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
            device = self.heads[0].weight.device
            dtype = self.heads[0].weight.dtype
            self.pos_weight = pos_weight.to(device=device, dtype=dtype)

    def set_label_weights(self, weights: Optional[torch.Tensor]):
        if weights is None:
            self.label_weights = None
        else:
            device = self.heads[0].weight.device
            dtype = self.heads[0].weight.dtype
            self.label_weights = weights.to(device=device, dtype=dtype)

    def get_token_predictions(self, token_logits: torch.Tensor, threshold: float = 0.5):
        probs = torch.sigmoid(token_logits)
        return (probs > threshold).int()


__all__ = ["MultiHeadTokenDeberta"]
