"""Classification, field-alignment, and source-fusion objectives."""
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossConfig:
    lambda_op: float = 0.08
    lambda_tr: float = 0.04
    lambda_bal: float = 0.08
    lambda_ent: float = 0.03
    label_smoothing: float = 0.02
    positive_class_weight: float = 1.0
    structural_ramp_epochs: int = 6

    def __post_init__(self) -> None:
        if not 0 <= self.label_smoothing <= 1:
            raise ValueError("label_smoothing must lie in [0, 1]")
        if min(self.lambda_op, self.lambda_tr, self.lambda_bal, self.lambda_ent) < 0:
            raise ValueError("Loss coefficients must be nonnegative")
        if self.positive_class_weight <= 0 or self.structural_ramp_epochs < 1:
            raise ValueError("Class weight and ramp duration must be positive")


def cipher_objective(output: dict, labels: torch.Tensor, epoch: int,
                     config: LossConfig = LossConfig()) -> dict[str, torch.Tensor]:
    pair_logits = output["pair_logits"]
    smoothed_labels = labels * (1 - config.label_smoothing) + config.label_smoothing / 2
    loss_bce = F.binary_cross_entropy_with_logits(
        pair_logits, smoothed_labels,
        pos_weight=pair_logits.new_tensor(config.positive_class_weight),
    )
    loss_op = torch.stack([
        value.square().mean() for value in output["field_alignment_residuals"]
    ]).mean()
    loss_tr = torch.stack(output["transport_penalties"]).mean()
    operator_weights = output["operator_source_weights"]
    loss_bal = torch.stack([
        F.mse_loss(weights.mean(dim=0), weights.new_full((weights.shape[1],), 1 / weights.shape[1]))
        for weights in operator_weights
    ]).mean()
    all_weights = output["local_source_weights"] + operator_weights
    gate_entropy = torch.stack([
        -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=-1).mean()
        for weights in all_weights
    ]).mean()
    ramp = min(1.0, max(0, epoch) / config.structural_ramp_epochs)
    loss_field = config.lambda_op * loss_op
    loss_source = (config.lambda_tr * loss_tr + config.lambda_bal * loss_bal
                   - config.lambda_ent * gate_entropy)
    return {"loss": loss_bce + ramp * (loss_field + loss_source),
            "loss_bce": loss_bce, "loss_op": loss_op, "loss_tr": loss_tr,
            "loss_bal": loss_bal, "gate_entropy": gate_entropy,
            "loss_field": loss_field, "loss_source": loss_source}
