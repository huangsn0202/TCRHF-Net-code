"""Final classification plus the unchanged three-relation auxiliary loss."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F

try:
    from .model_2 import RELATION_NAMES
except ImportError:
    from model_2 import RELATION_NAMES


class JointLoss(nn.Module):
    def __init__(self, relation_loss_weight: float = 0.20) -> None:
        super().__init__()
        if relation_loss_weight < 0:
            raise ValueError("relation_loss_weight must be non-negative.")
        self.relation_loss_weight = relation_loss_weight

    def forward(
        self, outputs: Dict[str, torch.Tensor], labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        classification = F.cross_entropy(outputs["logits"], labels)
        relation_parts = {
            name: F.cross_entropy(outputs[f"logits_{name}"], labels)
            for name in RELATION_NAMES
        }
        relation = torch.stack(list(relation_parts.values())).mean()
        total = classification + self.relation_loss_weight * relation
        components = {
            "total": total,
            "classification": classification,
            "relation": relation,
        }
        components.update(
            {f"relation_{name}": value for name, value in relation_parts.items()}
        )
        return total, components


