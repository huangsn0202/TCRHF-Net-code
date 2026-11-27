"""Shared product-only pairwise relation encoding retained from model _2."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F


RELATION_NAMES = ("text_image", "text_caption", "image_caption")


class ProductRelationEncoder(nn.Module):
    def __init__(self, feature_dim: int = 1024, dropout: float = 0.2) -> None:
        super().__init__()
        self.relation_norm = nn.LayerNorm(feature_dim, elementwise_affine=False)
        self.product_proj = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(feature_dim)

    def _encode_pair(
        self, first: torch.Tensor, second: torch.Tensor
    ) -> torch.Tensor:
        first_norm = self.relation_norm(first)
        second_norm = self.relation_norm(second)
        relation = self.product_proj(first_norm * second_norm)
        return self.output_norm(self.dropout(F.gelu(relation)))

    def forward(
        self, h_text: torch.Tensor, h_image: torch.Tensor, h_caption: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return {
            "r_text_image": self._encode_pair(h_text, h_image),
            "r_text_caption": self._encode_pair(h_text, h_caption),
            "r_image_caption": self._encode_pair(h_image, h_caption),
        }

