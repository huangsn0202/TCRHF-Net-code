"""Pair-preserving compression for a variable number of image-caption features."""

from __future__ import annotations

import math
from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F


class PairedQueryCompressor(nn.Module):
    """Compress N paired I/C vectors to one I vector and one C vector.

    The reference Weibo implementation uses learnable queries to aggregate a
    variable visual sequence. Here every image and description has already been
    encoded as one 1024-dimensional vector, so a single lightweight query score
    per pair is sufficient. I and C deliberately share the same pair weight;
    independent weights could combine an image from one source post with a
    description from another and destroy the original correspondence.
    """

    def __init__(self, input_dim: int = 1024) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        self.input_dim = int(input_dim)
        self.image_query = nn.Parameter(torch.zeros(input_dim))
        self.caption_query = nn.Parameter(torch.zeros(input_dim))
        self.score_scale = math.sqrt(2.0 * input_dim)

    def forward(
        self,
        image: torch.Tensor,
        caption: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if image.ndim != 3 or caption.ndim != 3:
            raise ValueError("I and C inputs must have shape [B,N,D].")
        if image.shape != caption.shape:
            raise ValueError(
                f"I/C input shapes differ: {tuple(image.shape)} vs {tuple(caption.shape)}"
            )
        if pair_mask.ndim != 2 or image.shape[:2] != pair_mask.shape:
            raise ValueError(
                f"Pair mask {tuple(pair_mask.shape)} does not match "
                f"I/C leading shape {tuple(image.shape[:2])}."
            )
        if int(image.shape[-1]) != self.input_dim:
            raise ValueError(
                f"Input feature dimension is {image.shape[-1]}; "
                f"expected {self.input_dim}."
            )

        valid = pair_mask.to(torch.bool)
        if not bool(valid.any(dim=1).all()):
            raise ValueError("Every sample must contain at least one valid I/C pair.")

        # Parameter-free normalization is used only to calculate comparable
        # attention scores. The weighted sums retain the original raw features.
        normalized_image = F.layer_norm(image, (self.input_dim,))
        normalized_caption = F.layer_norm(caption, (self.input_dim,))
        scores = (
            torch.einsum("bnd,d->bn", normalized_image, self.image_query)
            + torch.einsum("bnd,d->bn", normalized_caption, self.caption_query)
        ) / self.score_scale
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores.float(), dim=1).to(image.dtype)
        weights = weights * valid.to(weights.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-12)

        compressed_image = torch.sum(image * weights.unsqueeze(-1), dim=1)
        compressed_caption = torch.sum(caption * weights.unsqueeze(-1), dim=1)
        pair_entropy = -torch.sum(
            weights * torch.log(weights.clamp_min(1.0e-12)), dim=1
        )
        return {
            "compressed_image": compressed_image,
            "compressed_caption": compressed_caption,
            "pair_weights": weights,
            "pair_max_weight": weights.max(dim=1).values,
            "pair_entropy": pair_entropy,
            "pair_count": valid.sum(dim=1),
        }

