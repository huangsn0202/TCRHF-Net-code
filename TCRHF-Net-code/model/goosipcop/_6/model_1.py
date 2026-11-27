"""Independent full-dimensional view calibration retained from model _2."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class FullDimViewCalibrator(nn.Module):
    def __init__(self, dim: int = 1024, expansion: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        hidden_dim = dim * expansion
        self.pre_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        self.post_norm = nn.LayerNorm(dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.post_norm(features + self.ffn(self.pre_norm(features)))


class ViewCalibrationModule(nn.Module):
    def __init__(
        self, input_dim: int = 1024, feature_dim: int = 1024, dropout: float = 0.2
    ) -> None:
        super().__init__()
        if input_dim != feature_dim:
            raise ValueError("The model keeps every view at 1024 dimensions.")
        self.text_calibrator = FullDimViewCalibrator(input_dim, dropout=dropout)
        self.image_calibrator = FullDimViewCalibrator(input_dim, dropout=dropout)
        self.caption_calibrator = FullDimViewCalibrator(input_dim, dropout=dropout)

    def forward(
        self, text: torch.Tensor, image: torch.Tensor, caption: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return {
            "h_text": self.text_calibrator(text),
            "h_image": self.image_calibrator(image),
            "h_caption": self.caption_calibrator(caption),
        }

