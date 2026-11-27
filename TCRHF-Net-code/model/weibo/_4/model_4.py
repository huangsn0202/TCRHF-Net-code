"""TCRHF on three offline-encoded global vectors (paper sections 3.2-3.6)."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn

try:
    from .model_1 import ViewCalibrationModule
    from .model_2 import ProductRelationEncoder, RELATION_NAMES
    from .model_3 import ContentRelationFusion
except ImportError:
    from model_1 import ViewCalibrationModule
    from model_2 import ProductRelationEncoder, RELATION_NAMES
    from model_3 import ContentRelationFusion


class ConcatMLPFusionRumorModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 1024,
        feature_dim: int = 1024,
        num_classes: int = 2,
        decision_dim: int = 256,
        aggregation_hidden_dim: int = 256,
        input_dropout: float = 0.0,
        view_dropout: float = 0.2,
        relation_dropout: float = 0.2,
        fusion_dropout: float = 0.2,
        classifier_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        dropout_settings = {
            "input_dropout": input_dropout,
            "view_dropout": view_dropout,
            "relation_dropout": relation_dropout,
            "fusion_dropout": fusion_dropout,
            "classifier_dropout": classifier_dropout,
        }
        for name, value in dropout_settings.items():
            if not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0, 1), got {value}.")

        # The same module is called three times, generating an independent mask
        # for T, I and C while using one input-level probability.
        self.input_dropout = nn.Dropout(input_dropout)
        self.view_module = ViewCalibrationModule(
            input_dim=input_dim, feature_dim=feature_dim, dropout=view_dropout
        )
        self.relation_module = ProductRelationEncoder(
            feature_dim, dropout=relation_dropout
        )
        self.relation_classifiers = nn.ModuleDict(
            {name: nn.Linear(feature_dim, num_classes) for name in RELATION_NAMES}
        )

        self.final_classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(feature_dim, num_classes),
        )
        self.fusion = ContentRelationFusion(
            feature_dim=feature_dim,
            decision_dim=decision_dim,
            aggregation_hidden_dim=aggregation_hidden_dim,
            dropout=fusion_dropout,
        )

    def forward(
        self, text: torch.Tensor, image: torch.Tensor, caption: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        for name, features in (("text", text), ("image", image), ("caption", caption)):
            if features.ndim != 2 or features.size(-1) != self.input_dim:
                raise ValueError(f"{name} must have shape [B, {self.input_dim}]; got {tuple(features.shape)}.")
        if text.shape != image.shape or text.shape != caption.shape:
            raise ValueError("T/I/C must have matching batch and feature dimensions.")
        text_input = self.input_dropout(text)
        image_input = self.input_dropout(image)
        caption_input = self.input_dropout(caption)
        views = self.view_module(text_input, image_input, caption_input)
        relations = self.relation_module(
            views["h_text"], views["h_image"], views["h_caption"]
        )
        relation_logits = {
            f"logits_{name}": self.relation_classifiers[name](relations[f"r_{name}"])
            for name in RELATION_NAMES
        }
        fusion = self.fusion(
            views["h_text"],
            views["h_image"],
            views["h_caption"],
            relations["r_text_image"],
            relations["r_text_caption"],
            relations["r_image_caption"],
        )
        logits = self.final_classifier(fusion["fused"])
        return {
            **views,
            **relations,
            **relation_logits,
            **fusion,
            "logits": logits,
            "probabilities": torch.softmax(logits, dim=-1),
        }


ThreeViewRumorModel = ConcatMLPFusionRumorModel
