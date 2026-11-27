"""Concatenation-MLP group encoding followed by content-relation fusion."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F


class ThreeBranchConcatEncoder(nn.Module):
    """Preserve three branch identities and map 3D concatenation back to D."""

    def __init__(
        self,
        feature_dim: int = 1024,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or hidden_dim <= 0:
            raise ValueError("feature_dim and hidden_dim must be positive.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        self.concatenated_dim = 3 * feature_dim
        self.encoder = nn.Sequential(
            nn.LayerNorm(self.concatenated_dim),
            nn.Linear(self.concatenated_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, concatenated: torch.Tensor) -> torch.Tensor:
        if concatenated.ndim != 2 or concatenated.size(-1) != self.concatenated_dim:
            raise ValueError(
                f"Expected [B, {self.concatenated_dim}], got {tuple(concatenated.shape)}."
            )
        return self.encoder(concatenated)


class ContentRelationConcatMLPFusion(nn.Module):
    """Encode each evidence group by concat-MLP, then compare 256-D decisions."""

    def __init__(
        self,
        feature_dim: int = 1024,
        decision_dim: int = 256,
        aggregation_hidden_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or decision_dim <= 0:
            raise ValueError("feature_dim and decision_dim must be positive.")
        if 4 * decision_dim != feature_dim:
            raise ValueError("Four decision blocks must concatenate to feature_dim.")
        self.feature_dim = feature_dim
        self.decision_dim = decision_dim

        # Independent encoders retain branch positions within the content and
        # relation groups; unlike mean pooling, no branch identity is discarded.
        self.content_group_encoder = ThreeBranchConcatEncoder(
            feature_dim, aggregation_hidden_dim, dropout
        )
        self.relation_group_encoder = ThreeBranchConcatEncoder(
            feature_dim, aggregation_hidden_dim, dropout
        )

        # These are unchanged from model _5: map the two 1024-D group features
        # into 256-D decision representations before explicit comparison.
        self.content_projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, decision_dim),
        )
        self.relation_projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, decision_dim),
        )
        self.decision_norm = nn.LayerNorm(decision_dim, elementwise_affine=False)

    def forward(
        self,
        h_text: torch.Tensor,
        h_image: torch.Tensor,
        h_caption: torch.Tensor,
        r_text_image: torch.Tensor,
        r_text_caption: torch.Tensor,
        r_image_caption: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        content_stack = torch.stack((h_text, h_image, h_caption), dim=1)
        relation_stack = torch.stack(
            (r_text_image, r_text_caption, r_image_caption), dim=1
        )

        content_concat = torch.cat((h_text, h_image, h_caption), dim=-1)
        relation_concat = torch.cat(
            (r_text_image, r_text_caption, r_image_caption), dim=-1
        )
        content_group = self.content_group_encoder(content_concat)
        relation_group = self.relation_group_encoder(relation_concat)

        z_content = self.decision_norm(self.content_projection(content_group))
        z_relation = self.decision_norm(self.relation_projection(relation_group))
        content_relation_difference = torch.abs(z_content - z_relation)
        content_relation_product = z_content * z_relation
        fused = torch.cat(
            (
                z_content,
                z_relation,
                content_relation_difference,
                content_relation_product,
            ),
            dim=-1,
        )
        return {
            "content_stack": content_stack,
            "relation_stack": relation_stack,
            "content_concat": content_concat,
            "relation_concat": relation_concat,
            "content_group": content_group,
            "relation_group": relation_group,
            "z_content": z_content,
            "z_relation": z_relation,
            "content_relation_difference": content_relation_difference,
            "content_relation_product": content_relation_product,
            "content_relation_distance": content_relation_difference.mean(dim=-1),
            "content_relation_cosine": F.cosine_similarity(
                z_content, z_relation, dim=-1, eps=1e-8
            ),
            "fused": fused,
        }


# Short alias used by model_4.py.
ContentRelationFusion = ContentRelationConcatMLPFusion
