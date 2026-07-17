"""Anonymized core implementation of HSCNet.

This file keeps the model logic used in the paper while omitting dataset
loading, label lists, subject information, checkpoints, and experiment paths.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.ELU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.gate(x).view(x.size(0), x.size(1), 1, 1)
        return x * weight


class MultiScaleTemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes=(3, 7, 15),
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        base = out_channels // len(kernel_sizes)
        rem = out_channels % len(kernel_sizes)
        branches = []
        for i, kernel_size in enumerate(kernel_sizes):
            branch_channels = base + int(i < rem)
            branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        branch_channels,
                        kernel_size=(1, kernel_size),
                        padding=(0, kernel_size // 2),
                    ),
                    nn.BatchNorm2d(branch_channels),
                    nn.ELU(),
                    nn.Dropout(dropout),
                )
            )
        self.branches = nn.ModuleList(branches)
        self.mix = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mix(torch.cat([branch(x) for branch in self.branches], dim=1))


class TemporalTransformer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        depth: int = 2,
        mlp_ratio: int = 4,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=embed_dim * mlp_ratio,
                    dropout=dropout,
                    batch_first=True,
                    activation="gelu",
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class SemanticPrototypeHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        prototype_dim: int,
        temperature: float = 0.1,
        initial_prototypes: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, prototype_dim),
            nn.LayerNorm(prototype_dim),
        )
        if initial_prototypes is None:
            initial_prototypes = torch.empty(num_classes, prototype_dim)
            nn.init.xavier_uniform_(initial_prototypes)
        self.prototypes = nn.Parameter(initial_prototypes.clone().float())

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        projected = F.normalize(self.projector(features), dim=-1)
        prototypes = F.normalize(self.prototypes, dim=-1)
        logits = torch.matmul(projected, prototypes.t()) / self.temperature
        return logits, projected


class HSCNet(nn.Module):
    """Hierarchical Semantic Constraint Network.

    Inputs are expected as EEG tensors with shape `[batch, channels, time]`.
    Optional label embeddings may be initialized by external word-vector tools.
    """

    def __init__(
        self,
        in_channels: int,
        num_fine_classes: int,
        num_group_classes: int,
        embed_dim: int = 64,
        temporal_channels: int = 64,
        dropout: float = 0.5,
        transformer_depth: int = 2,
        transformer_heads: int = 4,
        prototype_temperature: float = 0.1,
        prototype_mix: float = 0.35,
        fine_label_init: Optional[torch.Tensor] = None,
        group_label_init: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.prototype_mix = prototype_mix

        self.shared_temporal_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 32), padding=(0, 16)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 2), (1, 2)),
            nn.Dropout(dropout),
        )

        self.state_branch = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=(in_channels, 1)),
            nn.BatchNorm2d(48),
            nn.ELU(),
            SEBlock(48, reduction=4),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(48, embed_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        self.dynamic_tcn = MultiScaleTemporalBlock(
            32,
            temporal_channels,
            kernel_sizes=(3, 7, 15),
            dropout=dropout,
        )
        self.dynamic_spatial = nn.Sequential(
            nn.Conv2d(temporal_channels, temporal_channels, kernel_size=(in_channels, 1)),
            nn.BatchNorm2d(temporal_channels),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.dynamic_pool = nn.AdaptiveAvgPool2d((1, 25))
        self.dynamic_projection = nn.Linear(temporal_channels, embed_dim)
        self.dynamic_transformer = TemporalTransformer(
            embed_dim,
            num_heads=transformer_heads,
            depth=transformer_depth,
            dropout=dropout,
        )
        self.dynamic_attention = nn.Sequential(nn.Linear(embed_dim, 1), nn.Softmax(dim=1))

        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.classifier_dropout = nn.Dropout(dropout)
        self.fine_classifier = nn.Linear(embed_dim, num_fine_classes)
        self.group_classifier = nn.Linear(embed_dim, num_group_classes)

        self.fine_label_embedding = nn.Embedding(num_fine_classes, embed_dim)
        self.group_label_embedding = nn.Embedding(num_group_classes, embed_dim)
        if fine_label_init is not None:
            self.fine_label_embedding.weight.data.copy_(fine_label_init.float())
        if group_label_init is not None:
            self.group_label_embedding.weight.data.copy_(group_label_init.float())

        fine_proto_init = self.fine_label_embedding.weight.detach()
        group_proto_init = self.group_label_embedding.weight.detach()
        self.fine_prototype_head = SemanticPrototypeHead(
            embed_dim,
            num_fine_classes,
            embed_dim,
            temperature=prototype_temperature,
            initial_prototypes=fine_proto_init,
        )
        self.group_prototype_head = SemanticPrototypeHead(
            embed_dim,
            num_group_classes,
            embed_dim,
            temperature=prototype_temperature,
            initial_prototypes=group_proto_init,
        )

    def encode(self, eeg: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared = self.shared_temporal_encoder(eeg.unsqueeze(1))

        state_feature = self.state_branch(shared)

        dynamic = self.dynamic_tcn(shared)
        dynamic = self.dynamic_spatial(dynamic)
        dynamic = self.dynamic_pool(dynamic)
        dynamic = dynamic.squeeze(2).permute(0, 2, 1)
        dynamic = self.dynamic_projection(dynamic)
        dynamic = self.dynamic_transformer(dynamic)
        attention = self.dynamic_attention(dynamic)
        dynamic_feature = (dynamic * attention).sum(dim=1)

        fused_feature = self.fusion(torch.cat([dynamic_feature, state_feature], dim=1))
        fused_feature = F.normalize(fused_feature, dim=-1)
        return {
            "state_feature": state_feature,
            "dynamic_feature": dynamic_feature,
            "fused_feature": fused_feature,
        }

    def forward(
        self,
        eeg: torch.Tensor,
        fine_labels: Optional[torch.Tensor] = None,
        group_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        features = self.encode(eeg)
        fused = features["fused_feature"]

        fine_cls_logits = self.fine_classifier(self.classifier_dropout(fused))
        group_cls_logits = self.group_classifier(self.classifier_dropout(fused))
        fine_proto_logits, fine_proto_feature = self.fine_prototype_head(fused)
        group_proto_logits, group_proto_feature = self.group_prototype_head(fused)

        outputs: Dict[str, Optional[torch.Tensor]] = {
            **features,
            "fine_logits": (1.0 - self.prototype_mix) * fine_cls_logits
            + self.prototype_mix * fine_proto_logits,
            "group_logits": (1.0 - self.prototype_mix) * group_cls_logits
            + self.prototype_mix * group_proto_logits,
            "fine_proto_feature": fine_proto_feature,
            "group_proto_feature": group_proto_feature,
            "fine_label_embedding": None,
            "group_label_embedding": None,
        }

        if fine_labels is not None:
            outputs["fine_label_embedding"] = F.normalize(
                self.fine_label_embedding(fine_labels),
                dim=-1,
            )
        if group_labels is not None:
            outputs["group_label_embedding"] = F.normalize(
                self.group_label_embedding(group_labels),
                dim=-1,
            )
        return outputs
