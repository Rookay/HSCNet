"""Core semantic objectives for HSCNet.

This module omits dataloaders, optimizer construction, logging, checkpointing,
and subject-level experiment orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HSCNetLossWeights:
    group_ce: float = 0.2
    contrastive: float = 0.2
    group_contrastive: float = 0.5
    hierarchy: float = 0.5
    branch_alignment: float = 0.05
    prototype_consistency: float = 0.5
    warmup_epochs: int = 20
    contrastive_temperature: float = 0.1


class HSCNetLoss(nn.Module):
    def __init__(
        self,
        weights: Optional[HSCNetLossWeights] = None,
        fine_to_group: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.weights = weights or HSCNetLossWeights()
        self.register_buffer(
            "fine_to_group",
            fine_to_group.long() if fine_to_group is not None else None,
            persistent=False,
        )
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction="batchmean")

    @staticmethod
    def cosine_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        return 1.0 - (x * y).sum(dim=-1).mean()

    def hierarchy_scale(self, epoch: int, max_epochs: int) -> float:
        warmup = self.weights.warmup_epochs
        if warmup <= 0:
            return 1.0
        progress = (epoch - warmup) / max(1, max_epochs - warmup)
        return float(max(0.0, min(1.0, progress)))

    def cross_space_contrastive(
        self,
        eeg_feature: torch.Tensor,
        label_embedding: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        eeg_feature = F.normalize(eeg_feature, dim=-1)
        label_embedding = F.normalize(label_embedding, dim=-1)
        logits = torch.matmul(eeg_feature, label_embedding.t())
        logits = logits / self.weights.contrastive_temperature

        positive = logits.diag()
        negative_mask = labels.unsqueeze(0) != labels.unsqueeze(1)
        negative_logits = logits.masked_fill(~negative_mask, float("-inf"))
        negative_logsumexp = torch.logsumexp(negative_logits, dim=1)
        denominator = torch.logaddexp(positive, negative_logsumexp)
        return -(positive - denominator).mean()

    def hierarchy_consistency(
        self,
        fine_logits: torch.Tensor,
        group_logits: torch.Tensor,
    ) -> torch.Tensor:
        if self.fine_to_group is None:
            return fine_logits.new_tensor(0.0)

        fine_prob = F.softmax(fine_logits, dim=1)
        group_prob = F.softmax(group_logits, dim=1)
        mapping = self.fine_to_group.to(fine_prob.device)

        group_from_fine = fine_prob.new_zeros(fine_prob.size(0), group_prob.size(1))
        group_from_fine.scatter_add_(
            1,
            mapping.unsqueeze(0).expand(fine_prob.size(0), -1),
            fine_prob,
        )
        group_from_fine = group_from_fine.clamp_min(1e-8)
        group_from_fine = group_from_fine / group_from_fine.sum(dim=1, keepdim=True)
        return self.kl(group_from_fine.log(), group_prob)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        fine_labels: torch.Tensor,
        group_labels: torch.Tensor,
        epoch: int,
        max_epochs: int,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        fine_ce = self.ce(outputs["fine_logits"], fine_labels)
        group_ce = self.ce(outputs["group_logits"], group_labels)

        fine_align = self.cosine_distance(
            outputs["dynamic_feature"],
            outputs["fine_label_embedding"],
        )
        group_align = self.cosine_distance(
            outputs["state_feature"],
            outputs["group_label_embedding"],
        )

        fine_proto_consistency = self.cosine_distance(
            outputs["dynamic_feature"],
            outputs["fine_proto_feature"],
        )
        group_proto_consistency = self.cosine_distance(
            outputs["state_feature"],
            outputs["group_proto_feature"],
        )

        if epoch <= self.weights.warmup_epochs:
            contrastive = outputs["fine_logits"].new_tensor(0.0)
            branch_scale = 0.5
        else:
            fine_contrastive = self.cross_space_contrastive(
                outputs["dynamic_feature"],
                outputs["fine_label_embedding"],
                fine_labels,
            )
            group_contrastive = self.cross_space_contrastive(
                outputs["state_feature"],
                outputs["group_label_embedding"],
                group_labels,
            )
            contrastive = fine_contrastive + self.weights.group_contrastive * group_contrastive
            branch_scale = 1.0

        hierarchy = self.hierarchy_consistency(outputs["fine_logits"], outputs["group_logits"])
        hierarchy = self.hierarchy_scale(epoch, max_epochs) * hierarchy

        total = (
            fine_ce
            + self.weights.group_ce * group_ce
            + self.weights.contrastive * contrastive
            + self.weights.hierarchy * hierarchy
            + branch_scale
            * self.weights.branch_alignment
            * (fine_align + group_align)
            + self.weights.prototype_consistency
            * (fine_proto_consistency + group_proto_consistency)
        )

        terms = {
            "fine_ce": fine_ce.detach(),
            "group_ce": group_ce.detach(),
            "fine_align": fine_align.detach(),
            "group_align": group_align.detach(),
            "contrastive": contrastive.detach(),
            "hierarchy": hierarchy.detach(),
            "prototype_consistency": (
                fine_proto_consistency + group_proto_consistency
            ).detach(),
        }
        return total, terms
