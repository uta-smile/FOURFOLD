"""ProteinGuide-compatible rate guidance implemented in this project.

The formulas follow Xiong et al.'s public MIT-licensed implementation, pinned
under ``vendor/proteinguide``.  This module is intentionally dependency-light
and is numerically cross-tested against the upstream functions.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch.nn import functional as F

PredictorLogProbability = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def tag_log_ratio(
    predictor_log_probability: PredictorLogProbability,
    tokens: torch.Tensor,
    time: torch.Tensor,
    vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """First-order Taylor estimate of every single-token property log ratio."""
    one_hot = F.one_hot(tokens.long(), num_classes=vocab_size).float()
    with torch.enable_grad():
        one_hot.requires_grad_(True)
        log_probability = predictor_log_probability(one_hot, time)
        gradient = torch.autograd.grad(log_probability.sum(), one_hot)[0]
    current_gradient = (one_hot * gradient).sum(dim=-1, keepdim=True)
    return gradient - current_gradient, log_probability.detach()
