from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TailThresholds:
    core_max: float
    extreme_min: float


def consensus_sequence(sequences: list[str]) -> str:
    if not sequences:
        raise ValueError("sequences cannot be empty")
    length = len(sequences[0])
    if any(len(sequence) != length for sequence in sequences):
        raise ValueError("all sequences must have equal length")
    alphabet = sorted(set("".join(sequences)))
    encoded = np.asarray([[alphabet.index(aa) for aa in sequence] for sequence in sequences])
    return "".join(alphabet[int(np.bincount(encoded[:, site]).argmax())] for site in range(length))


HYDROPHOBICITY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
CHARGE = {
    "A": 0.0, "R": 1.0, "N": 0.0, "D": -1.0, "C": 0.0, "Q": 0.0, "E": -1.0,
    "G": 0.0, "H": 0.1, "I": 0.0, "L": 0.0, "K": 1.0, "M": 0.0, "F": 0.0,
    "P": 0.0, "S": 0.0, "T": 0.0, "W": 0.0, "Y": 0.0, "V": 0.0,
}
VOLUME = {
    "A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5, "Q": 143.8,
    "E": 138.4, "G": 60.1, "H": 153.2, "I": 166.7, "L": 166.7, "K": 168.6,
    "M": 162.9, "F": 189.9, "P": 112.7, "S": 89.0, "T": 116.1, "W": 227.8,
    "Y": 193.6, "V": 140.0,
}


def mutation_signature(sequence: str, consensus: str) -> str:
    if len(sequence) != len(consensus):
        raise ValueError("sequence and consensus lengths differ")
    positions = [str(i) for i, (aa, ref) in enumerate(zip(sequence, consensus)) if aa != ref]
    return ",".join(positions) or "wt"


def stable_validation_group(signature: str, seed: int, validation_fraction: float) -> bool:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    digest = hashlib.sha256(f"{seed}:{signature}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return value < validation_fraction


def split_fitness_curriculum(
    frame: pd.DataFrame,
    core_quantile: float = 0.90,
    extreme_quantile: float = 0.995,
    validation_fraction: float = 0.10,
    seed: int = 42,
) -> tuple[dict[str, pd.DataFrame], TailThresholds, str]:
    required = {"sequence", "fitness"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if not 0.0 < core_quantile < extreme_quantile < 1.0:
        raise ValueError("quantiles must satisfy 0 < core < extreme < 1")
    data = frame.copy().reset_index(drop=True)
    data["sequence"] = data.sequence.astype(str)
    consensus = consensus_sequence(data.sequence.tolist())
    data["mutation_signature"] = [
        mutation_signature(sequence, consensus) for sequence in data.sequence
    ]
    core_max = float(data.fitness.quantile(core_quantile))
    extreme_min = float(data.fitness.quantile(extreme_quantile))
    data["fitness_band"] = np.where(
        data.fitness >= extreme_min,
        "extreme_test",
        np.where(data.fitness >= core_max, "tail_reward", "core"),
    )
    data["split"] = [
        "test"
        if band == "extreme_test"
        else (
            "validation"
            if stable_validation_group(signature, seed, validation_fraction)
            else "train"
        )
        for band, signature in zip(data.fitness_band, data.mutation_signature)
    ]
    parts = {
        "core_train": data[(data.fitness_band == "core") & (data.split == "train")],
        "core_validation": data[
            (data.fitness_band == "core") & (data.split == "validation")
        ],
        "tail_reward_train": data[
            (data.fitness_band == "tail_reward") & (data.split == "train")
        ],
        "tail_reward_validation": data[
            (data.fitness_band == "tail_reward") & (data.split == "validation")
        ],
        "extreme_test": data[data.fitness_band == "extreme_test"],
    }
    if any(part.empty for part in parts.values()):
        counts = {name: len(part) for name, part in parts.items()}
        raise ValueError(f"curriculum split produced an empty partition: {counts}")
    return (
        {name: part.reset_index(drop=True) for name, part in parts.items()},
        TailThresholds(core_max=core_max, extreme_min=extreme_min),
        consensus,
    )


def scalar_multitask_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    core_threshold: float,
    extreme_threshold: float,
    rank_weight: float = 0.3,
    threshold_weight: float = 0.2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    regression = F.smooth_l1_loss(prediction, target)
    permutation = torch.randperm(len(target), device=target.device)
    gaps = target - target[permutation]
    keep = gaps.abs() >= 0.25
    if bool(keep.any()):
        ranking = F.softplus(
            -gaps[keep].sign()
            * (prediction[keep] - prediction[permutation][keep])
        ).mean()
    else:
        ranking = prediction.new_zeros(())
    scale = prediction.new_tensor(0.5)
    core_bce = F.binary_cross_entropy_with_logits(
        (prediction - core_threshold) / scale,
        (target >= core_threshold).to(prediction.dtype),
    )
    extreme_bce = F.binary_cross_entropy_with_logits(
        (prediction - extreme_threshold) / scale,
        (target >= extreme_threshold).to(prediction.dtype),
    )
    threshold = 0.5 * (core_bce + extreme_bce)
    total = regression + rank_weight * ranking + threshold_weight * threshold
    return total, {
        "regression": regression.detach(),
        "ranking": ranking.detach(),
        "threshold": threshold.detach(),
    }
