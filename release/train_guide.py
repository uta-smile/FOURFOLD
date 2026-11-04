#!/usr/bin/env python3
"""Train the guide ensemble: five members scoring partially masked sequences.

Bottom 90% of fitness pretrains the continuous predictor. The 90%-99.5% band
is then added with extra sampling plus regression/ranking/threshold losses.
The top 0.5% is a blind extreme test and never affects optimisation or model
selection. Writes the split that train_scorer.py reuses, then member_*.pt and
a training_summary.json.

Landscape-agnostic: the dataset enters only through --source-csv, --split-dir
and --output-dir.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from fourfold.tail_curriculum import scalar_multitask_loss, split_fitness_curriculum
from fourfold.oracle import ALPHABET

AA_TO_ID = {aa: index for index, aa in enumerate(ALPHABET)}
MASK_ID = AA_TO_ID["X"]


class MaskedTailPredictor(nn.Module):
    def __init__(self, length: int = 104, hidden: tuple[int, int] = (800, 400)):
        super().__init__()
        self.length = length
        self.classes = len(ALPHABET)
        self.net = nn.Sequential(
            nn.Linear(length * self.classes + 4, hidden[0]),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(hidden[0], hidden[1]),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(hidden[1], 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim == 2:
            one_hot = torch.nn.functional.one_hot(
                tokens.long(), num_classes=self.classes
            ).float()
        elif tokens.ndim == 3 and tokens.shape[-1] == self.classes:
            one_hot = tokens.float()
        else:
            raise ValueError(f"unexpected tokens shape {tuple(tokens.shape)}")
        reveal = 1.0 - one_hot[..., MASK_ID].mean(1)
        time = torch.stack(
            (reveal, reveal.square(), torch.sin(torch.pi * reveal), torch.cos(torch.pi * reveal)),
            dim=-1,
        )
        return self.net(torch.cat((one_hot.flatten(1), time), dim=-1)).squeeze(-1)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode(sequences: list[str]) -> torch.Tensor:
    try:
        return torch.tensor(
            [[AA_TO_ID.get(aa, MASK_ID) for aa in sequence] for sequence in sequences],
            dtype=torch.long,
        )
    except ValueError as exc:
        raise ValueError("all sequences must have equal length") from exc


def random_mask(tokens: torch.Tensor, fractions: torch.Tensor) -> torch.Tensor:
    probabilities = fractions[:, None].expand_as(tokens)
    decisions = torch.rand(tokens.shape, device=tokens.device) < probabilities
    masked = tokens.clone()
    masked[decisions] = MASK_ID
    return masked


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | None]:
    if len(target) < 3:
        return {"n": len(target), "spearman": None, "pearson": None, "rmse": None, "mae": None}
    return {
        "n": int(len(target)),
        "spearman": float(spearmanr(target, prediction).statistic),
        "pearson": float(pearsonr(target, prediction).statistic),
        "rmse": float(np.sqrt(np.mean((target - prediction) ** 2))),
        "mae": float(np.mean(np.abs(target - prediction))),
        "prediction_mean": float(np.mean(prediction)),
        "label_mean": float(np.mean(target)),
    }


@torch.inference_mode()
def predict(
    model: nn.Module,
    tokens: torch.Tensor,
    labels: np.ndarray,
    batch_size: int,
    device: torch.device,
    label_mean: float,
    label_std: float,
    mask_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(seed + mask_count * 1009)
    values = []
    for start in range(0, len(tokens), batch_size):
        batch = tokens[start : start + batch_size].clone()
        if mask_count:
            random_values = torch.rand(batch.shape, generator=generator)
            for row in range(len(batch)):
                sites = torch.topk(random_values[row], min(mask_count, batch.shape[1])).indices
                batch[row, sites] = MASK_ID
        normalized = model(batch.to(device))
        values.append((normalized.float().cpu().numpy() * label_std) + label_mean)
    return labels, np.concatenate(values)


def make_loader(
    tokens: torch.Tensor,
    labels: torch.Tensor,
    bands: torch.Tensor,
    batch_size: int,
    tail_sampling_weight: float,
    seed: int,
) -> DataLoader:
    weights = torch.where(
        bands.bool(),
        torch.full_like(labels, tail_sampling_weight),
        torch.ones_like(labels),
    )
    sampler = WeightedRandomSampler(
        weights=weights.double(),
        num_samples=len(labels),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    return DataLoader(
        TensorDataset(tokens, labels, bands),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
    )


def train_member(args, parts, thresholds, member_index: int, device: torch.device) -> dict:
    seed = args.seed + member_index
    seed_all(seed)
    core_train = parts["core_train"]
    tail_train = parts["tail_reward_train"]
    validation = pd.concat(
        (parts["core_validation"], parts["tail_reward_validation"]), ignore_index=True
    )
    train_all = pd.concat((core_train, tail_train), ignore_index=True)
    label_mean = float(train_all.fitness.mean())
    label_std = float(train_all.fitness.std())
    q95 = float(train_all.fitness.quantile(0.95))
    core_threshold_z = (thresholds.core_max - label_mean) / label_std
    q95_z = (q95 - label_mean) / label_std

    core_tokens = encode(core_train.sequence.tolist())
    core_labels = torch.tensor(
        (core_train.fitness.to_numpy(np.float32) - label_mean) / label_std
    )
    core_bands = torch.zeros(len(core_train), dtype=torch.float32)
    all_tokens = encode(train_all.sequence.tolist())
    all_labels = torch.tensor(
        (train_all.fitness.to_numpy(np.float32) - label_mean) / label_std
    )
    all_bands = torch.tensor(
        (train_all.fitness.to_numpy() >= thresholds.core_max).astype(np.float32)
    )
    validation_tokens = encode(validation.sequence.tolist())
    validation_labels = validation.fitness.to_numpy(np.float32)
    validation_tail = validation.fitness.to_numpy() >= thresholds.core_max
    extreme = parts["extreme_test"]
    extreme_tokens = encode(extreme.sequence.tolist())
    extreme_labels = extreme.fitness.to_numpy(np.float32)

    core_loader = make_loader(
        core_tokens, core_labels, core_bands, args.batch_size, 1.0, seed
    )
    reward_loader = make_loader(
        all_tokens, all_labels, all_bands, args.batch_size,
        args.tail_sampling_weight, seed + 10_000,
    )
    model = MaskedTailPredictor(length=all_tokens.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    mask_values = torch.tensor(args.mask_fractions, device=device)
    mask_probabilities = torch.tensor(args.mask_probabilities, device=device)
    mask_probabilities = mask_probabilities / mask_probabilities.sum()

    member_dir = args.output_dir / f"member_{member_index}"
    member_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_score = -float("inf")
    best_epoch = 0
    stale = 0
    total_epochs = args.core_epochs + args.reward_epochs
    for epoch in range(1, total_epochs + 1):
        stage = "core" if epoch <= args.core_epochs else "tail_reward"
        loader = core_loader if stage == "core" else reward_loader
        model.train()
        running = {"total": 0.0, "regression": 0.0, "ranking": 0.0, "threshold": 0.0}
        seen = 0
        for clean, target, _ in loader:
            clean = clean.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            choices = torch.multinomial(mask_probabilities, len(clean), replacement=True)
            fractions = mask_values[choices]
            masked = random_mask(clean, fractions)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                prediction = model(masked)
                loss, components = scalar_multitask_loss(
                    prediction,
                    target,
                    core_threshold_z,
                    q95_z,
                    rank_weight=args.rank_weight,
                    threshold_weight=args.threshold_weight,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = len(clean)
            seen += count
            running["total"] += float(loss.detach()) * count
            for name, value in components.items():
                running[name] += float(value) * count

        y_val, p_val = predict(
            model, validation_tokens, validation_labels, args.eval_batch_size,
            device, label_mean, label_std, 0, seed,
        )
        overall = regression_metrics(y_val, p_val)
        tail = regression_metrics(y_val[validation_tail], p_val[validation_tail])
        selection_score = float(overall["spearman"]) + 0.5 * float(tail["spearman"])
        record = {
            "epoch": epoch,
            "stage": stage,
            "loss": {name: value / seen for name, value in running.items()},
            "validation": overall,
            "tail_validation": tail,
            "selection_score": selection_score,
        }
        history.append(record)
        checkpoint = {
            "format": "tail_curriculum_mlp_v1",
            "state_dict": model.state_dict(),
            "config": {
                "length": model.length,
                "classes": model.classes,
                "hidden": [800, 400],
                "time_features": 4,
            },
            "normalization": {"mean": label_mean, "std": label_std},
            "training": {
                "seed": seed,
                "epoch": epoch,
                "stage": stage,
                "core_quantile": args.core_quantile,
                "extreme_quantile": args.extreme_quantile,
            },
            "metrics": record,
        }
        epoch_path = member_dir / f"epoch_{epoch:02d}.pt"
        torch.save(checkpoint, epoch_path)
        print(json.dumps({"member": member_index, **record}), flush=True)
        if stage == "tail_reward":
            if selection_score > best_score + args.minimum_delta:
                best_score = selection_score
                best_epoch = epoch
                stale = 0
                shutil.copy2(epoch_path, member_dir / "best.pt")
            else:
                stale += 1
                if stale >= args.patience:
                    break

    best = torch.load(member_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["state_dict"])
    evaluations = {}
    for mask_count in args.eval_mask_counts:
        y_val, p_val = predict(
            model, validation_tokens, validation_labels, args.eval_batch_size,
            device, label_mean, label_std, mask_count, seed,
        )
        y_extreme, p_extreme = predict(
            model, extreme_tokens, extreme_labels, args.eval_batch_size,
            device, label_mean, label_std, mask_count, seed + 77_000,
        )
        evaluations[str(mask_count)] = {
            "validation": regression_metrics(y_val, p_val),
            "tail_validation": regression_metrics(
                y_val[validation_tail], p_val[validation_tail]
            ),
            "extreme_blind_test": regression_metrics(y_extreme, p_extreme),
        }
    final = dict(best)
    final["training"]["best_epoch"] = best_epoch
    final["metrics"] = {"history": history, "evaluations": evaluations}
    torch.save(final, args.output_dir / f"member_{member_index}.pt")
    (member_dir / "metrics.json").write_text(
        json.dumps(
            {
                "member": member_index,
                "seed": seed,
                "best_epoch": best_epoch,
                "history": history,
                "evaluations": evaluations,
            },
            indent=2,
        )
        + "\n"
    )
    return {"member": member_index, "seed": seed, "best_epoch": best_epoch, "evaluations": evaluations}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=Path("vendor/proteinguide/pdg/fitness/data/ube4b_processed.csv"),
    )
    parser.add_argument(
        "--split-dir", type=Path, default=Path("data/split")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("models/guide")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--core-quantile", type=float, default=0.90)
    parser.add_argument("--extreme-quantile", type=float, default=0.995)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--core-epochs", type=int, default=6)
    parser.add_argument("--reward-epochs", type=int, default=14)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--minimum-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--tail-sampling-weight", type=float, default=4.0)
    parser.add_argument("--rank-weight", type=float, default=0.3)
    parser.add_argument("--threshold-weight", type=float, default=0.2)
    parser.add_argument(
        "--mask-fractions",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 0.90],
    )
    parser.add_argument(
        "--mask-probabilities",
        type=float,
        nargs="+",
        default=[0.18, 0.18, 0.16, 0.14, 0.12, 0.10, 0.07, 0.05],
    )
    parser.add_argument(
        "--eval-mask-counts", type=int, nargs="+", default=[0, 5, 10, 25, 50, 75, 94]
    )
    args = parser.parse_args()
    if len(args.mask_fractions) != len(args.mask_probabilities):
        parser.error("mask fractions and probabilities must have equal lengths")

    frame = pd.read_csv(args.source_csv)
    parts, thresholds, consensus = split_fitness_curriculum(
        frame,
        core_quantile=args.core_quantile,
        extreme_quantile=args.extreme_quantile,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    args.split_dir.mkdir(parents=True, exist_ok=True)
    for name, part in parts.items():
        part.to_csv(args.split_dir / f"{name}.csv", index=False)
    manifest = {
        "source_csv": str(args.source_csv),
        "seed": args.seed,
        "core_quantile": args.core_quantile,
        "extreme_quantile": args.extreme_quantile,
        "core_max": thresholds.core_max,
        "extreme_min": thresholds.extreme_min,
        "consensus": consensus,
        "counts": {name: len(part) for name, part in parts.items()},
        "extreme_test_used_for_selection": False,
        "grouping": "mutation-position signature relative to dataset consensus",
    }
    (args.split_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summaries = [
        train_member(args, parts, thresholds, member, device)
        for member in range(args.members)
    ]
    result = {"manifest": manifest, "config": vars(args), "members": summaries}
    result["config"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in result["config"].items()
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
