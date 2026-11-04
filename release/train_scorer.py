#!/usr/bin/env python3
"""Train the scorer ensemble: five members scoring complete sequences.

A one-hot MLP, like the guide. An earlier ESM2-based scorer blind-tested at
0.195 Spearman on the top-0.5% holdout against 0.318 for this recipe.

This version reuses the predictor's winning input representation (one-hot
MLP) and multitask loss (regression + pairwise ranking + tail threshold), but
deliberately does *not* use the masked curriculum: the scorer's actual job
(picking the best candidate during local mask-5/10 search) only ever sees
complete sequences, so there is no reason to pay for mask-robustness training
it will never use. That is the one meaningful, use-case-driven difference
from the predictor (which must handle partial/masked context) -- not an
artificial one bolted on for "independence."

Uses the exact same frozen blind split as the predictor (extreme_test.csv
never touched during training/selection) and an independent seed family, so
this remains a genuinely separate model from the predictor, evaluated the
same rigorous way.
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

from fourfold.tail_curriculum import scalar_multitask_loss
from fourfold.oracle import ALPHABET

AA_TO_ID = {aa: index for index, aa in enumerate(ALPHABET)}
MASK_ID = AA_TO_ID["X"]


class CompleteSequenceScorer(nn.Module):
    """One-hot MLP scorer for complete sequences only (no mask curriculum)."""

    def __init__(self, length: int = 104, hidden: tuple[int, int] = (800, 400)):
        super().__init__()
        self.length = length
        self.classes = len(ALPHABET)
        self.net = nn.Sequential(
            nn.Linear(length * self.classes, hidden[0]), nn.GELU(), nn.Dropout(0.0),
            nn.Linear(hidden[0], hidden[1]), nn.GELU(), nn.Dropout(0.0),
            nn.Linear(hidden[1], 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        one_hot = torch.nn.functional.one_hot(tokens.long(), num_classes=self.classes).float()
        return self.net(one_hot.flatten(1)).squeeze(-1)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode(sequences: list[str]) -> torch.Tensor:
    return torch.tensor(
        [[AA_TO_ID.get(aa, MASK_ID) for aa in sequence] for sequence in sequences],
        dtype=torch.long,
    )


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | None]:
    if len(target) < 3:
        return {"n": len(target), "spearman": None, "pearson": None, "rmse": None, "mae": None}
    rng = np.random.default_rng(2026)
    a = rng.integers(0, len(target), 100_000)
    b = rng.integers(0, len(target), 100_000)
    valid = target[a] != target[b]
    pairwise = np.mean(
        np.sign(prediction[a[valid]] - prediction[b[valid]])
        == np.sign(target[a[valid]] - target[b[valid]])
    )
    return {
        "n": int(len(target)),
        "spearman": float(spearmanr(target, prediction).statistic),
        "pearson": float(pearsonr(target, prediction).statistic),
        "rmse": float(np.sqrt(np.mean((target - prediction) ** 2))),
        "mae": float(np.mean(np.abs(target - prediction))),
        "pairwise_accuracy": float(pairwise),
        "prediction_mean": float(np.mean(prediction)),
        "label_mean": float(np.mean(target)),
    }


@torch.inference_mode()
def predict(model, tokens, labels, batch_size, device, label_mean, label_std):
    model.eval()
    values = []
    for start in range(0, len(tokens), batch_size):
        batch = tokens[start : start + batch_size].to(device)
        normalized = model(batch)
        values.append(normalized.float().cpu().numpy() * label_std + label_mean)
    return labels, np.concatenate(values)


def make_loader(tokens, labels, bands, batch_size, tail_sampling_weight, seed):
    weights = torch.where(
        bands.bool(), torch.full_like(labels, tail_sampling_weight), torch.ones_like(labels)
    )
    sampler = WeightedRandomSampler(
        weights=weights.double(), num_samples=len(labels), replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    return DataLoader(
        TensorDataset(tokens, labels, bands), batch_size=batch_size, sampler=sampler,
        num_workers=0, pin_memory=True,
    )


def train_member(args, frames, member_index, device):
    seed = args.seed + member_index
    seed_all(seed)
    core_train = frames["core_train"]
    tail_train = frames["tail_reward_train"]
    validation = pd.concat(
        (frames["core_validation"], frames["tail_reward_validation"]), ignore_index=True
    )
    train_all = pd.concat((core_train, tail_train), ignore_index=True)
    label_mean = float(train_all.fitness.mean())
    label_std = float(train_all.fitness.std())
    core_max = float(tail_train.fitness.min())
    q95 = float(train_all.fitness.quantile(0.95))
    core_threshold_z = (core_max - label_mean) / label_std
    q95_z = (q95 - label_mean) / label_std

    core_tokens = encode(core_train.sequence.tolist())
    core_labels = torch.tensor((core_train.fitness.to_numpy(np.float32) - label_mean) / label_std)
    core_bands = torch.zeros(len(core_train), dtype=torch.float32)
    all_tokens = encode(train_all.sequence.tolist())
    all_labels = torch.tensor((train_all.fitness.to_numpy(np.float32) - label_mean) / label_std)
    all_bands = torch.tensor((train_all.fitness.to_numpy() >= core_max).astype(np.float32))
    validation_tokens = encode(validation.sequence.tolist())
    validation_labels = validation.fitness.to_numpy(np.float32)
    validation_tail = validation.fitness.to_numpy() >= core_max
    extreme = frames["extreme_test"]
    extreme_tokens = encode(extreme.sequence.tolist())
    extreme_labels = extreme.fitness.to_numpy(np.float32)

    core_loader = make_loader(core_tokens, core_labels, core_bands, args.batch_size, 1.0, seed)
    reward_loader = make_loader(
        all_tokens, all_labels, all_bands, args.batch_size, args.tail_sampling_weight, seed + 10_000
    )
    model = CompleteSequenceScorer(length=all_tokens.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

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
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                prediction = model(clean)
                loss, components = scalar_multitask_loss(
                    prediction, target, core_threshold_z, q95_z,
                    rank_weight=args.rank_weight, threshold_weight=args.threshold_weight,
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
            device, label_mean, label_std,
        )
        overall = regression_metrics(y_val, p_val)
        tail = regression_metrics(y_val[validation_tail], p_val[validation_tail])
        selection_score = float(overall["spearman"]) + 0.5 * float(tail["spearman"])
        record = {
            "epoch": epoch, "stage": stage,
            "loss": {name: value / seen for name, value in running.items()},
            "validation": overall, "tail_validation": tail, "selection_score": selection_score,
        }
        history.append(record)
        checkpoint = {
            "format": "one_hot_mlp_scorer",
            "state_dict": model.state_dict(),
            "config": {"length": model.length, "classes": model.classes, "hidden": [800, 400]},
            "normalization": {"mean": label_mean, "std": label_std},
            "training": {"seed": seed, "epoch": epoch, "stage": stage},
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
    y_extreme, p_extreme = predict(
        model, extreme_tokens, extreme_labels, args.eval_batch_size, device, label_mean, label_std
    )
    result = {
        "member": member_index, "seed": seed, "best_epoch": best_epoch,
        "extreme_blind_test": regression_metrics(y_extreme, p_extreme),
    }
    final = dict(best)
    final["training"]["best_epoch"] = best_epoch
    final["metrics"] = {"history": history, "result": result}
    torch.save(final, args.output_dir / f"member_{member_index}.pt")
    (member_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return result, p_extreme


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-dir", type=Path, default=Path("data/split")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("models/scorer"))
    parser.add_argument("--seed", type=int, default=5531)
    parser.add_argument("--members", type=int, default=5)
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
    args = parser.parse_args()

    frames = {
        name: pd.read_csv(args.split_dir / f"{name}.csv")
        for name in (
            "core_train", "core_validation", "tail_reward_train",
            "tail_reward_validation", "extreme_test",
        )
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = []
    extreme_predictions = []
    for member in range(args.members):
        result, p_extreme = train_member(args, frames, member, device)
        results.append(result)
        extreme_predictions.append(p_extreme)

    ensemble_mean = np.mean(extreme_predictions, axis=0)
    extreme_labels = frames["extreme_test"].fitness.to_numpy(np.float32)
    ensemble_metrics = regression_metrics(extreme_labels, ensemble_mean)

    config = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    summary = {
        "config": config,
        "extreme_test_used_for_selection": False,
        "members": results,
        "ensemble_extreme_blind_test": ensemble_metrics,
    }
    print("ENSEMBLE:", json.dumps(ensemble_metrics, indent=2), flush=True)
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
