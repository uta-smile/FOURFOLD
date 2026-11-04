#!/usr/bin/env python3
"""FOURFOLD: run the search over a set of measured starting sequences.

Loads the guide and scorer ensembles and ESM3, builds the mask schedule, and
runs each anchor through the two stages (five rounds at mask width 10, then
ten at width 5). With --fold-guard, ESMFold gates every proposal at mean
Calpha-pLDDT 0.80. The frozen oracle is instantiated only after every search
has finished, and is never read during one.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fourfold.esm3_lora import ESM3LoRAEditor
from fourfold.ensembles import GuideEnsemble, ScorerEnsemble
from fourfold.rejection_navigator import RejectionNavigator
from fourfold.oracle import OfficialFitnessEvaluator
from fourfold.noisy_predictor import (
    ESM3_CANONICAL_IDS, ESM3_MASK_TOKEN, encode_esm3_sequences)
from fourfold.rejection_navigator import CANONICAL_ORDER
from esm.utils.constants.esm3 import SEQUENCE_VOCAB
from search_stage import (
    TargetProbabilityGuide,
    build_schedules,
    mean_pairwise_hamming,
    nearest_hamming,
    search_stage,
)


PIPELINES = {
    "nav_rejection_fast": (
        (10, "current", {"acceptance_rule": "always", "selection_rule": "min_std",
                         "navigator": "rejection_fast"}),
        (5, "current", {"acceptance_rule": "always", "selection_rule": "min_std",
                        "navigator": "rejection_fast"}),
    ),
}


def bootstrap_mean_ci(values: np.ndarray, seed: int, draws: int = 10_000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def parse_anchor_ids(spec: str | None, total: int) -> set[int]:
    """`None` -> every anchor; otherwise a comma list of ids and a-b ranges."""
    if spec is None:
        return set(range(total))
    chosen: set[int] = set()
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            low, high = piece.split("-", 1)
            chosen.update(range(int(low), int(high) + 1))
        else:
            chosen.add(int(piece))
    out_of_range = sorted(i for i in chosen if not 0 <= i < total)
    if out_of_range:
        raise ValueError(f"anchor ids outside 0..{total - 1}: {out_of_range}")
    return chosen


def write_shard(shard_dir: Path, anchor_id: int, payload: dict) -> None:
    """Write one anchor's result atomically, so a kill cannot leave half a file."""
    final = shard_dir / f"anchor_{anchor_id:04d}.json"
    staging = shard_dir / f".anchor_{anchor_id:04d}.json.tmp"
    staging.write_text(json.dumps(payload))
    os.replace(staging, final)


def read_shards(shard_dir: Path, anchors: list[str], pipelines: list[str]) -> dict:
    """Rebuild the in-memory accumulators from every anchor's shard.

    Every anchor must be present and must still carry the sequence it was
    searched from: a shard written against a different anchor pool is a
    different experiment and is refused rather than merged.
    """
    outputs: dict[str, list[str]] = {name: [] for name in pipelines}
    all_candidates: list[dict] = []
    all_triplets: list[dict] = []
    stage_endpoints: list[dict] = []
    missing = []
    for anchor_id, anchor in enumerate(anchors):
        path = shard_dir / f"anchor_{anchor_id:04d}.json"
        if not path.exists():
            missing.append(anchor_id)
            continue
        shard = json.loads(path.read_text())
        if shard["anchor"] != anchor:
            raise ValueError(
                f"shard {path} was searched from a different anchor sequence")
        for name in pipelines:
            outputs[name].append(shard["outputs"][name])
        all_candidates.extend(shard["candidates"])
        all_triplets.extend(shard["triplets"])
        stage_endpoints.extend(shard["stage_endpoints"])
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} anchors not yet searched: {missing[:20]}")
    return {"outputs": outputs, "all_candidates": all_candidates,
            "all_triplets": all_triplets, "stage_endpoints": stage_endpoints}


def finalise(args, anchors: list[str], references: list[str],
             score_min: float, score_max: float, outputs: dict[str, list[str]],
             all_candidates: list[dict], all_triplets: list[dict],
             stage_endpoints: list[dict]) -> None:
    """Score every endpoint with the official oracle and write the run's files.

    Unchanged from when this was the tail of main(); it is a function only so
    that --merge-only can reach it without loading ESM3 or ESMFold.
    """
    pd.DataFrame(all_candidates).to_csv(args.output_dir / "candidate_audit.csv", index=False)
    pd.DataFrame(all_triplets).to_csv(args.output_dir / "candidate_triplets.csv", index=False)
    endpoint_frame = pd.DataFrame(stage_endpoints)
    sequence_frame = pd.DataFrame({"anchor_id": range(len(anchors)), "raw": anchors, **outputs})

    evaluator = OfficialFitnessEvaluator(args.vendor_root, args.dataset)
    raw_oracle = evaluator.predict(anchors, 128)
    sequence_frame["oracle_raw"] = raw_oracle
    endpoint_frame["oracle"] = evaluator.predict(endpoint_frame.sequence.astype(str).tolist(), 128)
    endpoint_frame.to_csv(args.output_dir / "stage_endpoints_and_oracle.csv", index=False)

    triplet_audit_summary: list[dict] = []
    if all_triplets:
        triplet_frame = pd.DataFrame(all_triplets)
        unique_sequences = pd.unique(pd.concat([triplet_frame.winner, triplet_frame.loser]))
        unique_scores = evaluator.predict(unique_sequences.astype(str).tolist(), 128)
        score_map = dict(zip(unique_sequences, unique_scores))
        triplet_frame["oracle_winner"] = triplet_frame.winner.map(score_map)
        triplet_frame["oracle_loser"] = triplet_frame.loser.map(score_map)
        triplet_frame["oracle_delta"] = triplet_frame.oracle_winner - triplet_frame.oracle_loser
        triplet_frame["oracle_correct"] = triplet_frame.oracle_delta > 0.0
        triplet_frame.to_csv(args.output_dir / "triplets_oracle_audit.csv", index=False)
        for keys, frame in triplet_frame.groupby(["pipeline", "stage", "triplet_type"]):
            triplet_audit_summary.append({
                "pipeline": keys[0],
                "stage": keys[1],
                "triplet_type": keys[2],
                "pairs": int(len(frame)),
                "oracle_direction_accuracy": float(frame.oracle_correct.mean()),
                "oracle_delta_mean": float(frame.oracle_delta.mean()),
                "oracle_delta_median": float(frame.oracle_delta.median()),
                "oracle_delta_min": float(frame.oracle_delta.min()),
            })
    summaries = {}
    for pipeline_name, sequences in outputs.items():
        oracle = evaluator.predict(sequences, 128)
        sequence_frame[f"oracle_{pipeline_name}"] = oracle
        gains = oracle - raw_oracle
        distance = np.asarray([
            sum(first != second for first, second in zip(sequence, anchor))
            for sequence, anchor in zip(sequences, anchors)
        ])
        novelty = nearest_hamming(sequences, references)
        ci_low, ci_high = bootstrap_mean_ci(gains, args.seed + 17)
        summaries[pipeline_name] = {
            "oracle_mean": float(oracle.mean()),
            "oracle_median": float(np.median(oracle)),
            "paired_gain_mean": float(gains.mean()),
            "paired_gain_median": float(np.median(gains)),
            "paired_gain_bootstrap_95_ci": [ci_low, ci_high],
            "paired_positive": int(np.sum(gains > 0.0)),
            "paired_negative": int(np.sum(gains < 0.0)),
            "paired_unchanged": int(np.sum(gains == 0.0)),
            "paired_gain_min": float(gains.min()),
            "paired_gain_max": float(gains.max()),
            "hamming_from_initial_mean": float(distance.mean()),
            "diversity": mean_pairwise_hamming(sequences),
            "novelty_mean": float(novelty.mean()),
            "triplets": int(sum(row["pipeline"] == pipeline_name for row in all_triplets)),
        }
    sequence_frame.to_csv(args.output_dir / "sequences_and_oracle.csv", index=False)
    summary = {
        "design": "paired random unfiltered Guide anchors; oracle loaded only after all searches",
        "anchors": len(anchors),
        "raw_oracle_mean": float(raw_oracle.mean()),
        "raw_oracle_median": float(np.median(raw_oracle)),
        "score_range_gate": [score_min, score_max],
        "pipelines": summaries,
        "triplet_oracle_audit": triplet_audit_summary,
        "config": vars(args),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, default=str, indent=2) + "\n")
    print(json.dumps(summary, default=str, indent=2))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--guide-pool", type=Path,
        default=Path("anchors/starts.csv"),
    )
    parser.add_argument(
        "--training-csv", type=Path,
        default=Path("data/split/core_train.csv"),
    )
    parser.add_argument(
        "--predictor-dir", type=Path,
        default=Path("models/guide"),
    )
    parser.add_argument(
        "--scorer-dir", type=Path,
        default=Path("models/scorer"),
    )
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--rejection-threshold", type=float, default=4.0)
    parser.add_argument(
        "--newpos-schedule", type=Path, default=None,
        help="json {anchor_id: {'mask10:0': n, ...}} of newly opened positions per "
             "round, replayed by the *_matched arms so their edit radius matches "
             "the reference arm's by construction",
    )
    parser.add_argument("--screen-groups", type=int, default=8)
    parser.add_argument("--candidate-pool", type=int, default=15)
    parser.add_argument("--risk-model-dir", type=Path, help="directory holding risk_model.joblib")
    parser.add_argument("--position-weights", type=Path, help="npy file of per-position sampling weights")
    parser.add_argument("--veto-rate-map", type=Path,
                        help="json of {anchor_id: rate}; arms whose random_veto_rate is "
                             "'matched' use the rate recorded for that anchor")
    parser.add_argument("--vendor-root", type=Path, default=Path("vendor/proteinguide"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("runs/fourfold"),
    )
    parser.add_argument(
        "--dataset", choices=("ube4b", "pabp", "creilov"), default="ube4b",
        help="selects the official frozen oracle used for final evaluation; "
             "predictor, scorer, guide pool, training csv and rejection "
             "threshold are passed separately so nothing is implicit",
    )
    parser.add_argument("--anchors", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2030)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--sampling-temperature", type=float, default=0.7,
                        help="generator sampling temperature. 0.7 is the default and "
                             "leaves every reported run "
                             "unchanged; raising it only affects proposal "
                             "diversity, not the R criterion or the selector")
    parser.add_argument("--fold-guard", action="store_true",
                        help="fold each approved move and send it back if the "
                             "structure degrades; tries the next-ranked candidate")
    parser.add_argument("--fold-floor", type=float, default=0.70,
                        help="absolute mean-pLDDT floor (CA atom, 0-1)")
    parser.add_argument("--fold-max-drop", type=float, default=0.03,
                        help="largest drop from the anchor's own pLDDT that is "
                             "tolerated; assayed variants sit at 0.857 +/- 0.013, "
                             "so 0.03 is already outside normal variation")
    parser.add_argument("--fold-max-tries", type=int, default=3,
                        help="candidates tried per generated batch")
    parser.add_argument("--refine-threshold", type=float, default=None,
                        help="PRO-SOUNDS' refinement: remask residues the generator "
                             "gives less than this of the vocabulary mass (their 0.70)")
    parser.add_argument("--refine-lambda", type=float, default=0.0,
                        help="weight on our guidance inside the refinement step; "
                             "0 is PRO-SOUNDS' unguided resample")
    parser.add_argument("--refine-temperature", type=float, default=0.7)
    parser.add_argument("--refine-top-p", type=float, default=0.9)
    parser.add_argument("--fold-shrink", action="store_true",
                        help="on guard failure fall back to a narrower mask "
                             "instead of redrawing at the same width")
    parser.add_argument("--fold-max-batches", type=int, default=4,
                        help="fresh batches drawn when a whole batch folds badly; "
                             "the round is skipped only after all of them fail")
    parser.add_argument("--rounds1", type=int, default=40,
                        help="single-position rounds in the schedule; 40 is what "
                             "every reported run used and is kept as the default")
    parser.add_argument("--edit-budget", type=int, default=None,
                        help="stop a search once it is this many substitutions "
                             "from its ORIGINAL anchor; off by default")
    parser.add_argument("--rounds10", type=int, default=10)
    parser.add_argument("--candidates10", type=int, default=10)
    parser.add_argument("--rounds5", type=int, default=20)
    parser.add_argument("--candidates5", type=int, default=20)
    parser.add_argument("--strict-margin", type=float, default=3.0)
    parser.add_argument("--minimum-agreement", type=float, default=0.8)
    parser.add_argument("--minimum-z", type=float, default=1.0)
    parser.add_argument("--residual-sigma", type=float, default=2.31)
    parser.add_argument(
        "--guide-form", choices=("linear", "probit"), default="linear",
        help="the guidance field. 'linear' is fitness-directed mean guidance and "
             "is what every result in this project was produced under; its "
             "gradient ignores --rejection-threshold, which is therefore inert "
             "under it. 'probit' is log P(fitness > threshold), where the "
             "threshold does act. The two open different positions and their "
             "results must not be pooled. Recorded in summary.json so a run "
             "always states which method produced it.")
    parser.add_argument("--guide-temperature", type=float, default=0.1)
    parser.add_argument(
        "--pipelines", nargs="+", choices=tuple(PIPELINES), default=tuple(PIPELINES),
    )
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--shard-dir", type=Path, default=None,
        help="write one JSON file per finished anchor here and skip anchors "
             "already written; nothing is scored until --merge-only")
    parser.add_argument(
        "--anchor-ids", default=None,
        help="restrict the search to these anchors of the sampled pool, as a "
             "comma list of ids and a-b ranges, e.g. 0-24,37")
    parser.add_argument(
        "--merge-only", action="store_true",
        help="search nothing; rebuild the run's files from --shard-dir and "
             "score them with the official oracle")
    args = parser.parse_args()

    required = [
        args.guide_pool, args.training_csv, args.predictor_dir,
        args.scorer_dir, args.vendor_root,
    ]
    if args.adapter_dir is not None:
        required.append(args.adapter_dir)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    if args.anchors < 2:
        raise ValueError("--anchors must be at least 2")
    pool_size = len(pd.read_csv(args.guide_pool, usecols=["sequence"]))
    if args.anchors > pool_size:
        raise ValueError(f"requested {args.anchors} anchors from pool of {pool_size}")
    if args.check_only:
        print(json.dumps({"status": "ok", "pool_size": pool_size, "config": vars(args)}, default=str, indent=2))
        return
    if args.merge_only:
        if args.shard_dir is None:
            raise ValueError("--merge-only needs --shard-dir")
        pool = pd.read_csv(args.guide_pool).sample(
            n=args.anchors, random_state=args.seed).reset_index(drop=True)
        anchors = pool.sequence.astype(str).tolist()
        training = pd.read_csv(args.training_csv)
        references = training.sequence.astype(str).tolist()
        merged = read_shards(args.shard_dir, anchors, list(args.pipelines))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        finalise(args, anchors, references,
                 float(training.fitness.min()) - 0.5,
                 float(training.fitness.max()) + 0.5,
                 merged["outputs"], merged["all_candidates"],
                 merged["all_triplets"], merged["stage_endpoints"])
        return

    if args.shard_dir is None and args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.shard_dir is not None:
        args.shard_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    pool = pd.read_csv(args.guide_pool).sample(n=args.anchors, random_state=args.seed).reset_index(drop=True)
    anchors = pool.sequence.astype(str).tolist()
    selected = parse_anchor_ids(args.anchor_ids, len(anchors))
    training = pd.read_csv(args.training_csv)
    references = training.sequence.astype(str).tolist()
    score_min = float(training.fitness.min()) - 0.5
    score_max = float(training.fitness.max()) + 0.5

    base_guide = GuideEnsemble.load(
        args.predictor_dir, device,
        guide_form=args.guide_form, residual_sigma=args.residual_sigma)
    tail_guide = TargetProbabilityGuide(base_guide, args.residual_sigma)
    scorer = ScorerEnsemble.load(args.scorer_dir, device, uncertainty_penalty=0.5)
    fold_plddt = None
    if args.fold_guard:
        from fourfold.esmfold import load_model as _load_esmfold
        _tokenizer, _esmfold = _load_esmfold()
        _seen: dict[str, float] = {}

        def fold_plddt(sequence: str) -> float:
            """Mean pLDDT over CA atoms, memoised.

            CA-only, matching every structural number reported for this work
            (wild type 0.856, assayed 0.857, our d<=6 arms 0.842). ProteinGuide's
            own metric averages the 37 atom slots instead and runs ~0.044 lower,
            so thresholds here are on the CA scale and are not their 0.8 gate.
            """
            if sequence not in _seen:
                batch = _tokenizer([sequence], return_tensors="pt", add_special_tokens=False)
                with torch.inference_mode():
                    out = _esmfold(batch["input_ids"].to(device),
                                   attention_mask=batch["attention_mask"].to(device))
                _seen[sequence] = float(out["plddt"][0, :, 1].mean())
            return _seen[sequence]
    refiner = None
    if args.refine_threshold is not None:
        from fourfold.esm3_lora import ESM3_BOS_TOKEN, ESM3_EOS_TOKEN

        _order = {a: i for i, a in enumerate(CANONICAL_ORDER)}

        def _full_logits(sequence: str) -> torch.Tensor:
            tokens = encode_esm3_sequences([sequence]).to(device)
            edge = lambda v: torch.full((1, 1), v, dtype=torch.long, device=device)
            wrapped = torch.cat((edge(ESM3_BOS_TOKEN), tokens, edge(ESM3_EOS_TOKEN)), 1)
            model = editor.model
            with torch.inference_mode(), model.disable_adapter(), torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16,
                    enabled=device.type == "cuda"):
                return model(sequence_tokens=wrapped).sequence_logits[0, 1:-1].float()

        _canonical_ids = torch.tensor(
            [SEQUENCE_VOCAB.index(a) for a in CANONICAL_ORDER], device=device)

        def refiner(sequence: str) -> str:
            held = torch.softmax(_full_logits(sequence), -1)[
                range(len(sequence)), [ESM3_CANONICAL_IDS[a] for a in sequence]]
            low = [i for i in range(len(sequence))
                   if float(held[i]) < args.refine_threshold]
            if not low:
                return sequence
            tokens = encode_esm3_sequences([sequence]).to(device)
            tokens[:, torch.as_tensor(low, device=device)] = ESM3_MASK_TOKEN
            logits = editor.sequence_logits_tokens(tokens, reference=True)[0].float()
            guidance = None
            if args.refine_lambda:
                target = float(scorer.predict_distribution([sequence], 128)[0][0]) + 0.25
                ratio, _ = tail_guide.tag_log_ratio(
                    tokens, torch.zeros(1, device=device), target, vocab_size=33)
                guidance = ratio.detach()[0][:, _canonical_ids].float()
            letters = list(sequence)
            for site in low:
                row = logits[site].clone()
                if guidance is not None:
                    row = row + args.refine_lambda * args.refine_temperature * guidance[site]
                row[_order[sequence[site]]] = -1e8
                probabilities = torch.softmax(row / args.refine_temperature, -1)
                ordered, index = torch.sort(probabilities, descending=True)
                keep = (ordered.cumsum(0) - ordered) < args.refine_top_p
                ordered = torch.where(keep, ordered, torch.zeros_like(ordered))
                pick = int(index[int(torch.multinomial(ordered / ordered.sum(), 1))])
                letters[site] = CANONICAL_ORDER[pick]
            return "".join(letters)

    if args.adapter_dir is None:
        editor = ESM3LoRAEditor.from_pretrained(device, "esm3_sm_open_v1", 16, 32, 0.05)
    else:
        editor = ESM3LoRAEditor.load_adapter(
            args.adapter_dir, device, "esm3_sm_open_v1", trainable=False
        )
    guides = {"current": base_guide, "safe_tail": tail_guide}
    veto_rate_map = None
    if args.veto_rate_map is not None:
        veto_rate_map = {int(k): float(v) for k, v in json.loads(args.veto_rate_map.read_text()).items()}
        print(json.dumps({"veto_rate_map": str(args.veto_rate_map),
                          "anchors": len(veto_rate_map),
                          "mean_rate": sum(veto_rate_map.values()) / len(veto_rate_map)}), flush=True)
    position_weights = None
    if args.position_weights is not None:
        position_weights = np.load(args.position_weights)
        print(json.dumps({
            "position_weights": str(args.position_weights),
            "min_relative": float(position_weights.min() * len(position_weights) / position_weights.sum()),
            "max_relative": float(position_weights.max() * len(position_weights) / position_weights.sum()),
        }), flush=True)
    risk_model = None
    if args.risk_model_dir is not None:
        import joblib

        risk_model = joblib.load(args.risk_model_dir / "risk_model.joblib")
        print(json.dumps({
            "risk_model": str(args.risk_model_dir),
            "threshold": risk_model["threshold"],
            "features": risk_model["features"],
        }), flush=True)

    newpos_schedule = (
        json.loads(args.newpos_schedule.read_text()) if args.newpos_schedule else None
    )
    outputs: dict[str, list[str]] = {name: [] for name in args.pipelines}
    all_candidates: list[dict] = []
    all_triplets: list[dict] = []
    stage_endpoints: list[dict] = []
    for anchor_id, anchor in enumerate(anchors):
        if anchor_id not in selected:
            continue
        if args.shard_dir is not None and (
                args.shard_dir / f"anchor_{anchor_id:04d}.json").exists():
            print(json.dumps({"anchor": anchor_id + 1, "total": len(anchors),
                              "resumed": True}), flush=True)
            continue
        anchor_candidates: list[dict] = []
        anchor_triplets: list[dict] = []
        anchor_endpoints: list[dict] = []
        anchor_outputs: dict[str, str] = {}
        schedules = {
            variant: build_schedules(
                len(anchor), args.seed + anchor_id * 1009 + variant * 7919,
                args.rounds10, args.rounds5, rounds1=args.rounds1,
            )
            for variant in range(5)
        }
        if position_weights is not None:
            schedules["informed"] = build_schedules(
                len(anchor), args.seed + anchor_id * 1009,
                args.rounds10, args.rounds5, rounds1=args.rounds1,
                position_weights=position_weights,
            )
        schedule = schedules[0]
        for pipeline_id, pipeline_name in enumerate(args.pipelines):
            current = anchor
            for stage_spec in PIPELINES[pipeline_name]:
                mask_count, variant = stage_spec[0], stage_spec[1]
                stage_kwargs = dict(stage_spec[2]) if len(stage_spec) > 2 else {}
                if "mutation_budget" in stage_kwargs:
                    stage_kwargs.setdefault("budget_reference", anchor)
                if stage_kwargs.get("random_veto_rate") == "matched":
                    if veto_rate_map is None:
                        raise ValueError("pipeline needs --veto-rate-map")
                    stage_kwargs["random_veto_rate"] = veto_rate_map.get(anchor_id, 0.0)
                if stage_kwargs.pop("use_risk_model", False):
                    if risk_model is None:
                        raise ValueError("pipeline needs --risk-model-dir")
                    stage_kwargs["risk_model"] = risk_model
                    stage_kwargs["risk_reference"] = anchor
                if mask_count == 10:
                    candidates = args.candidates10
                elif mask_count == 5:
                    candidates = args.candidates5
                else:
                    candidates = min(10, args.candidates5)
                if fold_plddt is not None:
                    stage_kwargs.setdefault("structure_guard", {
                        "fold": fold_plddt,
                        "reference": fold_plddt(anchor),
                        "floor": args.fold_floor,
                        "max_drop": args.fold_max_drop,
                        "max_tries": args.fold_max_tries,
                        "max_batches": args.fold_max_batches,
                        "shrink": args.fold_shrink,
                    })
                if refiner is not None:
                    stage_kwargs.setdefault("refiner", refiner)
                if args.edit_budget is not None:
                    stage_kwargs.setdefault("mutation_budget", args.edit_budget)
                    stage_kwargs.setdefault("budget_reference", anchor)
                navigator_key = stage_kwargs.pop("navigator", None)
                if navigator_key == "rejection_fast":
                    stage_kwargs["mask_navigator"] = RejectionNavigator(
                        editor, base_guide, threshold=args.rejection_threshold,
                        guide_temperature=args.guide_temperature,
                        screen_groups=args.screen_groups,
                        candidate_pool=args.candidate_pool)
                elif navigator_key is not None:
                    raise ValueError(f"unknown navigator: {navigator_key}")
                current, rows, triplets = search_stage(
                    editor=editor,
                    guide=guides[stage_kwargs.pop("guide_key", variant)],
                    scorer=scorer,
                    initial=current,
                    positions_schedule=schedules[stage_kwargs.pop("schedule_variant", 0)][mask_count],
                    mask_count=mask_count,
                    candidates_per_round=candidates,
                    rng_seed=args.seed + anchor_id * 100_003 + mask_count,
                    variant=variant,
                    minimum_delta=args.strict_margin,
                    minimum_agreement=args.minimum_agreement,
                    minimum_z=args.minimum_z,
                    score_min=score_min,
                    score_max=score_max,
                    guide_temperature=stage_kwargs.pop("guide_temperature", args.guide_temperature),
                    property_threshold=-1.5,
                    dynamic_target_gain=0.25,
                    patience=4,
                    scorer_batch_size=128,
                    reference_policy=args.adapter_dir is None,
                    sampling_temperature=args.sampling_temperature,
                    **stage_kwargs,
                )
                for row in rows:
                    row.update({"anchor_id": anchor_id, "pipeline": pipeline_name})
                for row in triplets:
                    row.update({"anchor_id": anchor_id, "pipeline": pipeline_name})
                anchor_candidates.extend(rows)
                anchor_triplets.extend(triplets)
                anchor_endpoints.append({
                    "anchor_id": anchor_id,
                    "pipeline": pipeline_name,
                    "stage": f"mask{mask_count}",
                    "sequence": current,
                })
            anchor_outputs[pipeline_name] = current
            print(json.dumps({
                "anchor": anchor_id + 1,
                "total": len(anchors),
                "pipeline": pipeline_name,
            }), flush=True)

        all_candidates.extend(anchor_candidates)
        all_triplets.extend(anchor_triplets)
        stage_endpoints.extend(anchor_endpoints)
        for name in args.pipelines:
            outputs[name].append(anchor_outputs[name])
        if args.shard_dir is not None:
            write_shard(args.shard_dir, anchor_id, {
                "anchor_id": anchor_id,
                "anchor": anchor,
                "outputs": anchor_outputs,
                "candidates": anchor_candidates,
                "triplets": anchor_triplets,
                "stage_endpoints": anchor_endpoints,
            })

    if args.shard_dir is not None:
        print(json.dumps({"status": "shards_written",
                          "searched": sorted(selected)}), flush=True)
        return

    finalise(args, anchors, references, score_min, score_max,
             outputs, all_candidates, all_triplets, stage_endpoints)


if __name__ == "__main__":
    main()
