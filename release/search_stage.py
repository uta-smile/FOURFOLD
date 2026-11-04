#!/usr/bin/env python3
"""One mask stage: the round loop that calls the four decisions.

`search_stage` runs a stage at a fixed mask width, asking WHERE for the
positions, WHAT for the candidates, WHICH for the one to carry forward and
WHETHER whether it may become the next state, retrying while the structural
check refuses. The evaluation oracle is never consulted here.
"""
from __future__ import annotations


import numpy as np
import torch
import torch.nn.functional as F

from fourfold.direct_exo import GeneratedCandidate, sample_local_group
from fourfold.ensembles import (
    GuideEnsemble,
    ScorerEnsemble,
    encode_sequences,
)


class TargetProbabilityGuide:
    """Threshold-dependent differentiable tail probability over a locked ensemble."""

    def __init__(self, base: GuideEnsemble, residual_sigma: float):
        if residual_sigma <= 0:
            raise ValueError("residual_sigma must be positive")
        self.base = base
        self.device = base.device
        self.residual_sigma = float(residual_sigma)

    def _log_probability_onehot(self, esm3_one_hot: torch.Tensor, threshold: float) -> torch.Tensor:
        alphabet_one_hot = esm3_one_hot.to(self.device) @ self.base.onehot_matrix
        members = []
        for model, mean, std in self.base.models:
            members.append(model(alphabet_one_hot).float() * std + mean)
        stacked = torch.stack(members)
        ensemble_mean = stacked.mean(0)
        ensemble_std = stacked.std(0, unbiased=False)
        total_sigma = torch.sqrt(ensemble_std.square() + self.residual_sigma ** 2)
        z = (ensemble_mean - float(threshold)) / total_sigma.clamp_min(1e-6)
        return torch.special.log_ndtr(z)

    def tag_log_ratio(
        self, esm3_tokens: torch.Tensor, t: torch.Tensor, threshold: float,
        vocab_size: int = 33,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del t
        one_hot = F.one_hot(esm3_tokens.long(), num_classes=vocab_size).float()
        with torch.enable_grad():
            one_hot.requires_grad_(True)
            log_probability = self._log_probability_onehot(one_hot, threshold)
            gradient = torch.autograd.grad(log_probability.sum(), one_hot)[0]
        current = (one_hot * gradient).sum(-1, keepdim=True)
        return gradient - current, log_probability.detach()

    @torch.inference_mode()
    def log_probability_above(self, esm3_tokens: torch.Tensor, threshold: float) -> torch.Tensor:
        one_hot = F.one_hot(esm3_tokens.long(), num_classes=33).float()
        return self._log_probability_onehot(one_hot, threshold)


def scorer_member_predictions(
    scorer: ScorerEnsemble, sequences: list[str], batch_size: int,
) -> np.ndarray:
    tokens = encode_sequences(sequences)
    members = []
    with torch.inference_mode():
        for model, mean, std in scorer.models:
            chunks = []
            for start in range(0, len(tokens), batch_size):
                batch = tokens[start:start + batch_size].to(scorer.device)
                chunks.append((model(batch).float().cpu().numpy() * std + mean))
            members.append(np.concatenate(chunks))
    return np.stack(members)


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def position_reliability(scorer, sequence: str, batch_size: int) -> np.ndarray:
    """Per-position score for how confidently the scorer can change a residue.

    Every single substitution of every position is scored, and the position's
    value is the smallest ensemble disagreement any substitution there attains:

        R_j = min over a of std of the scorer ensemble on x with position j set to a

    A low R_j means the position has at least one replacement the ensemble
    agrees about. This uses the scorer only, never the generator and never the
    evaluator, so it is computable at deployment time without any label.
    """
    variants = [
        sequence[:j] + aa + sequence[j + 1:]
        for j in range(len(sequence))
        for aa in AMINO_ACIDS
    ]
    members = scorer_member_predictions(scorer, variants, batch_size)
    spread = members.std(axis=0).reshape(len(sequence), len(AMINO_ACIDS))
    return spread.min(axis=1)


def position_gain(scorer, sequence: str, batch_size: int) -> np.ndarray:
    """Per-position best achievable gain the scorer believes in.

        G_j = max over a of  conservative(x with position j set to a) - conservative(x)

    The rejection criterion R ranks positions by how much ESM3's prior and the
    fitness guide disagree about the residue already there. That is not an
    estimate of what editing the position can win, and when only a handful of
    edits are allowed the ranking is the whole game: at d <= 15 R is
    indistinguishable from choosing positions at random (paired +0.010,
    51/100). Inside that range the calibrated object is the scorer itself, so
    this ranks positions by what the scorer says is reachable there.

    Conservative means mean - penalty * sd, the same quantity the acceptance
    rule compares, so a position is only attractive if its best substitution is
    one the ensemble also agrees about.
    """
    variants = [
        sequence[:j] + aa + sequence[j + 1:]
        for j in range(len(sequence))
        for aa in AMINO_ACIDS
    ]
    members = scorer_member_predictions(scorer, variants, batch_size)
    value = (members.mean(axis=0) - scorer.uncertainty_penalty * members.std(axis=0))
    value = value.reshape(len(sequence), len(AMINO_ACIDS))
    here = scorer_member_predictions(scorer, [sequence], batch_size)
    incumbent = float(here.mean() - scorer.uncertainty_penalty * here.std())
    return value.max(axis=1) - incumbent


def reliability_positions(
    scorer, sequence: str, mask_count: int, policy: str, batch_size: int,
) -> tuple[int, ...]:
    """Mask positions chosen by R_j instead of by the random schedule."""
    reliability = position_reliability(scorer, sequence, batch_size)
    order = np.argsort(reliability, kind="stable")
    if policy == "max_gain":
        return tuple(int(j) for j in sorted(
            np.argsort(-position_gain(scorer, sequence, batch_size),
                       kind="stable")[:mask_count]))
    if policy == "min_sigma":
        chosen = order[:mask_count]
    elif policy == "max_sigma":
        chosen = order[-mask_count:]
    else:
        raise ValueError(f"unknown mask policy {policy!r}")
    return tuple(int(j) for j in sorted(chosen))


def candidate_statistics(current_members: np.ndarray, candidate_members: np.ndarray) -> dict[str, np.ndarray]:
    deltas = candidate_members - current_members[:, None]
    mean = candidate_members.mean(0)
    std = candidate_members.std(0)
    delta_mean = deltas.mean(0)
    delta_std = deltas.std(0)
    direction = np.sign(delta_mean)
    agreement = (np.sign(deltas) == direction[None, :]).mean(0)
    z_margin = np.abs(delta_mean) / np.maximum(delta_std, 1e-6)
    conservative = mean - 0.5 * std
    return {
        "mean": mean, "std": std, "delta_mean": delta_mean,
        "delta_std": delta_std, "agreement": agreement,
        "z_margin": z_margin, "conservative": conservative,
    }


def mean_pairwise_hamming(sequences: list[str]) -> float:
    if len(sequences) < 2:
        return 0.0
    values = []
    for i, first in enumerate(sequences):
        for second in sequences[i + 1:]:
            values.append(sum(a != b for a, b in zip(first, second)))
    return float(np.mean(values))


def nearest_hamming(sequences: list[str], references: list[str]) -> np.ndarray:
    query = np.asarray([list(s) for s in sequences], dtype="U1")
    output = np.full(len(query), 10_000, dtype=np.int32)
    for start in range(0, len(references), 2000):
        ref = np.asarray([list(s) for s in references[start:start + 2000]], dtype="U1")
        output = np.minimum(output, (query[:, None, :] != ref[None, :, :]).sum(2).min(1))
    return output


def build_schedules(
    length: int, seed: int, rounds10: int, rounds5: int, rounds1: int = 0,
    position_weights: np.ndarray | None = None,
) -> dict[int, list[tuple[int, ...]]]:
    """Position schedules for each mask stage.

    ``position_weights`` biases which positions a round exposes. Historical
    single-position edits show that positions differ enormously in how often
    editing them ends in a large fitness loss, and that difference survives
    controlling for sequence background, so sampling positions uniformly
    spends the same generation budget on positions that are almost always
    harmful as on positions that are almost always safe.
    """
    rng = np.random.default_rng(seed)
    probabilities = None
    if position_weights is not None:
        probabilities = np.asarray(position_weights, dtype=float)[:length]
        probabilities = probabilities / probabilities.sum()
    schedule = {
        10: [tuple(rng.choice(length, 10, replace=False, p=probabilities).astype(int).tolist())
             for _ in range(rounds10)],
        5: [tuple(rng.choice(length, 5, replace=False, p=probabilities).astype(int).tolist())
            for _ in range(rounds5)],
    }
    for width in (2, 3):
        schedule[width] = [
            tuple(rng.choice(length, width, replace=False, p=probabilities).astype(int).tolist())
            for _ in range(rounds5)
        ]
    if rounds1 > 0:
        schedule[1] = [tuple(rng.choice(length, 1, replace=False, p=probabilities).astype(int).tolist())
                       for _ in range(rounds1)]
    return schedule


def search_stage(
    *, editor, guide, scorer, initial: str, positions_schedule: list[tuple[int, ...]],
    mask_count: int, candidates_per_round: int, rng_seed: int, variant: str,
    minimum_delta: float, minimum_agreement: float, minimum_z: float,
    score_min: float, score_max: float, guide_temperature: float,
    property_threshold: float, dynamic_target_gain: float, patience: int,
    scorer_batch_size: int, reference_policy: bool = True,
    uncertainty_halt: bool = False, uncertainty_window: int = 3,
    uncertainty_min_agreement: float = 0.55, uncertainty_min_zmargin: float = 0.5,
    mutation_budget: int | None = None, budget_reference: str | None = None,
    risk_model: dict | None = None, risk_reference: str | None = None,
    structure_guard: dict | None = None,
    refiner=None,
    random_veto_rate: float = 0.0,
    acceptance_rule: str = "margin", strict_acceptance_margin: float = 3.0,
    selection_rule: str = "conservative", guided: bool = True,
    mask_policy: str | None = None,
    mask_navigator=None,
    sampling_temperature: float = 0.7,
) -> tuple[str, list[dict], list[dict]]:
    """Search one mask-stage's rounds, optionally with a rolling uncertainty halt.

    ``uncertainty_halt`` (default off, fully backward compatible) does not
    change the acceptance rule at all -- it uses the exact same
    accept/reject test as the ``variant`` already selects. It only adds a
    second, independent stopping condition: after each ACCEPTED round, track
    a rolling window of that round's scorer-ensemble ``agreement`` and
    ``z_margin``. If both have been trending low across the last
    ``uncertainty_window`` accepted rounds, stop the stage early and keep the
    current sequence, instead of continuing to climb on a signal the
    ensemble itself is no longer confident about. This is a cheap, rule-based
    stand-in for a "does the search notice it's drifting off-distribution
    and back off" controller -- distinct from tightening the acceptance bar
    (already tested and found to make things worse), since it never rejects
    a step the existing rule would have accepted; it only decides *when to
    stop asking for more*.

    ``acceptance_rule`` selects how much the accept decision leans on the
    surrogate's calibrated scale, and nothing else about the search changes
    with it. ``rank`` asks only whether the best candidate outscores the
    incumbent; ``margin`` additionally demands a fixed absolute improvement,
    which is the pipeline's historical rule; ``strict_margin`` demands a much
    larger one. Measured landscapes show ranking outlasting calibration under
    mutation-depth shift, so the three form a dose-response in how early they
    should be expected to fail as a search moves away from measured support.

    ``mutation_budget`` (default off) is a second, independent and much
    simpler halting rule: stop the stage once the accepted sequence's
    Hamming distance from a reference sequence reaches the cap, regardless
    of what the scorer ensemble reports. This is the PEX / MULTI-evolve-style
    proximal-exploration proxy -- our own scorer/predictor pair is only ever
    validated up to roughly 6 mutations from a real training sequence, so a
    hard distance cap does not depend on the ensemble's own (possibly
    overconfident) self-assessment at all. ``budget_reference`` sets what the
    distance is measured against (pass the ORIGINAL anchor, not this stage's
    own start, to cap total drift across both the mask10 and mask5 stages
    together); it defaults to this stage's own ``initial`` when omitted.

    ``structure_guard`` folds the move the acceptance rule approved and sends it
    back if the fold degrades. It is a gradient constraint, not a floor: a floor
    on pLDDT is a distance cap in disguise, because pLDDT tracks edit distance
    here at Spearman -0.82, and an exact distance cap is already available for
    free. What this catches instead is a path that is structurally expensive per
    edit -- measured on this landscape, the rejection criterion at d<=15 makes
    FEWER edits than random (11.5 vs 14.9) and folds WORSE (0.775 vs 0.833),
    so it opens positions that cost more structure per change.

    It does NOT catch a surrogate being exploited: at a fixed d=6 the max-gain
    arm drives ensemble disagreement from 0.144 to 0.733 and folds 0.007 BETTER,
    so the guard would pass it. Structure and disagreement are separate gates.

    Keys: ``fold`` (sequence -> mean pLDDT), ``reference`` (the anchor's own
    pLDDT), ``max_drop``, ``floor``, ``max_tries``. Candidates are tried in the
    order the selection rule ranks them and the first that survives is taken.

    ``risk_model`` vetoes a move the acceptance rule already approved, when a
    learned classifier predicts the move is one of the rare large harmful
    steps. ``random_veto_rate`` is its control: it vetoes approved moves at a
    fixed rate with no discrimination at all, which separates "this model
    can tell good moves from catastrophic ones" from "taking fewer steps is
    simply better", something already known to be true here.
    """
    current = initial
    current_members = scorer_member_predictions(scorer, [current], scorer_batch_size)[:, 0]
    stale = 0
    round_rows: list[dict] = []
    triplets: list[dict] = []
    rng = np.random.default_rng(rng_seed)
    safe = variant != "current"
    accepted_agreement_history: list[float] = []
    accepted_zmargin_history: list[float] = []
    for round_id, positions in enumerate(positions_schedule):
        if mask_navigator is not None:
            positions = mask_navigator(current, len(positions), round_id)
        elif mask_policy is not None:
            positions = reliability_positions(
                scorer, current, len(positions), mask_policy, scorer_batch_size,
            )
        current_mean = float(current_members.mean())
        target = max(property_threshold, current_mean + dynamic_target_gain)
        def propose(where=None):
            """One fresh batch of candidates for this round, scored.

            Called again by the structure guard when a whole batch folds badly:
            the generator draws from `rng`, whose state has advanced, so a second
            call is a genuinely new proposal rather than the same one re-ranked.
            """
            generated = sample_local_group(
                editor, guide, current, positions if where is None else where,
                candidates_per_round, rng,
                guided=guided, guide_temperature=guide_temperature,
                property_threshold=target, sampling_temperature=sampling_temperature,
                sampling_top_k=10, sampling_top_p=0.95,
                reference_policy=reference_policy, forbid_reversion=not safe,
                use_tag_approximation=True, use_batched_reveal=True,
            )
            unique: dict[str, GeneratedCandidate] = {item.sequence: item for item in generated}
            batch = list(unique.values())
            scored = scorer_member_predictions(
                scorer, [item.sequence for item in batch], scorer_batch_size)
            return batch, candidate_statistics(current_members, scored), scored

        candidates, stats, members = propose()
        sequences = [item.sequence for item in candidates]
        no_op_index = len(candidates)
        all_sequences = sequences + [current]
        for candidate_id, sequence in enumerate(all_sequences):
            if candidate_id == no_op_index:
                values = current_members
                row_stats = {
                    "mean": float(values.mean()), "std": float(values.std()),
                    "delta_mean": 0.0, "delta_std": 0.0, "agreement": 1.0,
                    "z_margin": 0.0, "conservative": float(values.mean() - 0.5 * values.std()),
                    "actual_mutations": 0, "is_noop": True,
                }
            else:
                row_stats = {name: float(stats[name][candidate_id]) for name in stats}
                row_stats.update({
                    "actual_mutations": int(candidates[candidate_id].actual_mutations),
                    "is_noop": False,
                })
            round_rows.append({
                "stage": f"mask{mask_count}", "round_id": round_id,
                "variant": variant, "sequence_before": current,
                "positions": ",".join(map(str, positions)),
                "candidate_id": candidate_id, "candidate_sequence": sequence,
                "dynamic_target": target, **row_stats,
            })

        if variant == "current":
            if selection_rule == "random":
                best_id = int(rng.integers(0, len(stats["std"])))
            elif selection_rule == "min_std":
                best_id = int(np.argmin(stats["std"]))
            else:
                best_id = int(np.argmax(stats["conservative"]))
            incumbent = current_members.mean() - 0.5 * current_members.std()
            if acceptance_rule == "always":
                accepted = True
            elif acceptance_rule == "rank":
                accepted = bool(stats["conservative"][best_id] > incumbent)
            elif acceptance_rule == "strict_margin":
                accepted = bool(stats["delta_mean"][best_id] >= strict_acceptance_margin)
            else:
                accepted = bool(stats["conservative"][best_id] >= incumbent + 0.10)
        else:
            eligible = (
                (stats["delta_mean"] >= minimum_delta)
                & (stats["agreement"] >= minimum_agreement)
                & (stats["z_margin"] >= minimum_z)
                & (stats["mean"] >= score_min)
                & (stats["mean"] <= score_max)
            )
            ids = np.flatnonzero(eligible)
            accepted = bool(len(ids))
            best_id = int(ids[np.argmax(stats["delta_mean"][ids])]) if accepted else -1

        if accepted and structure_guard is not None:
            def rank(block_stats):
                if selection_rule == "min_std":
                    return np.argsort(block_stats["std"])
                if selection_rule == "random":
                    return rng.permutation(len(block_stats["std"]))
                return np.argsort(-block_stats["conservative"])

            def survives(sequence):
                plddt = structure_guard["fold"](sequence)
                return plddt, (plddt >= structure_guard["floor"]
                               and plddt >= structure_guard["reference"]
                               - structure_guard["max_drop"])

            widths = []
            if structure_guard.get("shrink"):
                w = len(positions)
                while w >= 1:
                    widths.append(w)
                    w //= 2
            else:
                widths = [len(positions)]
            per_width = max(1, structure_guard["max_batches"] // len(widths))

            passed, folded, best_plddt, used_width = False, 0, None, len(positions)
            refined_sequence = None
            for width in widths:
                if passed:
                    break
                for batch_id in range(per_width):
                    if width != len(positions) or batch_id:
                        narrower = (positions if width == len(positions) else
                                    (mask_navigator(current, width, round_id)
                                     if mask_navigator is not None
                                     else tuple(positions[:width])))
                        candidates, stats, members = propose(narrower)
                        used_width = width
                    for candidate_id in rank(stats)[:structure_guard["max_tries"]]:
                        sequence = candidates[int(candidate_id)].sequence
                        if refiner is not None:
                            sequence = refiner(sequence)
                        plddt, ok = survives(sequence)
                        folded += 1
                        if best_plddt is None or plddt > best_plddt:
                            best_plddt = plddt
                        if ok:
                            best_id, passed = int(candidate_id), True
                            refined_sequence = sequence
                            break
                    if passed:
                        round_rows[-1].update({"guard_folded": folded,
                                               "guard_width": used_width,
                                               "guard_plddt": best_plddt})
                        break
            if not passed:
                round_rows[-1].update({"guard_exhausted": True,
                                       "guard_folded": folded,
                                       "guard_width": used_width,
                                       "guard_plddt": best_plddt})
                accepted = False

        vetoed = False
        if accepted and (risk_model is not None or random_veto_rate > 0.0):
            if risk_model is not None:
                reference = risk_reference if risk_reference is not None else initial
                proposed = candidates[best_id].sequence
                feature_row = {
                    "scorer_delta": float(stats["delta_mean"][best_id]),
                    "member_agreement": float(stats["agreement"][best_id]),
                    "z_margin": float(stats["z_margin"][best_id]),
                    "mask_size": float(mask_count),
                    "round_id": float(round_id),
                    "is_noop_pair": 1.0,
                    "n_changed": float(sum(a != b for a, b in zip(proposed, current))),
                    "dist_winner_from_anchor": float(sum(a != b for a, b in zip(proposed, reference))),
                    "dist_loser_from_anchor": float(sum(a != b for a, b in zip(current, reference))),
                }
                feature_row["drift_increase"] = (
                    feature_row["dist_winner_from_anchor"] - feature_row["dist_loser_from_anchor"]
                )
                vector = np.asarray([[feature_row[name] for name in risk_model["features"]]])
                risk = float(risk_model["model"].predict_proba(vector)[0, 1])
                vetoed = risk >= risk_model["threshold"]
            else:
                vetoed = bool(rng.random() < random_veto_rate)
            if vetoed:
                round_rows[-1]["vetoed"] = True
                accepted = False

        if accepted:
            winner = candidates[best_id]
            before = current
            delta = float(stats["delta_mean"][best_id])
            current = winner.sequence
            current_members = members[:, best_id]
            if refiner is not None and refined_sequence is not None:
                current = refined_sequence
                current_members = scorer_member_predictions(
                    scorer, [refined_sequence], scorer_batch_size)[:, 0]
            stale = 0
            triplets.append({
                "stage": f"mask{mask_count}", "round_id": round_id,
                "variant": variant, "anchor": before,
                "positions": ",".join(map(str, positions)),
                "winner": winner.sequence, "loser": before,
                "triplet_type": "winner_vs_noop", "scorer_delta": delta,
                "member_agreement": float(stats["agreement"][best_id]),
                "z_margin": float(stats["z_margin"][best_id]),
            })
            rejected = np.flatnonzero(stats["delta_mean"] <= 0.0)
            if len(rejected):
                hard_id = int(rejected[np.argmax(stats["conservative"][rejected])])
                triplets.append({
                    "stage": f"mask{mask_count}", "round_id": round_id,
                    "variant": variant, "anchor": before,
                    "positions": ",".join(map(str, positions)),
                    "winner": winner.sequence, "loser": candidates[hard_id].sequence,
                    "triplet_type": "winner_vs_hard_loser", "scorer_delta": float(
                        stats["mean"][best_id] - stats["mean"][hard_id]
                    ), "member_agreement": float(stats["agreement"][best_id]),
                    "z_margin": float(stats["z_margin"][best_id]),
                })
            if uncertainty_halt:
                accepted_agreement_history.append(float(stats["agreement"][best_id]))
                accepted_zmargin_history.append(float(stats["z_margin"][best_id]))
                if len(accepted_agreement_history) >= uncertainty_window:
                    recent_agreement = float(np.mean(accepted_agreement_history[-uncertainty_window:]))
                    recent_zmargin = float(np.mean(accepted_zmargin_history[-uncertainty_window:]))
                    if recent_agreement < uncertainty_min_agreement or recent_zmargin < uncertainty_min_zmargin:
                        round_rows[-1]["uncertainty_halted"] = True
                        break
            if mutation_budget is not None:
                reference = budget_reference if budget_reference is not None else initial
                distance = sum(a != b for a, b in zip(current, reference))
                if distance >= mutation_budget:
                    round_rows[-1]["mutation_budget_halted"] = True
                    break
        else:
            stale += 1
            if stale >= patience:
                break
    return current, round_rows, triplets
