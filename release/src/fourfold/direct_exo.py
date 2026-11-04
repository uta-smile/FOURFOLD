from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from .esm3_backend import CANONICAL_ORDER
from .esm3_lora import ESM3LoRAEditor
from .noisy_predictor import ESM3_CANONICAL_IDS, ESM3_MASK_TOKEN, encode_esm3_sequences


@dataclass(frozen=True)
class GeneratedCandidate:
    sequence: str
    reference_log_prob: float
    actual_mutations: int | None = None


def sample_local_group(
    editor: ESM3LoRAEditor,
    guide,
    anchor: str,
    positions: tuple[int, ...],
    rollouts: int,
    rng: np.random.Generator,
    guided: bool,
    guide_temperature: float,
    property_threshold: float,
    sampling_temperature: float,
    sampling_top_k: int,
    sampling_top_p: float,
    reference_policy: bool = False,
    forbid_reversion: bool = True,
    use_tag_approximation: bool = False,
    use_batched_reveal: bool = False,
) -> list[GeneratedCandidate]:
    """Generate variants while unresolved sites remain genuinely masked.

    forbid_reversion=True (default) is the original exact-k protocol: a
    masked site may not resolve back to the anchor residue, so every
    position in `positions` is guaranteed to change. This is appropriate for
    small masks (5/10) where "exactly k residues changed" is the intended
    semantics and is what mask-5/10's confirmed results were measured under.

    forbid_reversion=False allows a masked site to resolve back to its
    anchor residue -- required for large masks (50/75+), where forcing every
    masked position to change is overly aggressive and not what "explore
    this region" should mean. Callers needing the real number of changed
    positions should count differences between the returned sequence and
    the anchor themselves (exposed here as `actual_mutations`).

    use_tag_approximation=False (default) reproduces the original behavior
    exactly: every one of the 20 canonical substitutions at a masked position
    is enumerated and scored with a separate guide.log_probability_above
    forward pass. use_tag_approximation=True instead calls
    guide.tag_log_ratio once per position (one forward + one backward,
    first-order Taylor estimate covering all substitutions at once) when the
    guide provides it -- much cheaper for large candidate/anchor counts, at
    the cost of an approximation instead of an exact score. Only affects the
    `guided` branch; has no effect when guided=False.

    use_batched_reveal=False (default) reproduces the original behavior
    exactly: editor.conditional_logits_tokens is called twice (policy +
    reference) for every single position in `positions`, sequentially, so
    later positions' model logits are conditioned on earlier positions
    already being revealed this round. This is what ESM3 actually costs (each
    call is ~50-100x more expensive than a guide/predictor call, dwarfing
    whatever use_tag_approximation saves) -- it is the real bottleneck for
    large-scale triplet construction, not the guide scoring step.
    use_batched_reveal=True instead calls editor.sequence_logits_tokens once
    per round (covering every position at once, like the confirmed-working
    TAGFlowSampler pathway does for full-mask generation), then slices out
    each position's logits from that single call. This changes the sampling
    semantics: all positions in this round are decided from the state at the
    start of the round, not sequentially conditioned on each other's
    within-round reveals. Requires re-validating against the confirmed mask-5
    numbers before treating results from this mode as comparable.
    """
    if rollouts < 1:
        raise ValueError("At least one rollout is required")
    if not positions or len(set(positions)) != len(positions):
        raise ValueError("Positions must be non-empty and unique")
    initial = encode_esm3_sequences([anchor])[0]
    for position in positions:
        initial[position] = ESM3_MASK_TOKEN
    current = initial.repeat(rollouts, 1).to(editor.device)
    reference_log_probs = np.zeros(rollouts, dtype=np.float64)
    canonical_ids = torch.tensor(
        [ESM3_CANONICAL_IDS[aa] for aa in CANONICAL_ORDER],
        dtype=torch.long, device=editor.device,
    )
    aa_to_action = {aa: index for index, aa in enumerate(CANONICAL_ORDER)}

    use_tag = use_tag_approximation and guided and hasattr(guide, "tag_log_ratio")
    round_log_ratio = None
    if use_tag:
        shared_mask1_state_initial = len(positions) == 1 and rollouts > 1 and not use_batched_reveal
        round_state = current[:1] if shared_mask1_state_initial else current
        time = 1.0 - round_state.eq(ESM3_MASK_TOKEN).float().mean(dim=1)
        round_log_ratio, _ = guide.tag_log_ratio(round_state, time, property_threshold, vocab_size=33)
        round_log_ratio = round_log_ratio.detach()

    round_whole_logits = None
    round_whole_reference_logits = None
    if use_batched_reveal:
        with torch.inference_mode():
            round_whole_logits = editor.sequence_logits_tokens(current, reference=reference_policy)
            round_whole_reference_logits = editor.sequence_logits_tokens(current, reference=True)

    for position in positions:
        shared_mask1_state = len(positions) == 1 and rollouts > 1 and not use_batched_reveal
        model_current = current[:1] if shared_mask1_state else current
        if use_batched_reveal:
            logits = round_whole_logits[:, position, :]
            reference_logits = round_whole_reference_logits[:, position, :]
        else:
            model_sites = [position] * len(model_current)
            with torch.inference_mode():
                logits = editor.conditional_logits_tokens(
                    model_current, model_sites, require_grad=False, reference=reference_policy
                )
                reference_logits = editor.conditional_logits_tokens(
                    model_current, model_sites, require_grad=False, reference=True
                )
                if shared_mask1_state:
                    logits = logits.repeat(rollouts, 1)
                    reference_logits = reference_logits.repeat(rollouts, 1)
        sampling_logits = logits / sampling_temperature
        if guided and use_tag:
            log_property = round_log_ratio[:, position, :][:, canonical_ids]
            if shared_mask1_state:
                log_property = log_property.repeat(rollouts, 1)
            sampling_logits = sampling_logits + log_property / guide_temperature
        elif guided:
            with torch.inference_mode():
                proposed = model_current[:, None, :].repeat(1, len(CANONICAL_ORDER), 1)
                proposed[:, :, position] = canonical_ids[None, :]
                log_property = guide.log_probability_above(
                    proposed.flatten(0, 1), property_threshold
                ).view(len(model_current), len(CANONICAL_ORDER))
                if shared_mask1_state:
                    log_property = log_property.repeat(rollouts, 1)
            sampling_logits = sampling_logits + log_property / guide_temperature
        if forbid_reversion:
            original_action = aa_to_action[anchor[position]]
            sampling_logits[:, original_action] = -torch.inf
        keep = min(sampling_top_k, len(CANONICAL_ORDER) - 1)
        top_values, top_indices = torch.topk(sampling_logits, keep, dim=-1)
        top_probs = torch.softmax(top_values, dim=-1).cpu().numpy()
        cumulative_before = np.cumsum(top_probs, axis=1) - top_probs
        top_probs[cumulative_before >= sampling_top_p] = 0.0
        top_probs /= top_probs.sum(axis=1, keepdims=True)
        reference_lp = F.log_softmax(reference_logits, dim=-1).cpu().numpy()
        actions = []
        if len(positions) == 1 and rollouts > 1:
            if rollouts > keep:
                raise ValueError("Unique mask-1 candidates require rollouts <= sampling_top_k")
            stable_logits = top_values[0].double().cpu().numpy()
            uniforms = rng.uniform(np.finfo(float).tiny, 1.0, size=keep)
            gumbels = -np.log(-np.log(uniforms))
            picked_ids = np.argsort(-(stable_logits + gumbels))[:rollouts]
            actions = [int(top_indices[0, picked]) for picked in picked_ids]
        else:
            for row in range(rollouts):
                picked = int(rng.choice(keep, p=top_probs[row]))
                actions.append(int(top_indices[row, picked]))
        for row, action in enumerate(actions):
            reference_log_probs[row] += float(reference_lp[row, action])
        action_tensor = torch.tensor(actions, dtype=torch.long, device=editor.device)
        current[:, position] = canonical_ids[action_tensor]

    inverse = {token_id: aa for aa, token_id in ESM3_CANONICAL_IDS.items()}
    output = []
    for row in range(rollouts):
        sequence = "".join(inverse[int(token)] for token in current[row].cpu().tolist())
        actual_mutations = sum(1 for a, b in zip(sequence, anchor) if a != b)
        output.append(
            GeneratedCandidate(sequence, float(reference_log_probs[row]), actual_mutations)
        )
    return output
