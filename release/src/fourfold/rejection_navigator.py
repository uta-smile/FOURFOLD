"""Open the residues the fitness objective has stopped supporting.

The generator alone has an opinion about what belongs at a position. Guidance
toward higher fitness gives a second opinion. Where the two agree, the target
asks for nothing; where guidance pulls probability off the residue that is
actually there, the objective is saying that residue no longer fits.

    R_i = p(x_i | x with i masked) - p(x_i | x with i masked, fitness guided)

This asks whether the current choice should be abandoned, not how much a
future substitution might gain. The distinction matters: every attempt in this
project to rank positions by predicted gain has lost to drawing them at random,
because a surrogate's largest predicted gains sit exactly where it extrapolates
worst. A rejection signal does not reward the surrogate for optimism.

Each position is scored with that position masked and the rest intact, so the
distribution is the one the generator would face there. All positions are
evaluated in one batch, and nothing is fitted: the score is recomputed from the
current sequence every round, so the opened set moves as the search moves.
"""
from __future__ import annotations

import numpy as np
import torch

from .direct_exo import CANONICAL_ORDER, ESM3_MASK_TOKEN
from .esm3_lora import encode_esm3_sequences


class RejectionNavigator:
    def __init__(self, editor, guide, threshold: float = 4.0,
                 sampling_temperature: float = 0.7, guide_temperature: float = 0.1,
                 sign: float = 1.0, batch_size: int = 128,
                 screen_groups: int | None = None, candidate_pool: int = 15,
                 score_mode: str = "rejection"):
        from esm.utils.constants.esm3 import SEQUENCE_VOCAB
        self.editor = editor
        self.guide = guide
        self.threshold = threshold
        if score_mode not in ("rejection", "confidence", "guide"):
            raise ValueError(f"unknown score_mode {score_mode!r}")
        self.score_mode = score_mode
        self.sampling_temperature = sampling_temperature
        self.guide_temperature = guide_temperature
        self.sign = sign
        self.batch_size = batch_size
        self.screen_groups = screen_groups
        self.candidate_pool = candidate_pool
        self.canonical = torch.tensor(
            [SEQUENCE_VOCAB.index(a) for a in CANONICAL_ORDER], device=editor.device)
        self.order = {a: i for i, a in enumerate(CANONICAL_ORDER)}
        self.log: list[dict] = []

    def _score_states(self, sequence: str, states, members) -> dict[int, float]:
        """Rejection for the positions opened in each state, batched."""
        out = {}
        batch = torch.cat(states, 0)
        for start in range(0, len(batch), self.batch_size):
            chunk = batch[start:start + self.batch_size]
            with torch.inference_mode():
                logits = self.editor.sequence_logits_tokens(chunk, reference=True).float()
            if self.score_mode == "confidence":
                for row in range(len(chunk)):
                    for position in members[start + row]:
                        unguided = torch.log_softmax(
                            logits[row, position] / self.sampling_temperature, dim=-1)
                        out[position] = float(unguided.exp()[self.order[sequence[position]]])
                continue
            time = 1.0 - chunk.eq(ESM3_MASK_TOKEN).float().mean(dim=1)
            ratio, _ = self.guide.tag_log_ratio(chunk, time, self.threshold, vocab_size=33)
            ratio = ratio.detach()
            for row in range(len(chunk)):
                for position in members[start + row]:
                    unguided = torch.log_softmax(
                        logits[row, position] / self.sampling_temperature, dim=-1)
                    guided = torch.log_softmax(
                        unguided + ratio[row, position, :][self.canonical].float()
                        / self.guide_temperature, dim=-1)
                    here = self.order[sequence[position]]
                    if self.score_mode == "guide":
                        out[position] = float(1.0 - (guided[here] - unguided[here]).exp())
                    else:
                        out[position] = float(unguided.exp()[here] - guided.exp()[here])
        return out

    def screen(self, sequence: str, groups: int) -> np.ndarray:
        """A cheap ranking: several distant positions opened per forward pass."""
        length = len(sequence)
        base = encode_esm3_sequences([sequence]).to(self.editor.device)
        states, members = [], []
        for offset in range(groups):
            positions = list(range(offset, length, groups))
            state = base.clone()
            state[0, positions] = ESM3_MASK_TOKEN
            states.append(state); members.append(positions)
        scored = self._score_states(sequence, states, members)
        return np.array([scored[p] for p in range(length)])

    def exact_subset(self, sequence: str, positions) -> dict[int, float]:
        """Leave-one-out rejection, the same quantity the full scan computes."""
        base = encode_esm3_sequences([sequence]).to(self.editor.device)
        states, members = [], []
        for p in positions:
            state = base.clone()
            state[0, p] = ESM3_MASK_TOKEN
            states.append(state); members.append([int(p)])
        return self._score_states(sequence, states, members)


    def rejection(self, sequence: str) -> np.ndarray:
        """R for every position, each scored with that position masked."""
        length = len(sequence)
        base = encode_esm3_sequences([sequence]).to(self.editor.device)
        states = base.repeat(length, 1)
        states[torch.arange(length), torch.arange(length)] = ESM3_MASK_TOKEN
        out = np.zeros(length)
        for start in range(0, length, self.batch_size):
            chunk = states[start:start + self.batch_size]
            with torch.inference_mode():
                logits = self.editor.sequence_logits_tokens(chunk, reference=True).float()
            if self.score_mode == "confidence":
                for row in range(len(chunk)):
                    position = start + row
                    unguided = torch.log_softmax(
                        logits[row, position] / self.sampling_temperature, dim=-1)
                    out[position] = float(unguided.exp()[self.order[sequence[position]]])
                continue
            time = 1.0 - chunk.eq(ESM3_MASK_TOKEN).float().mean(dim=1)
            ratio, _ = self.guide.tag_log_ratio(chunk, time, self.threshold, vocab_size=33)
            ratio = ratio.detach()
            for row in range(len(chunk)):
                position = start + row
                unguided = torch.log_softmax(
                    logits[row, position] / self.sampling_temperature, dim=-1)
                guided = torch.log_softmax(
                    unguided + ratio[row, position, :][self.canonical].float()
                    / self.guide_temperature, dim=-1)
                here = self.order[sequence[position]]
                if self.score_mode == "guide":
                    out[position] = float(1.0 - (guided[here] - unguided[here]).exp())
                else:
                    out[position] = float(unguided.exp()[here] - guided.exp()[here])
        return out

    def __call__(self, sequence: str, mask_count: int, round_id: int) -> tuple[int, ...]:
        if self.screen_groups is None:
            value = self.rejection(sequence)
            states_used = len(sequence)
        else:
            rough = self.screen(sequence, self.screen_groups)
            pool = np.argsort(-self.sign * rough)[:max(self.candidate_pool, mask_count)]
            exact = self.exact_subset(sequence, pool)
            value = np.full(len(sequence), -np.inf * self.sign)
            for p, v in exact.items():
                value[p] = v
            states_used = self.screen_groups + len(pool)
        chosen = np.argsort(-self.sign * value)[:mask_count]
        self.log.append({
            "round_id": round_id,
            "positions": ",".join(str(int(c)) for c in sorted(chosen)),
            "rejection_sum": float(value[chosen].sum()),
            "rejection_max": float(value.max()),
            "rejection_sd": float(value[np.isfinite(value)].std()),
            "states_evaluated": int(states_used),
        })
        return tuple(int(c) for c in sorted(chosen))
