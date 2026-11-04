"""The guide and scorer ensembles, and the guidance fields built on them.

`GuideEnsemble` wraps the guide ensemble: five `MaskedTailPredictor` members
that score partially masked sequences, supplying the mean and variance that
WHERE and WHAT steer on. `ScorerEnsemble` wraps the scorer ensemble: five
`CompleteSequenceScorer` members over complete sequences, supplying the
standard deviation WHICH minimises and the mean that sets the refinement
target.

This module covers the search-time predictors only. The evaluation oracle is
separate (`OfficialFitnessEvaluator` in oracle.py) and is read only
after a search has finished; it must never take part in candidate selection
or acceptance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .guidance import tag_log_ratio as _tag_log_ratio
from .noisy_predictor import ESM3_CANONICAL_IDS, ESM3_MASK_TOKEN
from .oracle import ALPHABET

AA_TO_ID = {aa: index for index, aa in enumerate(ALPHABET)}
MASK_ID = AA_TO_ID["X"]


def _build_esm3_to_alphabet_lookup(device: torch.device) -> torch.Tensor:
    max_id = max(max(ESM3_CANONICAL_IDS.values()), ESM3_MASK_TOKEN)
    lookup = torch.full((max_id + 1,), MASK_ID, dtype=torch.long)
    for aa, esm3_id in ESM3_CANONICAL_IDS.items():
        lookup[esm3_id] = AA_TO_ID[aa]
    return lookup.to(device)


def _build_esm3_to_alphabet_onehot_matrix(device: torch.device) -> torch.Tensor:
    """Differentiable version of the lookup: esm3_one_hot @ this matrix ==
    alphabet_one_hot. Needed so gradients can flow through the ESM3<->alphabet
    vocabulary remap for the first-order Taylor (TAG) guidance approximation
    -- a hard integer lookup has no gradient, a selection matrix does.
    """
    lookup = _build_esm3_to_alphabet_lookup(device)
    classes = len(ALPHABET)
    matrix = torch.zeros(lookup.shape[0], classes, device=device)
    matrix[torch.arange(lookup.shape[0], device=device), lookup] = 1.0
    return matrix


def encode_sequences(sequences: list[str]) -> torch.Tensor:
    return torch.tensor(
        [[AA_TO_ID.get(aa, MASK_ID) for aa in sequence] for sequence in sequences],
        dtype=torch.long,
    )


class MaskedTailPredictor(nn.Module):
    """Guide member: one-hot sequence plus four reveal-time features."""

    def __init__(self, length: int, hidden: tuple[int, int] = (800, 400)):
        super().__init__()
        self.length = length
        self.classes = len(ALPHABET)
        self.net = nn.Sequential(
            nn.Linear(length * self.classes + 4, hidden[0]), nn.GELU(), nn.Dropout(0.0),
            nn.Linear(hidden[0], hidden[1]), nn.GELU(), nn.Dropout(0.0),
            nn.Linear(hidden[1], 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim == 2:
            one_hot = F.one_hot(tokens.long(), num_classes=self.classes).float()
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


class CompleteSequenceScorer(nn.Module):
    """Scorer member: one-hot sequence over complete sequences only."""

    def __init__(self, length: int, hidden: tuple[int, int] = (800, 400)):
        super().__init__()
        self.length = length
        self.classes = len(ALPHABET)
        self.net = nn.Sequential(
            nn.Linear(length * self.classes, hidden[0]), nn.GELU(), nn.Dropout(0.0),
            nn.Linear(hidden[0], hidden[1]), nn.GELU(), nn.Dropout(0.0),
            nn.Linear(hidden[1], 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(tokens.long(), num_classes=self.classes).float()
        return self.net(one_hot.flatten(1)).squeeze(-1)


class GuideEnsemble:
    """The guide ensemble: five members scoring partially masked sequences."""

    GUIDE_FORMS = ("linear", "probit")

    def __init__(self, checkpoint_paths: list[Path], device: torch.device,
                 guide_form: str = "linear", residual_sigma: float = 2.31):
        if guide_form not in self.GUIDE_FORMS:
            raise ValueError(f"guide_form must be one of {self.GUIDE_FORMS}, got {guide_form!r}")
        self.device = device
        self.guide_form = guide_form
        self.residual_sigma = float(residual_sigma)
        if self.residual_sigma <= 0.0:
            raise ValueError("residual_sigma must be positive")
        self.lookup = _build_esm3_to_alphabet_lookup(device)
        self.onehot_matrix = _build_esm3_to_alphabet_onehot_matrix(device)
        self.models = []
        for path in checkpoint_paths:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            model = MaskedTailPredictor(
                length=checkpoint["config"]["length"], hidden=tuple(checkpoint["config"]["hidden"])
            )
            model.load_state_dict(checkpoint["state_dict"])
            model.to(device).eval()
            self.models.append(
                (model, checkpoint["normalization"]["mean"], checkpoint["normalization"]["std"])
            )

    @classmethod
    def load(cls, predictor_dir: str | Path, device: torch.device,
             guide_form: str = "linear", residual_sigma: float = 2.31) -> "GuideEnsemble":
        paths = sorted(Path(predictor_dir).glob("member_*.pt"))
        if not paths:
            raise FileNotFoundError(f"no member_*.pt checkpoints under {predictor_dir}")
        return cls(paths, device, guide_form=guide_form, residual_sigma=residual_sigma)

    def _combine(self, predictions: list[torch.Tensor], threshold: float) -> torch.Tensor:
        stacked = torch.stack(predictions)
        ensemble_mean = stacked.mean(0)
        if self.guide_form == "linear":
            return (ensemble_mean - threshold) / 0.5
        spread = stacked.std(0).detach() if len(predictions) > 1 else torch.zeros_like(ensemble_mean)
        sigma = torch.sqrt(spread.pow(2) + self.residual_sigma ** 2)
        return torch.special.log_ndtr((ensemble_mean - threshold) / sigma)

    @torch.inference_mode()
    def log_probability_above(self, esm3_tokens: torch.Tensor, threshold: float) -> torch.Tensor:
        """The guidance field. Under "linear" this is (mu - threshold) / 0.5, which
        is NOT a log probability and whose gradient ignores `threshold`; under
        "probit" it is log P(fitness > threshold). See GUIDE_FORMS above.
        """
        alphabet_tokens = self.lookup[esm3_tokens.to(self.device)]
        predictions = [model(alphabet_tokens).float() * std + mean
                       for model, mean, std in self.models]
        return self._combine(predictions, threshold)

    def _ensemble_log_probability_above_onehot(
        self, esm3_one_hot: torch.Tensor, threshold: float
    ) -> torch.Tensor:
        alphabet_one_hot = esm3_one_hot.to(self.device) @ self.onehot_matrix
        predictions = [model(alphabet_one_hot).float() * std + mean
                       for model, mean, std in self.models]
        return self._combine(predictions, threshold)

    def tag_log_ratio(
        self, esm3_tokens: torch.Tensor, t: torch.Tensor, threshold: float, vocab_size: int = 33,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """First-order Taylor estimate of every single-token property log
        ratio, covering all `vocab_size` substitutions at every position from
        one forward + one backward pass, instead of enumerating each
        substitution as a separate forward pass. Must not run under
        torch.inference_mode() (breaks the internal requires_grad_ call).
        """

        def predictor_log_probability(one_hot: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
            return self._ensemble_log_probability_above_onehot(one_hot, threshold)

        return _tag_log_ratio(predictor_log_probability, esm3_tokens, t, vocab_size)


class ScorerEnsemble:
    """The scorer ensemble: five members over complete sequences only.

    Supplies the standard deviation WHICH minimises and the mean that sets the
    refinement target.
    """

    def __init__(self, checkpoint_paths: list[Path], device: torch.device, uncertainty_penalty: float = 0.5):
        self.device = device
        self.uncertainty_penalty = float(uncertainty_penalty)
        self.models = []
        for path in checkpoint_paths:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            model = CompleteSequenceScorer(
                length=checkpoint["config"]["length"], hidden=tuple(checkpoint["config"]["hidden"])
            )
            model.load_state_dict(checkpoint["state_dict"])
            model.to(device).eval()
            self.models.append(
                (model, checkpoint["normalization"]["mean"], checkpoint["normalization"]["std"])
            )

    @classmethod
    def load(
        cls, scorer_dir: str | Path, device: torch.device, uncertainty_penalty: float = 0.5,
    ) -> "ScorerEnsemble":
        paths = sorted(Path(scorer_dir).glob("member_*.pt"))
        if not paths:
            raise FileNotFoundError(f"no member_*.pt checkpoints under {scorer_dir}")
        return cls(paths, device, uncertainty_penalty)

    @torch.inference_mode()
    def predict_distribution(
        self, sequences: list[str], batch_size: int = 128
    ) -> tuple[np.ndarray, np.ndarray]:
        tokens = encode_sequences(sequences)
        member_outputs = []
        for model, mean, std in self.models:
            values = []
            for start in range(0, len(tokens), batch_size):
                batch = tokens[start : start + batch_size].to(self.device)
                values.append(model(batch).float().cpu().numpy() * std + mean)
            member_outputs.append(np.concatenate(values))
        stacked = np.stack(member_outputs, axis=0)
        return stacked.mean(axis=0), stacked.std(axis=0)

    def predict(self, sequences: list[str], batch_size: int = 128) -> np.ndarray:
        mean, std = self.predict_distribution(sequences, batch_size)
        return mean - self.uncertainty_penalty * std
