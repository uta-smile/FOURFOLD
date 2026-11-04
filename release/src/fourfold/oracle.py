from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np


ALPHABET = "ARNDCQEGHILKMFPSTWYVX"


class OfficialFitnessEvaluator:
    """Frozen released oracle, deliberately exposed only through final evaluation.

    Works for any of the three ProteinGuide fitness datasets; the released
    weights and the tokenisation alphabet are both selected by dataset_name
    inside the vendor code, so nothing here is protein-length specific.
    """

    SUPPORTED = ("ube4b", "pabp", "creilov")

    def __init__(self, vendor_root: str | Path, dataset_name: str = "ube4b"):
        dataset_name = dataset_name.lower()
        if dataset_name not in self.SUPPORTED:
            raise ValueError(
                f"dataset_name must be one of {self.SUPPORTED}, got {dataset_name!r}"
            )
        self.dataset_name = dataset_name
        vendor_root = Path(vendor_root).resolve()
        potts_stub = types.ModuleType("potts")
        potts_stub.Potts = type("Potts", (), {})
        sys.modules.setdefault("potts", potts_stub)
        sys.path.insert(0, str(vendor_root))
        from pdg.fitness import utils as vendor_utils
        self._predict_function = vendor_utils.get_predicted_fitness
        loader = {
            "ube4b": vendor_utils.load_oracle_model_ube4b,
            "pabp": vendor_utils.load_oracle_model_pabp,
            "creilov": vendor_utils.load_oracle_model_creilov,
        }[dataset_name]
        self._models = loader()

    def predict(self, sequences: list[str], batch_size: int = 128) -> np.ndarray:
        return self._predict_function(
            sequences,
            models=self._models,
            batch_size=batch_size,
            dataset_name=self.dataset_name,
        )
