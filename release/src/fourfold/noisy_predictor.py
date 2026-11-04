from __future__ import annotations


import torch
from torch.nn import functional as F

ESM3_VOCAB_SIZE = 33
ESM3_MASK_TOKEN = 32
ESM3_CANONICAL_IDS = {
    amino_acid: token_id
    for token_id, amino_acid in enumerate(
        [
            "<cls>", "<pad>", "<eos>", "<unk>", "L", "A", "G", "V", "S",
            "E", "R", "T", "I", "D", "P", "K", "Q", "N", "F", "Y", "M",
            "H", "W", "C", "X", "B", "U", "Z", "O", ".", "-", "|", "<mask>",
        ]
    )
    if len(amino_acid) == 1 and amino_acid in "ACDEFGHIKLMNPQRSTVWY"
}


def encode_esm3_sequences(sequences: list[str]) -> torch.Tensor:
    if not sequences:
        return torch.empty((0, 0), dtype=torch.long)
    length = len(sequences[0])
    if any(len(sequence) != length for sequence in sequences):
        raise ValueError("Noisy predictor requires equal-length sequences")
    try:
        values = [[ESM3_CANONICAL_IDS[residue] for residue in sequence] for sequence in sequences]
    except KeyError as error:
        raise ValueError(f"Non-canonical amino acid: {error.args[0]}") from error
    return torch.tensor(values, dtype=torch.long)
