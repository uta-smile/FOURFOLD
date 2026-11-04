"""ESMFold loader for the WHETHER structural check."""
from __future__ import annotations


def load_model():
    """Return (tokenizer, model) for facebook/esmfold_v1 on the current CUDA device.

    The language tower runs in half precision; the folding trunk must stay in
    fp32. Casting the trunk makes its distogram come back all-NaN, after which
    no pLDDT is produced at all.
    """
    from transformers import AutoTokenizer, EsmForProteinFolding

    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1",
                                                 low_cpu_mem_usage=True)
    model = model.cuda().eval()
    model.esm = model.esm.half()
    model.trunk.set_chunk_size(64)
    return tokenizer, model
