from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from contextlib import nullcontext

import pandas as pd
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .esm3_backend import CANONICAL_ORDER
from .noisy_predictor import ESM3_CANONICAL_IDS, ESM3_MASK_TOKEN, encode_esm3_sequences


ESM3_BOS_TOKEN = 0
ESM3_EOS_TOKEN = 2
LORA_TARGETS = (
    "attn.layernorm_qkv.1",
    "attn.out_proj",
    "output_heads.sequence_head.0",
    "output_heads.sequence_head.3",
)


def sequence_tokens(sequences: list[str], device: torch.device) -> torch.Tensor:
    residues = encode_esm3_sequences(sequences).to(device)
    bos = torch.full((len(sequences), 1), ESM3_BOS_TOKEN, dtype=torch.long, device=device)
    eos = torch.full((len(sequences), 1), ESM3_EOS_TOKEN, dtype=torch.long, device=device)
    return torch.cat((bos, residues, eos), dim=1)


class PairDataset(Dataset):
    def __init__(self, csv_path: str | Path) -> None:
        frame = pd.read_csv(csv_path, usecols=["source_sequence", "target_sequence"])
        self.examples: list[tuple[str, str, tuple[int, ...]]] = []
        for source, target in frame.itertuples(index=False, name=None):
            source, target = str(source), str(target)
            if len(source) != len(target):
                continue
            positions = tuple(index for index, pair in enumerate(zip(source, target)) if pair[0] != pair[1])
            if positions:
                self.examples.append((source, target, positions))
        if not self.examples:
            raise ValueError("Pair-SFT dataset contains no editable sequence pairs")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        return self.examples[index]


def _collate_pairs(batch):
    return batch


@dataclass(frozen=True)
class PairSFTResult:
    steps: int
    mean_loss: float
    trainable_parameters: int


class ESM3LoRAEditor:
    """ESM3 local editor using the LoRA configuration from ProteinGuide."""

    def __init__(self, model, device: torch.device) -> None:
        self.model = model
        self.device = device
        self.canonical_token_ids = torch.tensor(
            [ESM3_CANONICAL_IDS[aa] for aa in CANONICAL_ORDER],
            dtype=torch.long,
            device=device,
        )

    @classmethod
    def from_pretrained(
        cls,
        device: torch.device,
        model_name: str = "esm3_sm_open_v1",
        rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.05,
    ) -> "ESM3LoRAEditor":
        from esm.models.esm3 import ESM3

        base = ESM3.from_pretrained(model_name, device=device)
        config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            target_modules=list(LORA_TARGETS),
        )
        return cls(get_peft_model(base, config).to(device), device)

    @classmethod
    def load_adapter(
        cls,
        adapter_dir: str | Path,
        device: torch.device,
        model_name: str = "esm3_sm_open_v1",
        trainable: bool = True,
    ) -> "ESM3LoRAEditor":
        from esm.models.esm3 import ESM3

        base = ESM3.from_pretrained(model_name, device=device)
        model = PeftModel.from_pretrained(
            base, str(adapter_dir), is_trainable=trainable
        ).to(device)
        return cls(model, device)

    @classmethod
    def from_pair_sft(
        cls,
        adapter_dir: str | Path,
        device: torch.device,
        model_name: str = "esm3_sm_open_v1",
        rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.05,
    ) -> "ESM3LoRAEditor":
        """Merge Pair-SFT, then add a fresh LoRA whose disabled state is the reference."""
        from esm.models.esm3 import ESM3

        base = ESM3.from_pretrained(model_name, device=device)
        pair_model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)
        pair_base = pair_model.merge_and_unload()
        online_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            target_modules=list(LORA_TARGETS),
        )
        return cls(get_peft_model(pair_base, online_config).to(device), device)

    def save_adapter(self, output_dir: str | Path) -> None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(output_dir))

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [parameter for parameter in self.model.parameters() if parameter.requires_grad]

    def _masked_tokens(self, contexts: list[str], positions: list[int]) -> torch.Tensor:
        if len(contexts) != len(positions):
            raise ValueError("Contexts and positions must have equal lengths")
        tokens = sequence_tokens(contexts, self.device)
        rows = torch.arange(len(contexts), device=self.device)
        columns = torch.tensor(positions, dtype=torch.long, device=self.device) + 1
        tokens[rows, columns] = ESM3_MASK_TOKEN
        return tokens

    def conditional_logits(
        self,
        contexts: list[str],
        positions: list[int],
        require_grad: bool = False,
        reference: bool = False,
    ) -> torch.Tensor:
        tokens = self._masked_tokens(contexts, positions)
        rows = torch.arange(len(contexts), device=self.device)
        columns = torch.tensor(positions, dtype=torch.long, device=self.device) + 1
        context = torch.enable_grad() if require_grad else torch.inference_mode()
        adapter_context = self.model.disable_adapter() if reference else nullcontext()
        with context, adapter_context, torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            output = self.model(sequence_tokens=tokens)
            logits = output.sequence_logits[rows, columns]
            logits = logits[:, self.canonical_token_ids]
        return logits.float()

    def conditional_logits_tokens(
        self,
        residue_tokens: torch.Tensor,
        positions: list[int],
        require_grad: bool = False,
        reference: bool = False,
    ) -> torch.Tensor:
        """Return logits at masked sites while preserving all other masks."""
        if residue_tokens.ndim != 2 or residue_tokens.shape[0] != len(positions):
            raise ValueError("residue_tokens must be [batch, length] and match positions")
        tokens = residue_tokens.to(self.device)
        rows = torch.arange(len(positions), device=self.device)
        sites = torch.tensor(positions, dtype=torch.long, device=self.device)
        if not torch.all(tokens[rows, sites] == ESM3_MASK_TOKEN):
            raise ValueError("Every queried position must still contain the mask token")
        bos = torch.full((len(tokens), 1), ESM3_BOS_TOKEN, dtype=torch.long, device=self.device)
        eos = torch.full((len(tokens), 1), ESM3_EOS_TOKEN, dtype=torch.long, device=self.device)
        model_tokens = torch.cat((bos, tokens, eos), dim=1)
        context = torch.enable_grad() if require_grad else torch.inference_mode()
        adapter_context = self.model.disable_adapter() if reference else nullcontext()
        with context, adapter_context, torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            output = self.model(sequence_tokens=model_tokens)
            logits = output.sequence_logits[rows, sites + 1]
            logits = logits[:, self.canonical_token_ids]
        return logits.float()

    def trajectory_log_prob(
        self,
        contexts: list[str],
        positions: list[int],
        targets: list[int],
        require_grad: bool,
        reference: bool = False,
    ) -> torch.Tensor:
        logits = self.conditional_logits(
            contexts, positions, require_grad=require_grad, reference=reference
        )
        target = torch.tensor(targets, dtype=torch.long, device=self.device)
        return F.log_softmax(logits, dim=-1).gather(-1, target[:, None]).squeeze(-1).sum()

    def sequence_logits_tokens(
        self,
        residue_tokens: torch.Tensor,
        require_grad: bool = False,
        reference: bool = False,
    ) -> torch.Tensor:
        """Return canonical-AA logits at every residue position."""
        if residue_tokens.ndim != 2:
            raise ValueError("residue_tokens must have shape [batch, length]")
        tokens = residue_tokens.to(self.device)
        bos = torch.full(
            (len(tokens), 1), ESM3_BOS_TOKEN, dtype=torch.long, device=self.device
        )
        eos = torch.full(
            (len(tokens), 1), ESM3_EOS_TOKEN, dtype=torch.long, device=self.device
        )
        model_tokens = torch.cat((bos, tokens, eos), dim=1)
        context = torch.enable_grad() if require_grad else torch.inference_mode()
        adapter_context = self.model.disable_adapter() if reference else nullcontext()
        with context, adapter_context, torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            output = self.model(sequence_tokens=model_tokens)
            logits = output.sequence_logits[:, 1:-1]
            logits = logits[:, :, self.canonical_token_ids]
        return logits.float()

    def fit_pairs(
        self,
        pair_csv: str | Path,
        max_steps: int,
        batch_size: int = 2,
        gradient_accumulation: int = 16,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        warmup_ratio: float = 0.1,
        seed: int = 42,
    ) -> PairSFTResult:
        dataset = PairDataset(pair_csv)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=_collate_pairs,
            generator=torch.Generator().manual_seed(seed),
        )
        parameters = self.trainable_parameters()
        optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
        warmup_steps = max(1, int(max_steps * warmup_ratio))

        def learning_rate_scale(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_scale)
        self.model.train()
        optimizer.zero_grad(set_to_none=True)
        iterator = iter(loader)
        losses: list[float] = []
        for step in range(max_steps):
            accumulated = 0.0
            for _ in range(gradient_accumulation):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    batch = next(iterator)
                contexts: list[str] = []
                positions: list[int] = []
                targets: list[int] = []
                for source, target, changed in batch:
                    for position in changed:
                        contexts.append(source)
                        positions.append(position)
                        targets.append(CANONICAL_ORDER.index(target[position]))
                logits = self.conditional_logits(contexts, positions, require_grad=True)
                target_tensor = torch.tensor(targets, dtype=torch.long, device=self.device)
                loss = F.cross_entropy(logits, target_tensor) / gradient_accumulation
                loss.backward()
                accumulated += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(accumulated)
        self.model.eval()
        return PairSFTResult(
            steps=max_steps,
            mean_loss=float(sum(losses) / len(losses)),
            trainable_parameters=sum(parameter.numel() for parameter in parameters),
        )
