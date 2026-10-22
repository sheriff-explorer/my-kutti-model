"""
training/dataset.py

Dataset utilities for all three training stages.

Pre-training / Mid-training: PackedDataset
──────────────────────────────────────────
For CLM (causal language modeling), we pack all tokenized text into a flat
array and serve random block_size-length chunks as (x, y) pairs where:
  x = tokens[i : i+block_size]
  y = tokens[i+1 : i+block_size+1]   ← x shifted by 1

This is the most efficient approach -- no wasted padding tokens.

Post-training (SFT): SFTDataset
───────────────────────────────
For supervised fine-tuning, we apply loss masking: question tokens are
set to -100 in the target tensor so cross_entropy ignores them.
The model only learns to predict the answer tokens.
"""

import os
import random
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# Packed Dataset (Pre-training / Mid-training)
# ─────────────────────────────────────────────────────────────────────────────

class PackedDataset(Dataset):
    """
    Flat packed dataset for causal language modeling.

    Takes a flat list of token IDs (all documents concatenated with EOS tokens
    between them) and serves (x, y) chunks of length block_size.

    x = tokens[i : i+block_size]        (input to model)
    y = tokens[i+1 : i+block_size+1]    (next-token targets)

    Why "packed"?
    ─────────────
    Alternative: pad each document to block_size separately.
    Problem: short documents waste most of the sequence with padding tokens.
    Packed: multiple short documents are concatenated into a single chunk.
    Result: ~100% token efficiency, every token contributes to the loss.

    EOS tokens between documents signal boundaries -- the model learns
    not to carry information across document boundaries.
    """

    def __init__(self, token_ids: List[int], block_size: int):
        """
        Args:
            token_ids:  Flat list of integer token IDs (all text concatenated)
            block_size: Length of each training chunk (= model's context window)
        """
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size

        # Number of valid starting positions
        # We need block_size+1 tokens for each (x, y) pair
        self.n_chunks = len(self.data) - block_size - 1

        if self.n_chunks <= 0:
            raise ValueError(
                f"Dataset has {len(self.data)} tokens, but block_size={block_size}. "
                "Need at least block_size+1 tokens."
            )

        print(f"PackedDataset: {len(self.data):,} tokens -> {self.n_chunks:,} training chunks")

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (x, y) pair where y = x shifted right by 1.

        For position i, the model sees x[i] and must predict x[i+1].
        This is the causal language modeling objective.
        """
        x = self.data[idx : idx + self.block_size]           # [block_size]
        y = self.data[idx + 1 : idx + 1 + self.block_size]   # [block_size]
        return x, y


def create_pretrain_dataloader(
    token_ids: List[int],
    block_size: int,
    batch_size: int,
    val_split: float = 0.05,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders for pre-training.

    Args:
        token_ids:   All tokenized text as a flat list
        block_size:  Model context window
        batch_size:  Sequences per batch
        val_split:   Fraction of data for validation (default: 5%)
        num_workers: Parallel data loading workers

    Returns:
        train_loader, val_loader
    """
    # Split into train / validation
    split_idx = int(len(token_ids) * (1 - val_split))
    train_ids = token_ids[:split_idx]
    val_ids   = token_ids[split_idx:]

    train_dataset = PackedDataset(train_ids, block_size)
    val_dataset   = PackedDataset(val_ids, block_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,       # Shuffle chunk starting positions
        num_workers=num_workers,
        pin_memory=True,    # Faster CPU->GPU transfer
        drop_last=True,     # Avoid partial batches at end
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(f"Train: {len(train_dataset):,} chunks | Val: {len(val_dataset):,} chunks")
    print(f"Train batches: {len(train_loader):,} | Val batches: {len(val_loader):,}")
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# SFT Dataset (Post-training)
# ─────────────────────────────────────────────────────────────────────────────

class SFTDataset(Dataset):
    """
    Supervised Fine-Tuning dataset with loss masking.

    For Q&A pairs formatted as:
        "Q: <question>\\nA: <answer>"

    We tokenize the full string, then mask the question tokens in the target
    with -100 so they don't contribute to the training loss.

    Why loss masking?
    ─────────────────
    Without masking, the model would try to predict the question tokens too,
    which:
      1. Wastes capacity -- the question format is fixed/templated
      2. Bakes in the assumption that questions are always asked, making the
         model "expect" to start every generation with a question.
      3. Adds noise: the model gets penalized for failing to predict the exact
         question wording, even if its answer would be correct.

    With masking (target=-100 for question tokens):
      - Only answer tokens contribute to the gradient
      - Model learns: "given this question, produce this answer"
      - Cleaner learning signal
    """

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        tokenizer,
        block_size: int,
        q_prefix: str = "Q: ",
        a_prefix: str = "\nA: ",
    ):
        """
        Args:
            pairs:      List of (question, answer) string tuples
            tokenizer:  BPETokenizer instance (already trained/loaded)
            block_size: Maximum sequence length (longer pairs are truncated)
            q_prefix:   Prefix for questions (default "Q: ")
            a_prefix:   Separator + answer prefix (default "\\nA: ")
        """
        self.samples = []

        for question, answer in pairs:
            # Format: "Q: <question>\nA: <answer>"
            q_text = q_prefix + question
            a_text = a_prefix + answer

            q_ids = tokenizer.encode(q_text)
            a_ids = tokenizer.encode(a_text)

            # Append EOS to signal end of answer
            a_ids = a_ids + [tokenizer.eos_id]

            full_ids = q_ids + a_ids

            # Truncate if necessary
            if len(full_ids) > block_size:
                full_ids = full_ids[:block_size]

            # Build input (x) and target (y)
            x = torch.tensor(full_ids[:-1], dtype=torch.long)  # all but last
            y = torch.tensor(full_ids[1:],  dtype=torch.long)  # all but first

            # Apply loss mask: set question token targets to -100
            # The answer starts at position len(q_ids) - 1 in the target
            # (offset by 1 because y is x shifted right)
            q_len = len(q_ids)
            mask_len = min(q_len - 1, len(y))  # how many y positions to mask
            y[:mask_len] = -100

            # Pad to uniform length for batching
            pad_len = block_size - 1 - len(x)
            if pad_len > 0:
                x = torch.cat([x, torch.zeros(pad_len, dtype=torch.long)])
                y = torch.cat([y, torch.full((pad_len,), -100, dtype=torch.long)])

            self.samples.append((x, y))

        print(f"SFTDataset: {len(self.samples)} Q&A pairs loaded")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def create_sft_dataloader(
    pairs: List[Tuple[str, str]],
    tokenizer,
    block_size: int,
    batch_size: int,
    val_fraction: float = 0.1,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train/val DataLoaders for SFT.

    Args:
        pairs:        List of (question, answer) tuples
        tokenizer:    Trained BPETokenizer
        block_size:   Model context window
        batch_size:   Batch size (smaller than pre-train, typically 16-32)
        val_fraction: Fraction of pairs for validation

    Returns:
        train_loader, val_loader
    """
    random.shuffle(pairs)
    split_idx = int(len(pairs) * (1 - val_fraction))
    train_pairs = pairs[:split_idx]
    val_pairs   = pairs[split_idx:]

    train_dataset = SFTDataset(train_pairs, tokenizer, block_size)
    val_dataset   = SFTDataset(val_pairs,   tokenizer, block_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader
