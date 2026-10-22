"""
model/tokenizer.py

BPE Tokenizer with vocab_size=512 for TinyGPT.

Uses HuggingFace `tokenizers` library internally for efficient BPE training.
Wraps it in a clean, minimal interface with save/load support.

Why BPE (Byte-Pair Encoding)?
─────────────────────────────
BPE is a subword tokenization algorithm. It starts from individual bytes
(256 tokens covering every possible byte value), then iteratively merges
the most frequent adjacent pair into a new token.

Example training process:
  Start: ["h", "e", "l", "l", "o", " ", "w", "o", "r", "l", "d"]
  Most common pair: ("l", "l") -> merge -> "ll"
  Next: ("o", " ") -> merge -> "o "
  ...continue until vocab_size reached.

With vocab_size=512: 256 byte tokens + 256 learned merges.
Common words like "the", "is", "in" get their own single token.
Rare words stay as byte sequences. No OOV (out-of-vocabulary) ever.
"""

import os
import json
from typing import List, Optional

# HuggingFace tokenizers for efficient BPE
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


class BPETokenizer:
    """
    Minimal BPE tokenizer wrapper.

    Usage:
        # Train from scratch:
        tok = BPETokenizer()
        tok.train(texts, vocab_size=512)
        tok.save("checkpoints/tokenizer.json")

        # Load existing:
        tok = BPETokenizer.load("checkpoints/tokenizer.json")

        # Encode / decode:
        ids = tok.encode("Hello world")      # [45, 12, 203]
        text = tok.decode([45, 12, 203])     # "Hello world"
    """

    # Special tokens used across all training stages
    PAD_TOKEN = "<pad>"    # Padding (not usually needed for CLM, but useful for SFT)
    EOS_TOKEN = "<eos>"    # End of sequence -- signals model to stop generating
    BOS_TOKEN = "<bos>"    # Beginning of sequence -- prefix for inference prompts
    UNK_TOKEN = "<unk>"    # Unknown (fallback; byte-level BPE should never hit this)

    SPECIAL_TOKENS = [PAD_TOKEN, EOS_TOKEN, BOS_TOKEN, UNK_TOKEN]

    def __init__(self):
        self._tokenizer: Optional[Tokenizer] = None

    # ─────────────────────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────────────────────

    def train(self, texts: List[str], vocab_size: int = 512) -> None:
        """
        Train a BPE tokenizer on a list of text strings.

        Args:
            texts:      List of strings (e.g. TinyStories paragraphs)
            vocab_size: Total vocabulary size including special tokens.
                        512 = 256 bytes + 256 merges + 4 special tokens.
        """
        print(f"Training BPE tokenizer (vocab_size={vocab_size}) on "
              f"{len(texts):,} texts...")

        # ByteLevel BPE: every byte is a valid base token -> no UNK in practice
        self._tokenizer = Tokenizer(BPE(unk_token=self.UNK_TOKEN))
        self._tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        self._tokenizer.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=self.SPECIAL_TOKENS,
            show_progress=True,
        )

        # Train from an in-memory iterator (avoids temp file I/O)
        self._tokenizer.train_from_iterator(
            iter(texts),
            trainer=trainer,
            length=len(texts),
        )

        actual_vocab = self._tokenizer.get_vocab_size()
        print(f"  Tokenizer trained. Actual vocab size: {actual_vocab}")

    # ─────────────────────────────────────────────────────────────
    # Encode / Decode
    # ─────────────────────────────────────────────────────────────

    def encode(self, text: str) -> List[int]:
        """
        Convert a string to a list of integer token IDs.

        Example:
            encode("Hello!")  ->  [45, 203, 7]
        """
        self._check_trained()
        return self._tokenizer.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        """
        Convert a list of integer token IDs back to a string.

        Example:
            decode([45, 203, 7])  ->  "Hello!"
        """
        self._check_trained()
        return self._tokenizer.decode(ids)

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Encode a batch of strings efficiently."""
        self._check_trained()
        return [enc.ids for enc in self._tokenizer.encode_batch(texts)]

    # ─────────────────────────────────────────────────────────────
    # Special token helpers
    # ─────────────────────────────────────────────────────────────

    @property
    def eos_id(self) -> int:
        """Token ID of <eos> -- used to signal end of generation."""
        self._check_trained()
        return self._tokenizer.token_to_id(self.EOS_TOKEN)

    @property
    def bos_id(self) -> int:
        """Token ID of <bos> -- prepended to prompts at inference."""
        self._check_trained()
        return self._tokenizer.token_to_id(self.BOS_TOKEN)

    @property
    def pad_id(self) -> int:
        """Token ID of <pad>."""
        self._check_trained()
        return self._tokenizer.token_to_id(self.PAD_TOKEN)

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size."""
        self._check_trained()
        return self._tokenizer.get_vocab_size()

    # ─────────────────────────────────────────────────────────────
    # Save / Load
    # ─────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save trained tokenizer to a JSON file."""
        self._check_trained()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._tokenizer.save(path)
        print(f"Tokenizer saved to {path}")

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Load a previously trained tokenizer from a JSON file."""
        tok = cls()
        tok._tokenizer = Tokenizer.from_file(path)
        print(f"Tokenizer loaded from {path} "
              f"(vocab_size={tok._tokenizer.get_vocab_size()})")
        return tok

    # ─────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────

    def _check_trained(self):
        if self._tokenizer is None:
            raise RuntimeError(
                "Tokenizer has not been trained or loaded. "
                "Call .train() or .load() first."
            )

    def __repr__(self) -> str:
        if self._tokenizer is None:
            return "BPETokenizer(not trained)"
        return f"BPETokenizer(vocab_size={self._tokenizer.get_vocab_size()})"
