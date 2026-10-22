"""
stages/01_pretrain.py

Stage 1: Pre-training TinyGPT from scratch on TinyStories.

What happens here:
  1. Download a slice of TinyStories from HuggingFace
  2. Train a BPE tokenizer on the corpus (vocab_size=512)
  3. Tokenize the entire corpus into a flat array of token IDs
  4. Create packed train/val DataLoaders
  5. Initialize TinyGPT from random weights
  6. Train for total_steps with cosine LR schedule
  7. Save tokenizer + model checkpoint

Run with:
    python stages/01_pretrain.py

Expected output:
  - Loss starts ~6.2 (= log(512), random guessing)
  - Loss should drop to ~1.5-2.0 after 5000 steps
  - Generated text should look like real English words with correct spacing
"""

import os
import sys
import numpy as np
import torch

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import ModelConfig
from model.tokenizer import BPETokenizer
from model.transformer import TinyGPT
from training.dataset import create_pretrain_dataloader
from training.trainer import Trainer


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Model architecture -- ~2M parameters
MODEL_CFG = ModelConfig(
    vocab_size  = 512,
    d_model     = 160,
    n_heads     = 4,
    n_layers    = 6,
    block_size  = 256,
    ffn_dim     = 640,
    dropout     = 0.1,
    tie_weights = True,
)

# Training hyperparameters
BATCH_SIZE      = 64      # Sequences per gradient step
TOTAL_STEPS     = 5000    # Training steps (not epochs -- more predictable)
WARMUP_STEPS    = 200     # Steps to ramp LR from 0 -> max_lr
MAX_LR          = 3e-3    # Peak learning rate (AdamW)
MIN_LR          = 3e-4    # Floor LR (= max_lr / 10)
WEIGHT_DECAY    = 0.1     # L2 regularization
GRAD_CLIP       = 1.0     # Gradient norm clipping

# Data
TINYSTORIES_SPLIT = "train[:10%]"  # 10% of TinyStories (~210K stories, ~50MB)
CHECKPOINT_DIR    = "checkpoints/pretrain"
TOKENIZER_PATH    = "checkpoints/tokenizer.json"


def download_and_prepare_data(tokenizer: BPETokenizer):
    """Download TinyStories and tokenize the full corpus."""
    print("\n[1/4] Downloading TinyStories dataset (10% slice)...")
    print("      This may take a minute on first run (downloads ~50MB).\n")

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not found. Run: pip install datasets")
        sys.exit(1)

    dataset = load_dataset(
        "roneneldan/TinyStories",
        split=TINYSTORIES_SPLIT,
    )

    texts = dataset["text"]
    print(f"  Downloaded {len(texts):,} stories.")

    # Train tokenizer if not already saved
    if not os.path.exists(TOKENIZER_PATH):
        print(f"\n[2/4] Training BPE tokenizer (vocab_size={MODEL_CFG.vocab_size})...")
        tokenizer.train(texts, vocab_size=MODEL_CFG.vocab_size)
        os.makedirs(os.path.dirname(TOKENIZER_PATH), exist_ok=True)
        tokenizer.save(TOKENIZER_PATH)
    else:
        print(f"\n[2/4] Loading existing tokenizer from {TOKENIZER_PATH}")
        tokenizer = BPETokenizer.load(TOKENIZER_PATH)

    # Tokenize corpus (or load from disk cache)
    cached_tokens_path = "checkpoints/pretrain_tokens.npy"
    if os.path.exists(cached_tokens_path):
        print(f"\n[3/4] Loading cached tokens from {cached_tokens_path}...")
        all_ids = np.load(cached_tokens_path)
        print(f"  Loaded {len(all_ids):,} tokens from cache.")
        return all_ids, tokenizer

    print(f"\n[3/4] Tokenizing {len(texts):,} stories with multi-threaded Rust batch encoder...")
    eos_id = tokenizer.eos_id
    
    # Fast multi-threaded batch encoding
    batch_size_tok = 25000
    all_token_lists = []
    
    for i in range(0, len(texts), batch_size_tok):
        chunk = texts[i : i + batch_size_tok]
        encoded_chunk = tokenizer.encode_batch(chunk)
        for enc in encoded_chunk:
            all_token_lists.extend(enc)
            all_token_lists.append(eos_id)
        print(f"  Tokenized {min(i + batch_size_tok, len(texts)):,}/{len(texts):,} stories...")

    all_ids = np.array(all_token_lists, dtype=np.uint16)
    print(f"  Saving tokenized cache to {cached_tokens_path}...")
    np.save(cached_tokens_path, all_ids)
    print(f"  Total tokens: {len(all_ids):,}")
    print(f"  Avg tokens per story: {len(all_ids)/len(texts):.0f}")
    return all_ids, tokenizer


def main():
    print("=" * 60)
    print(" TinyGPT -- Stage 1: Pre-training")
    print("=" * 60)
    print(f"\nModel config:\n{MODEL_CFG}")

    # Verify parameter count
    model = TinyGPT(MODEL_CFG)
    model.count_params()

    # Data preparation
    tokenizer = BPETokenizer()
    if os.path.exists(TOKENIZER_PATH):
        tokenizer = BPETokenizer.load(TOKENIZER_PATH)

    token_ids, tokenizer = download_and_prepare_data(tokenizer)

    # Create DataLoaders
    print(f"\n[4/4] Creating DataLoaders (batch_size={BATCH_SIZE}, block_size={MODEL_CFG.block_size})...")
    train_loader, val_loader = create_pretrain_dataloader(
        token_ids  = token_ids,
        block_size = MODEL_CFG.block_size,
        batch_size = BATCH_SIZE,
        val_split  = 0.05,
    )

    # Initialize trainer and run
    trainer = Trainer(
        model          = model,
        train_loader   = train_loader,
        val_loader     = val_loader,
        config         = MODEL_CFG,
        checkpoint_dir = CHECKPOINT_DIR,
        max_lr         = MAX_LR,
        min_lr         = MIN_LR,
        warmup_steps   = WARMUP_STEPS,
        total_steps    = TOTAL_STEPS,
        weight_decay   = WEIGHT_DECAY,
        grad_clip      = GRAD_CLIP,
        eval_interval  = 500,
        save_interval  = 1000,
        log_interval   = 50,
    )

    trainer.train()

    # Quick generation sample after training
    print("\n" + "=" * 60)
    print("Sample generation after pre-training:")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    prompt = "Once upon a time"
    prompt_ids = tokenizer.encode(prompt)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        generated = model.generate(
            idx,
            max_new_tokens=100,
            temperature=0.8,
            eos_id=tokenizer.eos_id,
        )

    generated_text = tokenizer.decode(generated[0].tolist())
    print(f"\nPrompt: '{prompt}'")
    print(f"Generated:\n{generated_text}\n")


if __name__ == "__main__":
    main()
