"""
stages/02_midtrain.py

Stage 2: Mid-training (Domain Adaptation / Continual Pre-training).

Purpose:
  Take the pre-trained base model and continue training it on a smaller,
  higher-quality, more focused corpus. This narrows the model's distribution
  toward our target domain (factual Q&A + story continuation) while retaining
  the general language knowledge from Stage 1.

Key differences from Stage 1:
  - Load existing checkpoint (not random init)
  - Much LOWER learning rate (5e-4 vs 3e-3) -- prevents catastrophic forgetting
  - Fewer steps (2000 vs 5000) -- smaller dataset, we don't want to overfit
  - Data is curated and domain-focused, not general web crawl

What is catastrophic forgetting?
  When you fine-tune a model on new data with too high a LR, the optimizer
  aggressively updates weights to minimize the new loss, "overwriting" the
  weights that encoded pre-trained knowledge. The model "forgets" what it
  learned in Stage 1. Low LR prevents this by making small, careful updates.

Run with:
    python stages/02_midtrain.py

Prerequisites:
    - Stage 1 must be complete (checkpoints/pretrain/best_model.pt exists)
    - checkpoints/tokenizer.json must exist
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import ModelConfig
from model.tokenizer import BPETokenizer
from model.transformer import TinyGPT
from training.dataset import create_pretrain_dataloader
from training.trainer import Trainer


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MODEL_CFG = ModelConfig(
    vocab_size  = 512,
    d_model     = 160,
    n_heads     = 4,
    n_layers    = 6,
    block_size  = 256,
    ffn_dim     = 640,
    dropout     = 0.05,   # Lower dropout in later stages (less regularization needed)
    tie_weights = True,
)

# Mid-training uses LOWER lr and FEWER steps
BATCH_SIZE   = 32
TOTAL_STEPS  = 2000
WARMUP_STEPS = 50
MAX_LR       = 5e-4     # 6× lower than pre-training
MIN_LR       = 5e-5
WEIGHT_DECAY = 0.1
GRAD_CLIP    = 1.0

PRETRAIN_CHECKPOINT = "checkpoints/pretrain/best_model.pt"
TOKENIZER_PATH      = "checkpoints/tokenizer.json"
CHECKPOINT_DIR      = "checkpoints/midtrain"


def generate_curated_corpus(tokenizer: BPETokenizer) -> list:
    """
    Generate a curated domain-focused corpus for mid-training.

    This combines:
    1. Simple factual statements (entity-attribute pairs)
    2. Short story templates with cause-effect structure

    In a real project, you'd use actual high-quality data (Wikipedia summaries,
    curated Q&A datasets, etc.). For our 2M model, synthetic data works fine.
    """
    print("  Generating curated mid-training corpus...")

    texts = []

    # ── Domain 1: Simple factual statements ─────────────────────────────────
    facts = [
        "The sky is blue during the day and orange at sunset.",
        "Water is made of hydrogen and oxygen.",
        "The sun is a star at the center of our solar system.",
        "Dogs are friendly animals that like to play and run.",
        "Cats sleep a lot and like to be comfortable and warm.",
        "Trees grow tall and have leaves that turn colors in autumn.",
        "Birds have wings and can fly high in the sky.",
        "Fish live in water and breathe through their gills.",
        "The moon goes around the Earth and shines at night.",
        "Rain comes from clouds and helps plants grow.",
        "Apples grow on trees and are red, green, or yellow.",
        "Children learn new things every day at school.",
        "The ocean is very deep and full of many creatures.",
        "Snow falls in winter when it is very cold outside.",
        "Flowers bloom in spring when the weather gets warm.",
        "Bread is made from flour, water, and baked in an oven.",
        "Fire is hot and gives light but must be handled carefully.",
        "Books contain stories and information that people enjoy.",
        "A family lives together and takes care of each other.",
        "Friends make life happy and fun by playing together.",
    ]

    # Repeat and vary to create more data
    for _ in range(100):
        texts.extend(facts)

    # ── Domain 2: Short story snippets ──────────────────────────────────────
    story_templates = [
        "Once there was a little {animal} who lived in the {place}. "
        "Every day it would {action}. One day, something special happened.",

        "A young {character} wanted to learn how to {skill}. "
        "At first it was hard, but with practice, they got better and better.",

        "In a small village, there lived a kind {character}. "
        "Everyone loved them because they always helped their neighbors.",

        "The little {animal} found a shiny {object} in the forest. "
        "It did not know what it was, so it asked its friends for help.",

        "One morning, {name} woke up and saw that the sky was very bright. "
        "It was going to be a wonderful day full of adventures.",
    ]

    animals     = ["rabbit", "fox", "bear", "bird", "cat", "dog", "turtle", "owl"]
    places      = ["forest", "meadow", "pond", "garden", "hill", "valley"]
    actions     = ["search for food", "explore the woods", "sing a song", "play with friends"]
    characters  = ["child", "girl", "boy", "farmer", "baker"]
    skills      = ["read", "draw", "cook", "swim", "ride a bicycle"]
    objects     = ["stone", "leaf", "acorn", "flower", "feather"]
    names       = ["Lily", "Tom", "Emma", "Jack", "Sophie", "Ben", "Anna", "Max"]

    import random
    random.seed(42)

    for _ in range(500):
        template = random.choice(story_templates)
        story = template.format(
            animal    = random.choice(animals),
            place     = random.choice(places),
            action    = random.choice(actions),
            character = random.choice(characters),
            skill     = random.choice(skills),
            object    = random.choice(objects),
            name      = random.choice(names),
        )
        texts.append(story)

    print(f"  Generated {len(texts):,} text samples.")

    # Tokenize
    eos_id = tokenizer.eos_id
    all_ids = []
    for text in texts:
        ids = tokenizer.encode(text)
        all_ids.extend(ids)
        all_ids.append(eos_id)

    print(f"  Total tokens: {len(all_ids):,}")
    return all_ids


def main():
    print("=" * 60)
    print(" TinyGPT -- Stage 2: Mid-training (Domain Adaptation)")
    print("=" * 60)

    # Load tokenizer
    if not os.path.exists(TOKENIZER_PATH):
        print(f"ERROR: Tokenizer not found at {TOKENIZER_PATH}")
        print("       Please run Stage 1 first: python stages/01_pretrain.py")
        sys.exit(1)
    tokenizer = BPETokenizer.load(TOKENIZER_PATH)

    # Load pre-trained model
    if not os.path.exists(PRETRAIN_CHECKPOINT):
        print(f"ERROR: Pre-trained checkpoint not found at {PRETRAIN_CHECKPOINT}")
        print("       Please run Stage 1 first: python stages/01_pretrain.py")
        sys.exit(1)

    print(f"\nLoading pre-trained checkpoint: {PRETRAIN_CHECKPOINT}")
    checkpoint = torch.load(PRETRAIN_CHECKPOINT, map_location="cpu", weights_only=False)

    model = TinyGPT(MODEL_CFG)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.count_params()
    print(f"  Loaded from step {checkpoint['step']} "
          f"(pre-train val loss: {checkpoint['best_val_loss']:.4f})")

    # Generate curated corpus
    print("\nPreparing mid-training corpus...")
    token_ids = generate_curated_corpus(tokenizer)

    # DataLoaders
    train_loader, val_loader = create_pretrain_dataloader(
        token_ids  = token_ids,
        block_size = MODEL_CFG.block_size,
        batch_size = BATCH_SIZE,
        val_split  = 0.1,
    )

    # Train
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
        eval_interval  = 200,
        save_interval  = 500,
        log_interval   = 25,
    )

    trainer.train()

    # Sample generation
    print("\n" + "=" * 60)
    print("Sample generation after mid-training:")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    prompts = [
        "The sky is",
        "Once there was a little rabbit",
        "Water is made of",
    ]

    for prompt in prompts:
        ids = tokenizer.encode(prompt)
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            gen = model.generate(idx, max_new_tokens=60, temperature=0.8)
        print(f"\nPrompt: '{prompt}'")
        print(f"Output: {tokenizer.decode(gen[0].tolist())}")


if __name__ == "__main__":
    main()
