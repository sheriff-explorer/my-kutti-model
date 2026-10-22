"""
stages/03_posttrain.py

Stage 3: Post-training -- Supervised Fine-Tuning (SFT).

Purpose:
  Convert the mid-trained language model into an interactive assistant that
  responds to structured prompts. We train it on Q&A pairs covering:
    1. Simple factual questions ("What color is the sky?")
    2. Story continuation ("Begin a story about a rabbit...")

Key technique: LOSS MASKING
  In standard CLM, the model is trained to predict every token.
  In SFT, we only want the model to learn to generate the ANSWER.
  Question tokens are masked to target=-100 (ignored by cross_entropy).

  Example:
    Input:  "Q: What color is the sky?\nA: The sky is blue."
    Tokens: [Q, :,  , W, h, a, t, ...,  A, :,  , T, h, e,  , s, k, y, ...]
    Target: [-100, -100, ..., -100,      A, :,  , T, h, e, ...]
             ← question masked ->          ← answer supervised ->

  This ensures the model learns: "given this question context, generate this answer"
  rather than "memorize the exact question text".

Run with:
    python stages/03_posttrain.py

Prerequisites:
    - Stage 2 must be complete (checkpoints/midtrain/best_model.pt exists)
    - checkpoints/tokenizer.json must exist
"""

import json
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import ModelConfig
from model.tokenizer import BPETokenizer
from model.transformer import TinyGPT
from training.dataset import create_sft_dataloader
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
    dropout     = 0.0,    # No dropout in SFT -- dataset is small, we need all signal
    tie_weights = True,
)

# Very conservative hyperparams for SFT
BATCH_SIZE   = 16      # Smaller batch: dataset is small, large batch = fewer updates
TOTAL_STEPS  = 1000
WARMUP_STEPS = 50
MAX_LR       = 1e-4    # Very low -- we're fine-tuning, not relearning
MIN_LR       = 1e-5
WEIGHT_DECAY = 0.01    # Minimal weight decay in SFT
GRAD_CLIP    = 1.0

MIDTRAIN_CHECKPOINT = "checkpoints/midtrain/best_model.pt"
TOKENIZER_PATH      = "checkpoints/tokenizer.json"
CHECKPOINT_DIR      = "checkpoints/posttrain"


def build_sft_dataset():
    """
    Build the Q&A + Story Continuation SFT pairs.

    Format: (question, answer) tuples.
    The SFTDataset class will handle:
      - Prepending "Q: " to question
      - Prepending "\\nA: " to answer
      - Loss masking of question tokens
      - Appending EOS to answer

    We combine two categories:
    1. Factual Q&A -- tests factual recall and simple reasoning
    2. Story starters -- tests creative continuation, leverages TinyStories pre-training
    """
    sft_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "posttrain_sft.jsonl")
    if os.path.exists(sft_file):
        print(f"  Loading SFT dataset from flat file: {sft_file}")
        augmented = []
        with open(sft_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    augmented.append((item["instruction"], item["response"]))
        print(f"SFT dataset: {len(augmented)} Q&A pairs loaded from flat file.")
        return augmented

    qa_pairs = []

    # ── Category 1: Simple Factual Q&A ──────────────────────────────────────
    factual_pairs = [
        ("What color is the sky?",          "The sky is blue during the day."),
        ("What do fish do in water?",        "Fish swim in water and breathe through their gills."),
        ("What do birds use to fly?",        "Birds use their wings to fly through the sky."),
        ("Where does rain come from?",       "Rain comes from clouds in the sky."),
        ("What do trees have on them?",      "Trees have leaves, branches, and roots."),
        ("What is the sun?",                 "The sun is a big star that gives us light and warmth."),
        ("What do dogs like to do?",         "Dogs like to play, run, and be with their owners."),
        ("Where do fish live?",              "Fish live in water, in rivers, lakes, and the ocean."),
        ("What falls from the sky in winter?","Snow falls from the sky when it is very cold."),
        ("What helps plants grow?",          "Sunlight, water, and soil help plants grow."),
        ("What is the moon?",                "The moon is a large rock that goes around the Earth."),
        ("What do cats like?",               "Cats like to sleep, play, and be warm and comfortable."),
        ("What season do flowers bloom?",    "Flowers bloom in spring when it gets warm."),
        ("What is bread made from?",         "Bread is made from flour and water and is baked in an oven."),
        ("What do eyes do?",                 "Eyes help us see the world around us."),
        ("What is water made of?",           "Water is made of tiny parts called hydrogen and oxygen."),
        ("What do ears do?",                 "Ears help us hear sounds and music."),
        ("What makes a rainbow?",            "A rainbow appears when sunlight shines through rain drops."),
        ("Where do apples grow?",            "Apples grow on trees in orchards."),
        ("What is fire?",                    "Fire is hot and bright and gives light and warmth."),
        ("What do bees make?",               "Bees make honey in their hives."),
        ("What is a library?",               "A library is a place with many books that people can read."),
        ("What do children do at school?",   "Children learn to read, write, and discover new things at school."),
        ("What is a nest?",                  "A nest is a home that birds build to lay their eggs."),
        ("What do frogs eat?",               "Frogs eat insects like flies and beetles."),
        ("What is a forest?",                "A forest is a large area filled with many trees."),
        ("What happens in autumn?",          "In autumn, leaves change colors and fall from the trees."),
        ("What does a farmer do?",           "A farmer grows food and takes care of animals."),
        ("What is a river?",                 "A river is a large stream of water that flows to the sea."),
        ("What do rabbits eat?",             "Rabbits eat carrots, grass, and other vegetables."),
        ("What is the ocean?",               "The ocean is a vast body of salt water covering much of Earth."),
        ("What does a baker make?",          "A baker makes bread, cakes, and other delicious food."),
        ("What is wind?",                    "Wind is moving air that we can feel on our face."),
        ("What do butterflies eat?",         "Butterflies drink nectar from flowers."),
        ("What is a mountain?",              "A mountain is a very tall piece of land that rises high."),
        ("Why do we sleep?",                 "We sleep so our bodies and minds can rest and grow stronger."),
        ("What is a bridge?",                "A bridge is a structure that helps people cross over water or roads."),
        ("What do bears eat?",               "Bears eat fish, berries, and honey."),
        ("What is the color of grass?",      "Grass is green and grows in fields and gardens."),
        ("What is a cloud?",                 "A cloud is made of tiny drops of water floating in the sky."),
    ]
    qa_pairs.extend(factual_pairs)

    # ── Category 2: Story Starters / Completions ─────────────────────────────
    story_pairs = [
        (
            "Begin a story about a rabbit in the forest.",
            "Once there was a little rabbit who lived in a big green forest. "
            "Every morning, it hopped through the tall grass looking for carrots."
        ),
        (
            "Start a story about a child who finds a treasure.",
            "One sunny afternoon, a young girl named Lily was playing in her garden. "
            "She dug a hole and found a shiny box buried deep in the ground."
        ),
        (
            "Begin a story about a kind dragon.",
            "In a land far away, there lived a dragon who was different from all the others. "
            "Instead of breathing fire to scare people, he used it to bake bread for the village."
        ),
        (
            "Start a story about a talking fish.",
            "Deep in a calm blue lake lived a small orange fish who could talk. "
            "Nobody knew about this gift until one day a little boy sat by the water."
        ),
        (
            "Begin a story about a lost puppy.",
            "A tiny brown puppy wandered away from its home one rainy afternoon. "
            "It ran through the wet streets, looking for a friendly face."
        ),
        (
            "Start a story about a brave little bird.",
            "High up in a tall oak tree, a tiny sparrow named Pip wanted to fly. "
            "All the other birds could soar above the clouds, but Pip was afraid."
        ),
        (
            "Begin a story about a wizard's apprentice.",
            "Every morning, a young boy named Finn would sweep the wizard's tower. "
            "One day, he accidentally knocked over a glowing blue bottle."
        ),
        (
            "Start a story about a snowman who comes to life.",
            "On the coldest night of winter, three children built a snowman in the yard. "
            "In the morning, they were amazed to find it had moved closer to the house."
        ),
        (
            "Begin a story about a kind farmer.",
            "Old Farmer Green lived on a small hill with lots of animals. "
            "He fed them every morning and sang songs to make them happy."
        ),
        (
            "Start a story about a child who learns to read.",
            "Maya sat under the big apple tree with her first real book. "
            "The words were hard at first, but she kept trying, one page at a time."
        ),
    ]
    qa_pairs.extend(story_pairs)

    # Augment by repeating with slight variation (simple data augmentation)
    augmented = []
    for q, a in qa_pairs:
        augmented.append((q, a))
        augmented.append(("Tell me: " + q, a))  # Vary the question phrasing slightly

    print(f"SFT dataset: {len(augmented)} Q&A pairs "
          f"({len(factual_pairs)} factual + {len(story_pairs)} story + augmented)")
    return augmented


def main():
    print("=" * 60)
    print(" TinyGPT -- Stage 2: Post-training (SFT)")
    print("=" * 60)

    # Load tokenizer
    if not os.path.exists(TOKENIZER_PATH):
        print(f"ERROR: Tokenizer not found at {TOKENIZER_PATH}")
        sys.exit(1)
    tokenizer = BPETokenizer.load(TOKENIZER_PATH)

    # Load mid-trained model
    if not os.path.exists(MIDTRAIN_CHECKPOINT):
        print(f"ERROR: Mid-trained checkpoint not found at {MIDTRAIN_CHECKPOINT}")
        print("       Please run Stage 2 first: python stages/02_midtrain.py")
        sys.exit(1)

    print(f"\nLoading mid-trained checkpoint: {MIDTRAIN_CHECKPOINT}")
    checkpoint = torch.load(MIDTRAIN_CHECKPOINT, map_location="cpu", weights_only=False)

    model = TinyGPT(MODEL_CFG)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.count_params()
    print(f"  Loaded from step {checkpoint['step']}")

    # Build SFT dataset
    print("\nBuilding SFT dataset...")
    pairs = build_sft_dataset()

    train_loader, val_loader = create_sft_dataloader(
        pairs        = pairs,
        tokenizer    = tokenizer,
        block_size   = MODEL_CFG.block_size,
        batch_size   = BATCH_SIZE,
        val_fraction = 0.15,
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
        eval_interval  = 100,
        save_interval  = 250,
        log_interval   = 20,
    )

    trainer.train()

    # Evaluation: test Q&A format
    print("\n" + "=" * 60)
    print("Q&A Evaluation after SFT:")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    test_questions = [
        "Q: What color is the sky?\nA:",
        "Q: Where do fish live?\nA:",
        "Q: Begin a story about a rabbit in the forest.\nA:",
        "Q: What is the sun?\nA:",
    ]

    for prompt in test_questions:
        ids = tokenizer.encode(prompt)
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            gen = model.generate(idx, max_new_tokens=80, temperature=0.7,
                                 eos_id=tokenizer.eos_id)
        output = tokenizer.decode(gen[0].tolist())
        print(f"\n{prompt}")
        print(f"-> {output[len(prompt):]}")


if __name__ == "__main__":
    main()
