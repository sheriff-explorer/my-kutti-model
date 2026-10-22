"""
evaluation/run_eval.py

CLI tool to evaluate any TinyGPT checkpoint across:
  1. Perplexity on a holdout test set
  2. Question Answering Benchmark (Exact Match and F1)
  3. Qualitative comparison of responses across checkpoints

Usage:
  # Evaluate single checkpoint:
  python evaluation/run_eval.py --checkpoint checkpoints/posttrain/best_model.pt

  # Compare pretrain vs midtrain vs posttrain:
  python evaluation/run_eval.py --compare
"""

import os
import sys
import argparse
import json

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import ModelConfig
from model.tokenizer import BPETokenizer
from model.transformer import TinyGPT
from evaluation.evaluator import ModelEvaluator
from evaluation.benchmarks import BenchmarkSuite


BENCHMARK_QA = [
    ("What color is the sky?", "The sky is blue during the day."),
    ("Where do fish live?", "Fish live in water, in rivers, lakes, and the ocean."),
    ("What do birds use to fly?", "Birds use their wings to fly through the sky."),
    ("What is the sun?", "The sun is a big star that gives us light and warmth."),
    ("What falls from the sky in winter?", "Snow falls from the sky when it is very cold."),
    ("What do bees make?", "Bees make honey in their hives."),
    ("What is water made of?", "Water is made of tiny parts called hydrogen and oxygen."),
    ("What color is grass?", "Grass is green and grows in fields and gardens."),
]


def load_model_and_tokenizer(checkpoint_path: str, tokenizer_path: str):
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    tokenizer = BPETokenizer.load(tokenizer_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    cfg = ModelConfig(**checkpoint["config"])
    model = TinyGPT(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, tokenizer


def evaluate_checkpoint(checkpoint_path: str, tokenizer_path: str):
    print(f"\n========================================================")
    print(f" Evaluating Checkpoint: {checkpoint_path}")
    print(f"========================================================")
    
    model, tokenizer = load_model_and_tokenizer(checkpoint_path, tokenizer_path)
    evaluator = ModelEvaluator(model, tokenizer)

    results = evaluator.evaluate_qa_benchmark(BENCHMARK_QA)
    
    print(f"\nResults on Standard Benchmark ({results['samples_evaluated']} questions):")
    print(f"  Exact Match (EM): {results['exact_match_pct']:.1f}%")
    print(f"  F1 Score (Tokens): {results['f1_score_pct']:.1f}%")
    print("\nSample Outputs:")
    for item in results["sample_details"][:4]:
        print(f"\n  Q: {item['question']}")
        print(f"  Expected: {item['gold_answer']}")
        print(f"  Got:      {item['predicted']}")
        print(f"  Score:    EM={item['exact_match']}, F1={item['f1']:.2f}")

    # Standard Dataset Benchmarks
    print("\n--- Standard Benchmarks ---")
    suite = BenchmarkSuite(model, tokenizer)
    try:
        suite.evaluate_wikitext_perplexity(split="test[:200]")
    except Exception as e:
        print(f"  WikiText PPL skipped: {e}")
        
    try:
        suite.evaluate_arc_easy(split="test[:50]")
    except Exception as e:
        print(f"  ARC-Easy skipped: {e}")

    try:
        suite.evaluate_boolq(split="validation[:50]")
    except Exception as e:
        print(f"  BoolQ skipped: {e}")

    return results


def run_comparison(tokenizer_path: str):
    stages = [
        ("Pre-train", "checkpoints/pretrain/best_model.pt"),
        ("Mid-train", "checkpoints/midtrain/best_model.pt"),
        ("Post-train (SFT)", "checkpoints/posttrain/best_model.pt"),
    ]

    print("\n========================================================")
    print(" Running Cross-Stage Progression Comparison")
    print("========================================================")

    comparison_results = []
    for stage_name, ckpt in stages:
        if not os.path.exists(ckpt):
            print(f"  [Skipping {stage_name}: {ckpt} not found yet]")
            continue
        print(f"\n--- Testing {stage_name} ---")
        res = evaluate_checkpoint(ckpt, tokenizer_path)
        comparison_results.append({
            "stage": stage_name,
            "em": res["exact_match_pct"],
            "f1": res["f1_score_pct"]
        })

    if comparison_results:
        print("\n========================================================")
        print(" Summary Progression Table")
        print("========================================================")
        print(f"{'Stage':<20} | {'Exact Match':<12} | {'F1 Score':<10}")
        print("-" * 50)
        for r in comparison_results:
            print(f"{r['stage']:<20} | {r['em']:>10.1f}% | {r['f1']:>8.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Evaluate TinyGPT checkpoints")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/posttrain/best_model.pt")
    parser.add_argument("--tokenizer", type=str, default="checkpoints/tokenizer.json")
    parser.add_argument("--compare", action="store_true", help="Compare all available stages")
    args = parser.parse_args()

    if args.compare:
        run_comparison(args.tokenizer)
    else:
        evaluate_checkpoint(args.checkpoint, args.tokenizer)


if __name__ == "__main__":
    main()
