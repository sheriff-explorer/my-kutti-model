"""
inference/generate.py

Generate text using trained TinyGPT PyTorch checkpoints.

Usage:
  # Single prompt:
  python inference/generate.py --prompt "Once upon a time"

  # Custom checkpoint & parameters:
  python inference/generate.py --checkpoint checkpoints/pretrain/best_model.pt --temperature 0.7 --max_tokens 100

  # Interactive mode:
  python inference/generate.py --interactive
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import ModelConfig
from model.tokenizer import BPETokenizer
from model.transformer import TinyGPT


def load_model(checkpoint_path: str, tokenizer_path: str, device: str):
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    tokenizer = BPETokenizer.load(tokenizer_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    cfg = ModelConfig(**checkpoint["config"])
    model = TinyGPT(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    step = checkpoint.get("step", "unknown")
    val_loss = checkpoint.get("best_val_loss", None)
    val_str = f" (best val loss: {val_loss:.4f})" if val_loss is not None else ""
    print(f"Loaded checkpoint from {checkpoint_path} [Step: {step}{val_str}]")
    print(f"Device: {device}")
    return model, tokenizer


def generate_text(model, tokenizer, prompt: str, max_tokens: int = 80, temperature: float = 0.7, device: str = "cpu"):
    prompt_ids = tokenizer.encode(prompt)
    if len(prompt_ids) == 0:
        prompt_ids = [tokenizer.bos_id]
    
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(
            idx=idx,
            max_new_tokens=max_tokens,
            temperature=temperature,
            eos_id=tokenizer.eos_id,
        )
    return tokenizer.decode(out[0].tolist())


def main():
    parser = argparse.ArgumentParser(description="TinyGPT Text Generation")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/pretrain/best_model.pt")
    parser.add_argument("--tokenizer", type=str, default="checkpoints/tokenizer.json")
    parser.add_argument("--prompt", type=str, default="Once upon a time, there was a little girl named Lily.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=80)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--interactive", action="store_true", help="Run interactive prompting session")
    args = parser.parse_args()

    model, tokenizer = load_model(args.checkpoint, args.tokenizer, args.device)

    if args.interactive:
        print("\nEntering interactive mode. Type 'exit' or 'quit' to stop.\n")
        while True:
            try:
                prompt = input("Prompt > ").strip()
                if not prompt or prompt.lower() in ["exit", "quit"]:
                    break
                gen = generate_text(model, tokenizer, prompt, args.max_tokens, args.temperature, args.device)
                print(f"\n{gen}\n" + "-" * 50)
            except (KeyboardInterrupt, EOFError):
                break
        print("\nExiting.")
    else:
        print("=" * 60)
        print(f"Prompt: {args.prompt}")
        print("=" * 60)
        gen = generate_text(model, tokenizer, args.prompt, args.max_tokens, args.temperature, args.device)
        print("Generated Output:\n")
        print(gen)
        print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
