"""
inference/run_inference.py

CPU Inference using ONNX Runtime -- no PyTorch needed at runtime.

This script demonstrates:
1. Loading the INT8 ONNX model
2. Loading the tokenizer (pure Python, no ML dependencies)
3. Running autoregressive text generation in a loop
4. Simple CLI interface

The ONNX Runtime + tokenizer is all you need to deploy this model.
Total inference dependencies: ~15MB (onnxruntime + tokenizers)
vs PyTorch training: ~2GB+

Run with:
    # Interactive mode:
    python inference/run_inference.py

    # Single prompt:
    python inference/run_inference.py --prompt "Q: What color is the sky? A:" --max_tokens 50

    # Use FP32 model (higher quality, slower):
    python inference/run_inference.py --model outputs/model_fp32.onnx
"""

import os
import sys
import argparse
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────────────────────
# Default Paths
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MODEL_PATH     = "outputs/model_int8.onnx"
DEFAULT_TOKENIZER_PATH = "checkpoints/tokenizer.json"
DEFAULT_MAX_TOKENS     = 100
DEFAULT_TEMPERATURE    = 0.8
BLOCK_SIZE             = 256   # Must match training block_size


# ─────────────────────────────────────────────────────────────────────────────
# ONNX Runtime Inference Session
# ─────────────────────────────────────────────────────────────────────────────

class TinyGPTInference:
    """
    Minimal inference wrapper for the exported ONNX TinyGPT model.

    No PyTorch. Uses only:
      - onnxruntime (inference engine)
      - tokenizers (fast BPE tokenizer, ~3MB)
      - numpy (tensor operations)
    """

    def __init__(self, model_path: str, tokenizer_path: str):
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as e:
            print(f"ERROR: Missing dependency: {e}")
            print("Install: pip install onnxruntime tokenizers")
            sys.exit(1)

        print(f"Loading ONNX model: {model_path}")
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        # Log available execution providers
        providers = self.session.get_providers()
        print(f"  Execution providers: {providers}")

        print(f"Loading tokenizer: {tokenizer_path}")
        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self.vocab_size = self._tokenizer.get_vocab_size()
        self.eos_id     = self._tokenizer.token_to_id("<eos>")
        self.bos_id     = self._tokenizer.token_to_id("<bos>")

        # Warm-up run (first inference is slower due to JIT compilation)
        dummy = np.zeros((1, 4), dtype=np.int64)
        self.session.run(None, {"input_ids": dummy})
        print(f"  Model ready. Vocab size: {self.vocab_size}")

    def encode(self, text: str) -> list:
        return self._tokenizer.encode(text).ids

    def decode(self, ids: list) -> str:
        return self._tokenizer.decode(ids)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        verbose: bool = False,
    ) -> str:
        """
        Autoregressive text generation using ONNX Runtime.

        Algorithm:
          1. Tokenize prompt -> list of token IDs
          2. For each new token:
             a. Run forward pass (ONNX) -> logits [1, T, vocab_size]
             b. Take logits at last position -> [vocab_size]
             c. Apply temperature scaling
             d. Softmax -> probability distribution
             e. Sample (or greedy if temp=0)
             f. Append to sequence
             g. Stop if EOS generated
          3. Decode all token IDs -> string

        Args:
            prompt:         Text prompt to continue
            max_new_tokens: Max tokens to generate
            temperature:    Sampling temperature (lower = more focused)
            verbose:        Print each token as it's generated

        Returns:
            Full generated text (prompt + response)
        """
        # Tokenize prompt
        token_ids = self.encode(prompt)

        if verbose:
            print(prompt, end="", flush=True)

        generated_ids = list(token_ids)
        t0 = time.time()
        n_generated = 0

        for _ in range(max_new_tokens):
            # Crop to block_size (model's max context window)
            context = generated_ids[-BLOCK_SIZE:]
            input_array = np.array([context], dtype=np.int64)  # [1, T]

            # ONNX forward pass -> logits [1, T, vocab_size]
            logits = self.session.run(None, {"input_ids": input_array})[0]

            # Take logits at last token position -> [vocab_size]
            next_logits = logits[0, -1, :]  # shape: [vocab_size]

            # Sample next token
            if temperature == 0.0:
                # Greedy: always pick the highest probability token
                next_id = int(np.argmax(next_logits))
            else:
                # Temperature-scaled softmax sampling
                next_logits = next_logits / temperature
                # Stable softmax: subtract max before exp to prevent overflow
                next_logits -= np.max(next_logits)
                probs = np.exp(next_logits)
                probs /= probs.sum()
                next_id = int(np.random.choice(self.vocab_size, p=probs))

            generated_ids.append(next_id)
            n_generated += 1

            if verbose:
                token_str = self.decode([next_id])
                print(token_str, end="", flush=True)

            # Stop at EOS
            if next_id == self.eos_id:
                break

        if verbose:
            print()  # Newline after generation

        elapsed = time.time() - t0
        tps = n_generated / elapsed if elapsed > 0 else 0

        return self.decode(generated_ids), tps


# ─────────────────────────────────────────────────────────────────────────────
# CLI Interface
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run TinyGPT inference using ONNX Runtime (CPU)"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_PATH,
        help=f"Path to ONNX model (default: {DEFAULT_MODEL_PATH})"
    )
    parser.add_argument(
        "--tokenizer", default=DEFAULT_TOKENIZER_PATH,
        help=f"Path to tokenizer JSON (default: {DEFAULT_TOKENIZER_PATH})"
    )
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Text prompt. If not provided, runs interactive mode."
    )
    parser.add_argument(
        "--max_tokens", type=int, default=DEFAULT_MAX_TOKENS,
        help=f"Max tokens to generate (default: {DEFAULT_MAX_TOKENS})"
    )
    parser.add_argument(
        "--temperature", type=float, default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE}). "
             f"0.0=greedy, 1.0=random, 0.7=balanced"
    )
    args = parser.parse_args()

    # Check files exist
    for path, name in [(args.model, "ONNX model"), (args.tokenizer, "Tokenizer")]:
        if not os.path.exists(path):
            print(f"ERROR: {name} not found at {path}")
            sys.exit(1)

    # Load model
    print("\n" + "=" * 60)
    print(" TinyGPT Inference -- ONNX Runtime (CPU)")
    print("=" * 60 + "\n")

    model = TinyGPTInference(args.model, args.tokenizer)

    print(f"\nSettings: max_tokens={args.max_tokens}, temperature={args.temperature}")
    print(f"Tips: Use 'Q: <question>\\nA:' format for Q&A mode.\n")

    if args.prompt:
        # Single prompt mode
        print(f"Prompt: {args.prompt}\n")
        output, tps = model.generate(
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            verbose=True,
        )
        print(f"\n[Generated {tps:.1f} tokens/sec]\n")
    else:
        # Interactive mode
        print("Interactive mode -- type prompts below (Ctrl+C to exit)\n")
        print("-" * 60)

        while True:
            try:
                prompt = input("\nPrompt: ").strip()
                if not prompt:
                    continue

                print("\nGenerating: ", end="")
                output, tps = model.generate(
                    prompt,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    verbose=True,
                )
                print(f"[{tps:.1f} tok/s]")

            except KeyboardInterrupt:
                print("\n\nExiting. Goodbye!")
                break


if __name__ == "__main__":
    main()
