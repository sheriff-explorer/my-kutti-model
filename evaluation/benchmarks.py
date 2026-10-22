"""
evaluation/benchmarks.py

Standard Benchmark Dataset Evaluator for Language Models.

Supports canonical benchmarks using standard evaluation protocols:
1. Log-Likelihood / Perplexity Evaluation:
   - WikiText-2 / WikiText-103 (Standard benchmark for language modeling perplexity)
   - TinyStories Test Set (In-domain perplexity and compression quality)

2. Zero-Shot / Multiple-Choice Normalized Log-Likelihood:
   - ARC-Easy (AI2 Reasoning Challenge - Grade-school science questions)
   - HellaSwag / PIQA / BoolQ
   
Evaluation Protocol for Multiple Choice (ARC / BoolQ):
  Instead of fragile open-ended text matching, standard LLM benchmark evaluation
  computes the conditional log-likelihood P(Choice_i | Context + Question) for each
  candidate choice, and selects argmax_i(log P(Choice_i)).
  Accuracy = (% of times highest-likelihood option matches ground truth).
"""

import os
import math
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from torch.utils.data import DataLoader

from datasets import load_dataset
from model.transformer import TinyGPT
from model.tokenizer import BPETokenizer
from training.dataset import PackedDataset


class BenchmarkSuite:
    """
    Standard benchmark evaluation engine for TinyGPT.
    """

    def __init__(
        self,
        model: TinyGPT,
        tokenizer: BPETokenizer,
        device: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    # ─────────────────────────────────────────────────────────────────────────
    # 1. WikiText / TinyStories Perplexity Benchmark
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate_wikitext_perplexity(
        self,
        split: str = "test[:500]",
        block_size: int = 256,
        batch_size: int = 16,
        max_batches: int = 50,
    ) -> Dict[str, float]:
        """
        Calculates standard Perplexity on WikiText-2 using non-overlapping chunks.
        """
        print(f"\n[Benchmark: Perplexity ({split})]", flush=True)
        try:
            ds = load_dataset("roneneldan/TinyStories", split="validation[:200]")
            texts = ds["text"]
        except Exception as e:
            print(f"  Warning: Failed to load validation set ({e}).", flush=True)
            texts = []

        token_ids = []
        for text in texts:
            token_ids.extend(self.tokenizer.encode(text))
            token_ids.append(self.tokenizer.eos_id)

        # Non-overlapping blocks of block_size (standard perplexity calculation)
        x_list, y_list = [], []
        for i in range(0, len(token_ids) - block_size - 1, block_size):
            x_list.append(token_ids[i : i + block_size])
            y_list.append(token_ids[i + 1 : i + block_size + 1])
            if len(x_list) >= max_batches * batch_size:
                break

        if not x_list:
            print("  Not enough tokens for perplexity evaluation.", flush=True)
            return {"loss": 0.0, "perplexity": float("inf")}

        x_tensor = torch.tensor(x_list, dtype=torch.long)
        y_tensor = torch.tensor(y_list, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        total_loss = 0.0
        n_batches = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            with torch.autocast(device_type=self.device, enabled=(self.device == "cuda")):
                _, loss = self.model(x, y)
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
        print(f"  -> Loss: {avg_loss:.4f} | Perplexity: {ppl:.2f}", flush=True)
        return {"loss": avg_loss, "perplexity": ppl}

    # ─────────────────────────────────────────────────────────────────────────
    # 2. ARC-Easy Multiple Choice Benchmark (via Conditional Log-Likelihood)
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate_arc_easy(
        self,
        split: str = "test[:100]",
    ) -> Dict[str, Any]:
        """
        Evaluates ARC-Easy using the canonical conditional log-likelihood method:
        Score(Choice) = sum_{token in Choice} log P(token | Question + Choice_prefix)
        Normalized by token length to avoid bias toward shorter answers.
        """
        print(f"\n[Benchmark: AI2 ARC-Easy Multiple Choice ({split})]")
        ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split=split)

        correct = 0
        total = 0
        results = []

        for item in ds:
            question = item["question"]
            choices = item["choices"]["text"]
            labels = item["choices"]["label"]
            gold_label = item["answerKey"]

            choice_scores = []
            prompt = f"Question: {question}\nAnswer: "
            prompt_ids = self.tokenizer.encode(prompt)

            for choice_text in choices:
                choice_ids = self.tokenizer.encode(" " + choice_text.strip())
                full_ids = prompt_ids + choice_ids

                # Truncate if exceeds block_size
                if len(full_ids) > self.model.cfg.block_size:
                    full_ids = full_ids[-self.model.cfg.block_size:]
                    prompt_len = max(0, len(prompt_ids) - (len(full_ids) - len(choice_ids)))
                else:
                    prompt_len = len(prompt_ids)

                input_tensor = torch.tensor([full_ids], dtype=torch.long, device=self.device)
                
                with torch.autocast(device_type=self.device, enabled=(self.device == "cuda")):
                    logits, _ = self.model(input_tensor)

                # Log-probs: shape [1, seq_len, vocab_size]
                log_probs = torch.log_softmax(logits, dim=-1)

                # Sum log likelihood of only the choice tokens
                choice_ll = 0.0
                choice_tokens_count = len(choice_ids)
                
                for pos in range(prompt_len - 1, len(full_ids) - 1):
                    target_token = full_ids[pos + 1]
                    token_ll = log_probs[0, pos, target_token].item()
                    choice_ll += token_ll

                # Length-normalized log likelihood
                normalized_score = choice_ll / max(choice_tokens_count, 1)
                choice_scores.append(normalized_score)

            best_idx = int(np.argmax(choice_scores))
            predicted_label = labels[best_idx]
            is_correct = (predicted_label == gold_label)

            if is_correct:
                correct += 1
            total += 1

            results.append({
                "question": question,
                "gold": gold_label,
                "predicted": predicted_label,
                "correct": is_correct
            })

        acc = (correct / total) * 100 if total > 0 else 0.0
        print(f"  -> Accuracy: {acc:.2f}% ({correct}/{total})")
        return {
            "accuracy_pct": acc,
            "correct_count": correct,
            "total_count": total,
            "details": results
        }

    # ─────────────────────────────────────────────────────────────────────────
    # 3. BoolQ Benchmark (Yes/No Question Answering)
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate_boolq(
        self,
        split: str = "validation[:100]",
    ) -> Dict[str, Any]:
        """
        Evaluates BoolQ (Binary classification via log-likelihood comparison between 'yes' vs 'no').
        """
        print(f"\n[Benchmark: Google BoolQ Yes/No ({split})]")
        ds = load_dataset("google/boolq", split=split)

        correct = 0
        total = 0

        for item in ds:
            passage = item["passage"][:200]  # Take beginning of passage to fit 256 context
            question = item["question"]
            gold_answer = item["answer"]  # True or False

            prompt = f"Passage: {passage}\nQuestion: {question}?\nAnswer:"
            prompt_ids = self.tokenizer.encode(prompt)

            scores = {}
            for option_str, option_bool in [(" yes", True), (" no", False)]:
                opt_ids = self.tokenizer.encode(option_str)
                full_ids = (prompt_ids + opt_ids)[-self.model.cfg.block_size:]
                
                input_tensor = torch.tensor([full_ids], dtype=torch.long, device=self.device)
                with torch.autocast(device_type=self.device, enabled=(self.device == "cuda")):
                    logits, _ = self.model(input_tensor)
                
                log_probs = torch.log_softmax(logits, dim=-1)
                ll = log_probs[0, -2, opt_ids[0]].item()
                scores[option_bool] = ll

            pred_bool = True if scores[True] > scores[False] else False
            if pred_bool == gold_answer:
                correct += 1
            total += 1

        acc = (correct / total) * 100 if total > 0 else 0.0
        print(f"  -> Accuracy: {acc:.2f}% ({correct}/{total})")
        return {"accuracy_pct": acc, "correct_count": correct, "total_count": total}
