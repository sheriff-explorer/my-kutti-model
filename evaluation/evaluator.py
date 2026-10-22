"""
evaluation/evaluator.py

Comprehensive Evaluation Framework for TinyGPT.

This module evaluates the model across 3 key pillars of language model assessment:

1. Perplexity (PPL):
   - Definition: exp(CrossEntropyLoss).
   - Intuition: The effective "branching factor" or average number of equally probable
     words the model is choosing between at each step.
   - A model with random guessing across 512 tokens has PPL = 512.
   - A well-trained model achieves PPL between 3.0 and 8.0.

2. Benchmark Q&A Evaluation (Exact Match & F1 / Token Overlap):
   - Evaluates factual knowledge recall and question following.
   - Compares generated answers against ground truth using:
     - Exact Match (EM): 1 if normalized strings match identically, 0 otherwise.
     - Token Overlap / F1 Score: Precision & recall of bag-of-words overlap.

3. Stage Comparison / Progression Tracking:
   - Tracks how the model evolves across Pre-train -> Mid-train -> Post-train.
"""

import os
import re
import math
import string
import json
from typing import List, Tuple, Dict, Any, Optional

import torch
from torch.utils.data import DataLoader

from model.transformer import TinyGPT
from model.tokenizer import BPETokenizer
from model.config import ModelConfig


# ─────────────────────────────────────────────────────────────────────────────
# Text Normalization Helpers (Standard SQuAD/TriviaQA eval style)
# ─────────────────────────────────────────────────────────────────────────────

def normalize_text(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score between prediction and ground truth."""
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(ground_truth).split()

    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0

    num_same = sum(min(pred_tokens.count(w), gold_tokens.count(w)) for w in common)
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Compute binary exact match after normalization."""
    return float(normalize_text(prediction) == normalize_text(ground_truth))


# ─────────────────────────────────────────────────────────────────────────────
# Model Evaluator Class
# ─────────────────────────────────────────────────────────────────────────────

class ModelEvaluator:
    """
    Evaluates a TinyGPT model checkpoint.
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

    @torch.no_grad()
    def evaluate_perplexity(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Calculates Cross-Entropy Loss and Perplexity over a validation DataLoader.

        Perplexity = exp(loss)
        """
        total_loss = 0.0
        total_batches = 0

        for x, y in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)
            with torch.autocast(
                device_type=self.device,
                dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                enabled=(self.device == "cuda"),
            ):
                _, loss = self.model(x, y)

            total_loss += loss.item()
            total_batches += 1

        avg_loss = total_loss / max(total_batches, 1)
        try:
            ppl = math.exp(avg_loss)
        except OverflowError:
            ppl = float("inf")

        return {
            "val_loss": avg_loss,
            "perplexity": ppl,
        }

    @torch.no_grad()
    def evaluate_qa_benchmark(
        self,
        qa_pairs: List[Tuple[str, str]],
        max_new_tokens: int = 40,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Evaluates Q&A performance against ground-truth answers.
        """
        results = []
        total_em = 0.0
        total_f1 = 0.0

        for question, gold_answer in qa_pairs:
            prompt = f"Q: {question}\nA:"
            input_ids = self.tokenizer.encode(prompt)
            idx = torch.tensor([input_ids], dtype=torch.long, device=self.device)

            generated_tokens = self.model.generate(
                idx=idx,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                eos_id=self.tokenizer.eos_id,
            )

            full_text = self.tokenizer.decode(generated_tokens[0].tolist())
            
            if "\nA:" in full_text:
                pred_answer = full_text.split("\nA:", 1)[1].strip()
            elif "A:" in full_text:
                pred_answer = full_text.split("A:", 1)[1].strip()
            else:
                pred_answer = full_text[len(prompt):].strip()

            pred_clean = pred_answer.split("\n")[0].strip()

            em = compute_exact_match(pred_clean, gold_answer)
            f1 = compute_f1(pred_clean, gold_answer)

            total_em += em
            total_f1 += f1

            results.append({
                "question": question,
                "gold_answer": gold_answer,
                "predicted": pred_clean,
                "exact_match": em,
                "f1": f1,
            })

        n = len(qa_pairs)
        return {
            "samples_evaluated": n,
            "exact_match_pct": (total_em / n) * 100 if n > 0 else 0.0,
            "f1_score_pct": (total_f1 / n) * 100 if n > 0 else 0.0,
            "sample_details": results,
        }
