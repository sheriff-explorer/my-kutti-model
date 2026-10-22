"""
training/trainer.py

Generic Trainer -- handles the full training loop for all three stages.

Responsibilities:
  - AdamW optimizer setup
  - Cosine LR schedule with warmup
  - Gradient clipping (prevent exploding gradients)
  - Periodic evaluation (track train/val loss)
  - Checkpoint saving (save best model + periodic saves)
  - Loss logging (to console + log file)
"""

import os
import time
import json
from typing import Optional

import torch
from torch.utils.data import DataLoader

from model.transformer import TinyGPT
from model.config import ModelConfig
from training.scheduler import get_lr


class Trainer:
    """
    Training loop for TinyGPT.

    Usage:
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=cfg,
            checkpoint_dir="checkpoints/pretrain",
            max_lr=3e-3,
            min_lr=3e-4,
            warmup_steps=200,
            total_steps=5000,
            grad_clip=1.0,
            eval_interval=500,
            save_interval=1000,
        )
        trainer.train()
    """

    def __init__(
        self,
        model: TinyGPT,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: ModelConfig,
        checkpoint_dir: str,
        # LR schedule
        max_lr: float = 3e-3,
        min_lr: float = 3e-4,
        warmup_steps: int = 200,
        total_steps: int = 5000,
        # Optimizer
        weight_decay: float = 0.1,
        beta1: float = 0.9,
        beta2: float = 0.95,
        # Training stability
        grad_clip: float = 1.0,
        # Logging
        eval_interval: int = 500,
        eval_batches: int = 100,
        save_interval: int = 1000,
        log_interval: int = 50,
        # Device
        device: Optional[str] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.checkpoint_dir = checkpoint_dir

        # Schedule params
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

        # Stability
        self.grad_clip = grad_clip
        """
        Gradient clipping:
        If the gradient norm exceeds grad_clip, all gradients are scaled down
        so that the total norm equals grad_clip.

        Why?
        ────
        Sometimes a single bad batch causes extremely large gradients (gradient
        explosion). Without clipping, this would completely destabilize the model
        weights in one step. With clipping, the update direction is preserved but
        the magnitude is bounded. grad_clip=1.0 is the standard for transformers.
        """

        # Logging
        self.eval_interval = eval_interval
        self.eval_batches = eval_batches
        self.save_interval = save_interval
        self.log_interval = log_interval

        # Device setup
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model = self.model.to(self.device)

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

        # ── Optimizer ─────────────────────────────────────────────────────
        # Separate parameter groups: weight decay only on "weight" params,
        # NOT on bias or LayerNorm params (they don't benefit from weight decay).
        #
        # Why separate groups?
        # Weight decay (L2 regularization) encourages weights to be small.
        # For LayerNorm and biases, this is inappropriate -- they set scale/shift
        # and should be free to take any value. Applying decay to them hurts training.
        decay_params     = [p for n, p in model.named_parameters() if p.dim() >= 2]
        no_decay_params  = [p for n, p in model.named_parameters() if p.dim() < 2]

        param_groups = [
            {"params": decay_params,    "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=max_lr,        # Will be overridden each step by scheduler
            betas=(beta1, beta2),
            eps=1e-8,
            fused=True if self.device == "cuda" else False,
            # fused=True: PyTorch's fused CUDA AdamW is significantly faster on GPU
        )
        """
        AdamW Hyperparameters:
        ──────────────────────
        beta1=0.9:  Exponential decay rate for the first moment (mean of gradients).
                    Controls how much of the previous gradient direction to keep.
                    0.9 = 90% old direction + 10% new gradient.

        beta2=0.95: Exponential decay rate for the second moment (variance of gradients).
                    Controls the adaptive per-parameter learning rate scaling.
                    Standard Adam uses 0.999; LLMs typically use 0.95 for more
                    responsive adaptation to gradient variance changes.
                    (Karpathy's nanoGPT, LLaMA training both use 0.95)

        eps=1e-8:   Small constant added to denominator to prevent division by zero.
                    When gradient variance is very small, eps prevents LR from
                    exploding to infinity.

        weight_decay=0.1: L2 regularization strength. Applied to weight matrices
                    but NOT to biases or LayerNorm parameters (hence two param groups).
        """

        # Training state
        self.step = 0
        self.best_val_loss = float("inf")
        self.log_data = []

        print(f"\nTrainer initialized:")
        print(f"  Device: {self.device}")
        if self.device == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Steps: {total_steps} | Warmup: {warmup_steps}")
        print(f"  LR: {max_lr:.2e} -> {min_lr:.2e} (cosine)")
        print(f"  Batch size: {train_loader.batch_size}")
        tokens_per_step = train_loader.batch_size * config.block_size
        print(f"  Tokens per step: {tokens_per_step:,}")
        print(f"  Total tokens seen: {tokens_per_step * total_steps:,}")

    # ──────────────────────────────────────────────────────────────────────────
    # Main Training Loop
    # ──────────────────────────────────────────────────────────────────────────

    def train(self) -> None:
        """
        Run the full training loop.

        Each step:
          1. Get next batch from DataLoader
          2. Set LR from cosine schedule
          3. Forward pass -> compute loss
          4. Backward pass -> compute gradients
          5. Clip gradients (stability)
          6. Optimizer step -> update weights
          7. Zero gradients for next step
          8. Periodic: evaluate, log, save checkpoint
        """
        self.model.train()
        train_iter = iter(self.train_loader)

        t_start = time.time()
        running_loss = 0.0

        print(f"\n{'='*60}")
        print(f"Starting training for {self.total_steps} steps")
        print(f"{'='*60}\n")

        while self.step < self.total_steps:
            # ── Get batch ─────────────────────────────────────────────────
            try:
                x, y = next(train_iter)
            except StopIteration:
                # End of epoch -- restart iterator
                train_iter = iter(self.train_loader)
                x, y = next(train_iter)

            x = x.to(self.device)
            y = y.to(self.device)

            # ── Set learning rate ─────────────────────────────────────────
            # Update LR every step according to cosine schedule
            lr = get_lr(
                step=self.step,
                max_lr=self.max_lr,
                min_lr=self.min_lr,
                warmup_steps=self.warmup_steps,
                total_steps=self.total_steps,
            )
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # ── Forward + Loss ────────────────────────────────────────────
            # Use mixed precision (bfloat16) on CUDA for speed.
            # bfloat16: 16-bit float with same exponent range as float32.
            # Reduces memory bandwidth by 2×, speeds up matmuls significantly.
            # Falls back to float32 on CPU.
            with torch.autocast(
                device_type=self.device,
                dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                enabled=(self.device == "cuda"),
            ):
                logits, loss = self.model(x, y)

            # ── Backward Pass ─────────────────────────────────────────────
            # Compute gradients of loss w.r.t. all model parameters.
            # This uses automatic differentiation (autograd) -- PyTorch tracks
            # all operations and computes gradients via the chain rule.
            self.optimizer.zero_grad(set_to_none=True)
            # set_to_none=True: slightly faster than zeroing (avoids memory write)

            loss.backward()

            # ── Gradient Clipping ─────────────────────────────────────────
            # Compute the global gradient norm and clip to grad_clip.
            # norm = sqrt(sum of squared gradients across all parameters)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip
            )

            # ── Optimizer Step ────────────────────────────────────────────
            # Update weights: w = w - lr * (adam_adjusted_gradient)
            self.optimizer.step()

            # Track loss
            running_loss += loss.item()
            self.step += 1

            # ── Logging ───────────────────────────────────────────────────
            if self.step % self.log_interval == 0:
                avg_loss = running_loss / self.log_interval
                elapsed = time.time() - t_start
                tokens_seen = self.step * self.train_loader.batch_size * self.config.block_size
                tps = tokens_seen / elapsed

                print(
                    f"Step {self.step:>6}/{self.total_steps} | "
                    f"loss={avg_loss:.4f} | "
                    f"lr={lr:.2e} | "
                    f"grad_norm={grad_norm:.3f} | "
                    f"tok/s={tps:,.0f}",
                    flush=True
                )
                self.log_data.append({
                    "step": self.step, "train_loss": avg_loss,
                    "lr": lr, "grad_norm": float(grad_norm),
                })
                running_loss = 0.0

            # ── Evaluation ────────────────────────────────────────────────
            if self.step % self.eval_interval == 0:
                val_loss = self._evaluate()
                print(f"\n  > Val loss: {val_loss:.4f} (best: {self.best_val_loss:.4f})\n", flush=True)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint("best_model.pt", val_loss)
                self._save_logs()

                self.model.train()

            # ── Periodic Save ─────────────────────────────────────────────
            if self.step % self.save_interval == 0:
                self._save_checkpoint(f"step_{self.step}.pt")
                self._save_logs()

        # Final checkpoint
        self._save_checkpoint("final_model.pt")
        self._save_logs()
        print(f"\nTraining complete. Best val loss: {self.best_val_loss:.4f}")

    # ──────────────────────────────────────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _evaluate(self) -> float:
        """
        Compute average loss on the validation set.

        @torch.no_grad(): disables gradient computation for efficiency.
        Validation doesn't need gradients -- we're not updating weights.
        This also reduces memory usage by ~50%.
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for i, (x, y) in enumerate(self.val_loader):
            if self.eval_batches and i >= self.eval_batches:
                break
            x = x.to(self.device)
            y = y.to(self.device)
            with torch.autocast(
                device_type=self.device,
                dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                enabled=(self.device == "cuda"),
            ):
                _, loss = self.model(x, y)
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    # ──────────────────────────────────────────────────────────────────────────
    # Checkpointing
    # ──────────────────────────────────────────────────────────────────────────

    def _save_checkpoint(self, filename: str, val_loss: float = None) -> None:
        """
        Save model state dict + optimizer state + training metadata.

        Why save optimizer state?
        ─────────────────────────
        Adam maintains per-parameter momentum (m) and variance (v) estimates.
        If you resume training without the optimizer state, Adam restarts with
        zero momentum -> unstable learning for the first few hundred steps.
        Always save optimizer state for resumable training.
        """
        path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step": self.step,
            "best_val_loss": self.best_val_loss,
            "config": {
                "d_model": self.config.d_model,
                "n_heads": self.config.n_heads,
                "n_layers": self.config.n_layers,
                "vocab_size": self.config.vocab_size,
                "block_size": self.config.block_size,
                "ffn_dim": self.config.ffn_dim,
                "dropout": self.config.dropout,
                "tie_weights": self.config.tie_weights,
            },
        }
        torch.save(checkpoint, path)
        tag = f" (val_loss={val_loss:.4f})" if val_loss else ""
        print(f"  [OK] Checkpoint saved: {path}{tag}")

    def _save_logs(self) -> None:
        """Save training log to JSON for later analysis/plotting."""
        log_path = os.path.join(self.checkpoint_dir, "training_log.json")
        with open(log_path, "w") as f:
            json.dump(self.log_data, f, indent=2)
        print(f"  [OK] Training log saved: {log_path}")
