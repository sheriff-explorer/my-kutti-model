"""
training/scheduler.py

Cosine Learning Rate Schedule with Linear Warmup.

Why do we need a LR schedule?
───────────────────────────────
A fixed learning rate is rarely optimal:

1. START (warmup): Model weights are random. A large LR causes huge, destabilizing
   gradient steps. We ramp from 0 -> max_lr over warmup_steps to let the model
   stabilize before full-speed learning.

2. MIDDLE: Train at max_lr -- the model learns the fastest here.

3. END (cosine decay): Reduce LR gradually. As the model approaches a good minimum,
   large steps "overshoot" and oscillate. Smaller steps allow fine convergence.
   Cosine is smooth (no abrupt changes), which empirically works better than
   step decay (e.g., halving LR every N steps).

The full schedule looks like:
   LR
   │         /‾‾‾‾‾\
   │        /         \
   │       /             ‾‾‾‾‾‾‾‾‾‾‾‾_____
   │      /
   │_____/
   └──────────────────────────────────────> steps
     warmup    cosine decay to min_lr
"""

import math


def get_lr(
    step: int,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
    total_steps: int,
) -> float:
    """
    Compute the learning rate for a given training step.

    Args:
        step:          Current training step (0-indexed)
        max_lr:        Peak learning rate (reached after warmup)
        min_lr:        Minimum LR at end of training (typically max_lr / 10)
        warmup_steps:  Steps to linearly increase from 0 -> max_lr
        total_steps:   Total training steps (after which LR = min_lr)

    Returns:
        lr: float -- learning rate to use at this step

    Phases:
        1. Warmup  [0, warmup_steps):    LR = max_lr * (step / warmup_steps)
        2. Decay   [warmup_steps, total]: LR = cosine decay from max_lr -> min_lr
        3. Plateau [total+]:              LR = min_lr (clamp)
    """
    # Phase 1: Linear warmup
    if step < warmup_steps:
        # Linearly ramp from 0 to max_lr
        return max_lr * (step / warmup_steps)

    # Phase 3: After total_steps, hold at min_lr
    if step >= total_steps:
        return min_lr

    # Phase 2: Cosine annealing
    # Map current step to [0, 1] range within the decay window
    decay_steps = total_steps - warmup_steps
    progress = (step - warmup_steps) / decay_steps  # 0.0 -> 1.0

    # Cosine decay: starts at max_lr, ends at min_lr
    # cos(0) = 1, cos(π) = -1
    # (1 + cos(π * progress)) / 2 goes from 1 -> 0
    cosine_coefficient = 0.5 * (1.0 + math.cos(math.pi * progress))
    lr = min_lr + cosine_coefficient * (max_lr - min_lr)
    return lr
