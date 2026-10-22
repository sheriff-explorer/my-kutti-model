"""
model/config.py

ModelConfig -- single source of truth for all hyperparameters.

Every hyperparameter is documented with its purpose and the reasoning
behind the chosen default value for the ~2M parameter TinyGPT.
"""

import json
import os
from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    # ─────────────────────────────────────────
    # Vocabulary / Tokenizer
    # ─────────────────────────────────────────

    vocab_size: int = 512
    """
    Number of unique tokens the model knows.

    We use a small BPE tokenizer trained on TinyStories with vocab_size=512.
    This gives us 256 raw byte tokens + 256 learned BPE merges.

    Why 512?
    - Large vocab (e.g. 50K like GPT-2) -> embeddings alone cost 50K×160 = 8M params.
      That's 4× our entire model budget, impossible.
    - Small vocab (e.g. 64 chars) -> very long sequences (every letter is one token).
      Longer sequences = quadratic attention cost and harder learning.
    - 512 hits the sweet spot: covers all bytes, learns common short words as single tokens.
    """

    # ─────────────────────────────────────────
    # Architecture Dimensions
    # ─────────────────────────────────────────

    d_model: int = 160
    """
    The 'hidden dimension' -- every token is represented as a vector of this size
    throughout the entire network.

    Think of d_model as the 'width' of the model. Every embedding, every attention
    output, every FFN output is exactly d_model-dimensional.

    Why 160?
    - d_model=160 gives us ~2M total params with our other choices.
    - Too small (e.g. 32): not enough capacity to learn useful representations.
    - Too large (e.g. 512): exceeds param budget.
    - 160 must be divisible by n_heads (160 / 4 = 40 per head -- clean split).
    """

    n_heads: int = 4
    """
    Number of parallel attention heads in Multi-Head Self-Attention.

    The total d_model is split equally across heads:
        head_dim = d_model / n_heads = 160 / 4 = 40

    Each head operates in its own 40-dimensional subspace and can specialize
    in learning different types of relationships (syntactic, semantic, positional, etc.)

    Why 4?
    - Must divide d_model evenly (160 / 4 = 40). [OK]
    - 2 heads: too few specialization slots.
    - 8 heads: head_dim = 20, which is very small and reduces each head's expressiveness.
    - 4 is the balanced choice for d_model=160.
    """

    n_layers: int = 6
    """
    Number of stacked Transformer blocks (depth of the network).

    Each layer allows the model to build on representations from previous layers,
    creating increasingly abstract and compositional features.

    Why 6?
    - Depth is generally more efficient than width for language modeling.
    - 6 layers allows for a good hierarchy: low layers learn syntax/morphology,
      mid layers learn semantics, high layers learn pragmatics/discourse.
    - GPT-1 used 12 layers at d_model=768; we scale down proportionally.
    - More layers -> more compute per forward pass (but still fast at this scale).
    """

    ffn_dim: int = 640
    """
    The 'inner' (expanded) dimension of the Feed-Forward Network.

    The FFN expands from d_model -> ffn_dim, applies GELU activation,
    then contracts back from ffn_dim -> d_model.

    Standard: ffn_dim = 4 × d_model = 4 × 160 = 640

    Why 4×?
    - Convention from the original 'Attention Is All You Need' paper (Vaswani 2017).
    - The expansion creates a higher-dimensional space where non-linear transformations
      can be computed. Research suggests FFN layers act as key-value memory stores.
    - Reducing this (e.g. 2×) saves params but hurts model quality.
    - We keep the standard 4× ratio.
    """

    block_size: int = 256
    """
    The maximum context length (sequence length) in tokens.

    During training, all sequences are packed into chunks of exactly block_size tokens.
    At inference, the model cannot look back further than block_size tokens.

    Why 256?
    - 256 BPE tokens ≈ 150-200 characters ≈ 1-3 sentences of context.
    - Attention is O(n²) in sequence length. Doubling block_size -> 4× more compute.
    - 256 gives enough context for coherent short stories/answers without huge cost.
    - Also determines the size of positional embeddings: block_size × d_model.
    """

    # ─────────────────────────────────────────
    # Regularization
    # ─────────────────────────────────────────

    dropout: float = 0.1
    """
    Dropout probability: randomly zero out this fraction of activations during training.

    Acts as a regularizer -- prevents the model from relying too heavily on any single
    activation or co-adapting neurons (which leads to overfitting).

    Why 0.1?
    - Small model = high overfitting risk -> some dropout helps.
    - Too high (e.g. 0.5): model can't learn efficiently; too much signal destroyed.
    - 0.1 is the standard for small transformers.
    - Dropout is DISABLED at inference (model.eval() sets all dropout to 0).
    """

    # ─────────────────────────────────────────
    # Weight Tying
    # ─────────────────────────────────────────

    tie_weights: bool = True
    """
    Whether to share weights between the input token embedding and the output LM head.

    The token embedding matrix has shape [vocab_size, d_model] = [512, 160].
    The LM head (output projection) also has shape [vocab_size, d_model].

    With tie_weights=True, we use the SAME matrix for both, saving:
        512 × 160 = 81,920 parameters

    Why tie?
    - Saves 82K params (4% of our budget) -- significant at 2M scale.
    - Often IMPROVES quality: the model learns consistent representations
      for both 'what token am I (input)' and 'what token should come next (output)'.
    - Press & Wolf (2017) showed weight tying is beneficial for language models.
    """

    def count_params_estimate(self) -> int:
        """
        Estimate total trainable parameters analytically.
        Returns an integer approximation.
        """
        V, D, H, L, F, B = (
            self.vocab_size, self.d_model, self.n_heads,
            self.n_layers, self.ffn_dim, self.block_size
        )
        tok_emb = V * D                         # shared with LM head if tie_weights
        pos_emb = B * D
        per_block = (
            3 * D * D +                         # QKV projection (fused)
            D * D +                             # Output projection
            D * F +                             # FFN up
            F * D +                             # FFN down
            2 * 2 * D                           # 2 LayerNorms (weight + bias each)
        )
        all_blocks = L * per_block
        final_ln = 2 * D
        lm_head = 0 if self.tie_weights else V * D
        return tok_emb + pos_emb + all_blocks + final_ln + lm_head

    def save(self, path: str) -> None:
        """Save config to JSON for reproducibility."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        print(f"Config saved to {path}")

    @classmethod
    def load(cls, path: str) -> "ModelConfig":
        """Load config from JSON."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    def __repr__(self) -> str:
        est = self.count_params_estimate()
        return (
            f"ModelConfig(\n"
            f"  vocab_size={self.vocab_size}, d_model={self.d_model},\n"
            f"  n_heads={self.n_heads}, n_layers={self.n_layers},\n"
            f"  block_size={self.block_size}, ffn_dim={self.ffn_dim},\n"
            f"  dropout={self.dropout}, tie_weights={self.tie_weights}\n"
            f"  -> estimated params: {est:,} ({est/1e6:.3f}M)\n"
            f")"
        )
