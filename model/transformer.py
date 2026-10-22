"""
model/transformer.py

TinyGPT -- A ~2M parameter decoder-only Transformer.

Architecture overview (forward pass):
  Token IDs -> Token Embedding + Positional Embedding
            -> [TransformerBlock × n_layers]
            -> Final LayerNorm
            -> LM Head (weight-tied to token embedding)
            -> Logits [batch, seq_len, vocab_size]

This file is written to be maximally educational -- every design decision
and tensor shape is explained inline.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import ModelConfig


# ═══════════════════════════════════════════════════════════════════════════════
# Sub-module 1: Causal Multi-Head Self-Attention
# ═══════════════════════════════════════════════════════════════════════════════

class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention.

    'Self' = the input attends to itself (not a separate encoder).
    'Causal' = each position can only attend to earlier positions (autoregressive).
    'Multi-Head' = multiple parallel attention operations, each in a subspace.

    Parameter breakdown (per block, with d_model=160, n_heads=4):
      QKV projection: [d_model, 3 × d_model] = [160, 480] -> 76,800 params
      Out projection: [d_model, d_model]      = [160, 160] -> 25,600 params
      Total: 102,400 params per block
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0, (
            f"d_model ({cfg.d_model}) must be divisible by n_heads ({cfg.n_heads})"
        )

        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads  # 160 // 4 = 40

        # ── Q, K, V projections fused into a single matrix ──────────────────
        # Instead of three separate [d_model, d_model] matrices for Q, K, V,
        # we use one [d_model, 3*d_model] matrix and chunk it.
        # This is faster due to a single matmul + memory locality.
        #
        # No bias: bias=False reduces params and is common in modern LLMs.
        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)

        # ── Output projection ──────────────────────────────────────────────
        # After all heads are concatenated back together [B, T, d_model],
        # we apply one final linear to mix the head outputs.
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        # ── Dropout ───────────────────────────────────────────────────────
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, C]  where B=batch, T=seq_len, C=d_model

        Returns:
            out: [B, T, C]  same shape as input
        """
        B, T, C = x.shape   # e.g. [64, 256, 160]

        # ── Step 1: Compute Q, K, V ─────────────────────────────────────────
        # qkv: [B, T, 3*C] -> split into 3 tensors of [B, T, C]
        qkv = self.qkv_proj(x)
        d = self.d_model
        q = qkv[:, :, :d]
        k = qkv[:, :, d:2*d]
        v = qkv[:, :, 2*d:]

        # Reshape for multi-head: [B, T, C] -> [B, n_heads, T, head_dim]
        # Each head sees a different 40-dimensional "slice" of the representation.
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, T, hd]
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, T, hd]
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, T, hd]

        # ── Step 2: Scaled Dot-Product Attention with Causal Mask ────────────
        #
        #   att = softmax(Q @ Kᵀ / √head_dim) @ V
        #
        # The scaling by 1/√head_dim prevents dot products from growing too large
        # (which would push softmax into near-zero gradient regions).
        #
        # Causal masking: position i cannot attend to positions j > i.
        # We use PyTorch's efficient built-in which uses Flash Attention when available.
        #
        # is_causal=True automatically creates the triangular mask:
        #   [1, 0, 0]
        #   [1, 1, 0]
        #   [1, 1, 1]
        # where 0 means "masked / not visible"
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True,
        )
        # y: [B, H, T, hd]

        # ── Step 3: Concatenate heads and project ──────────────────────────
        # Merge all heads back: [B, H, T, hd] -> [B, T, H*hd] = [B, T, d_model]
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Final linear projection to mix information across heads
        out = self.resid_dropout(self.out_proj(y))
        return out  # [B, T, C]


# ═══════════════════════════════════════════════════════════════════════════════
# Sub-module 2: Feed-Forward Network
# ═══════════════════════════════════════════════════════════════════════════════

class FFNBlock(nn.Module):
    """
    Position-wise Feed-Forward Network.

    Applied independently at each position (no cross-token interaction).
    Acts as a key-value memory: the up-projection selects 'what to look up',
    the activation gates it, and the down-projection writes the result.

    Architecture:
        Linear(d_model -> ffn_dim)  ->  GELU  ->  Linear(ffn_dim -> d_model)
        [160 -> 640]                   GELU     [640 -> 160]

    Parameter breakdown (d_model=160, ffn_dim=640):
        Up:   160 × 640 = 102,400
        Down: 640 × 160 = 102,400
        Total: 204,800 per block
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()

        # Up-projection: expands to higher-dimensional space
        self.up = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)

        # Down-projection: contracts back to d_model
        self.down = nn.Linear(cfg.ffn_dim, cfg.d_model, bias=False)

        # Dropout for regularization
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]

        Returns:
            out: [B, T, d_model]

        The GELU activation is smooth (unlike ReLU's hard zero).
        GELU(x) ≈ x * Φ(x) where Φ is the Gaussian CDF.
        Preferred over ReLU in modern transformer architectures (BERT, GPT-2+).
        """
        return self.dropout(self.down(F.gelu(self.up(x))))


# ═══════════════════════════════════════════════════════════════════════════════
# Sub-module 3: Transformer Block
# ═══════════════════════════════════════════════════════════════════════════════

class TransformerBlock(nn.Module):
    """
    One complete Transformer block = Attention + FFN, each with Pre-LN + Residual.

    Pre-LayerNorm (Pre-LN) design:
        x = x + Attention(LayerNorm(x))    ← normalize BEFORE attention
        x = x + FFN(LayerNorm(x))          ← normalize BEFORE FFN

    Why Pre-LN instead of Post-LN (original paper)?
    - Pre-LN: more stable training, allows higher LRs, no warmup strictly needed.
    - Post-LN: the original design; can diverge without careful LR warmup.
    - GPT-2, LLaMA, and most modern models use Pre-LN.

    The residual connections (x = x + ...) are critical:
    - Create a "gradient highway" for backprop -- gradients flow directly to early layers.
    - Model learns small corrections (residuals) rather than full new representations.
    - Without residuals, 6-layer networks are very hard to train.

    Parameter breakdown:
        LayerNorm × 2:  2 × 2 × 160  =    640
        Attention:                     102,400
        FFN:                           204,800
        Total per block:               307,840
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()

        # LayerNorm before attention
        # Parameters: weight (γ) and bias (β), each of size d_model
        self.ln1 = nn.LayerNorm(cfg.d_model)

        # Multi-head causal self-attention
        self.attn = CausalSelfAttention(cfg)

        # LayerNorm before FFN
        self.ln2 = nn.LayerNorm(cfg.d_model)

        # Feed-forward network
        self.ffn = FFNBlock(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pre-LN forward:
            x -> LN -> Attention -> + x (residual)
              -> LN -> FFN       -> + x (residual)
        """
        # Attention sub-layer with residual
        x = x + self.attn(self.ln1(x))

        # FFN sub-layer with residual
        x = x + self.ffn(self.ln2(x))

        return x


# ═══════════════════════════════════════════════════════════════════════════════
# Main Model: TinyGPT
# ═══════════════════════════════════════════════════════════════════════════════

class TinyGPT(nn.Module):
    """
    TinyGPT -- ~2M parameter decoder-only language model.

    Complete forward pass:
        1. Token IDs -> token embeddings        [B, T] -> [B, T, d_model]
        2. Add positional embeddings           [B, T, d_model]
        3. Apply dropout                       (regularization)
        4. Pass through n_layers TransformerBlocks
        5. Final LayerNorm
        6. LM Head projection to vocab logits  [B, T, d_model] -> [B, T, vocab_size]
        7. (Optional) Compute cross-entropy loss against shifted targets
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.config = cfg

        # ── Embeddings ──────────────────────────────────────────────────────
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        """
        Token Embedding Table: [vocab_size, d_model] = [512, 160] = 81,920 params

        Maps each integer token ID to a d_model-dimensional vector.
        nn.Embedding is essentially a lookup table -- internally it's a matrix
        where row i contains the embedding for token ID i.

        These vectors are learned during training. After training, similar tokens
        will have similar embedding vectors (measured by cosine similarity).
        """

        self.position_embedding = nn.Embedding(cfg.block_size, cfg.d_model)
        """
        Positional Embedding Table: [block_size, d_model] = [256, 160] = 40,960 params

        Maps each position index (0, 1, 2, ..., 255) to a d_model-dimensional vector.
        This is added to the token embedding so the model knows WHERE each token is.

        Alternative: RoPE (Rotary Position Embeddings) -- adds position information
        inside the attention computation instead of the embedding. Used in LLaMA.
        We use learned positional embeddings for simplicity.
        """

        # ── Embedding dropout ────────────────────────────────────────────────
        self.embed_dropout = nn.Dropout(cfg.dropout)

        # ── Transformer Blocks ───────────────────────────────────────────────
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        """
        ModuleList of n_layers=6 TransformerBlock instances.
        Each block: 307,840 params.
        All 6 blocks: 1,847,040 params (93.8% of total model params!).

        Why ModuleList instead of Sequential?
        - More explicit: we iterate manually in forward(), which is clearer.
        - Allows easy access to individual blocks (e.g., for layer-wise LR decay later).
        """

        # ── Final LayerNorm ──────────────────────────────────────────────────
        self.final_ln = nn.LayerNorm(cfg.d_model)
        """
        Applied after all transformer blocks, before the LM head.
        Stabilizes the representation before the output projection.
        Params: 2 × d_model = 320
        """

        # ── LM Head ─────────────────────────────────────────────────────────
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        """
        Projects [B, T, d_model] -> [B, T, vocab_size].
        Output is raw logits (unnormalized scores) for each vocabulary token.
        Loss = cross_entropy(logits, targets) = -log(softmax(logits)[target]).
        """

        # ── Weight Tying ─────────────────────────────────────────────────────
        if cfg.tie_weights:
            self.lm_head.weight = self.token_embedding.weight
            """
            Share the token embedding matrix with the LM head.

            Shape must match: both are [vocab_size, d_model].
            nn.Linear stores weights transposed, but .weight gives the original
            [out_features, in_features] = [vocab_size, d_model] matrix.

            This enforces: the vector used to ENCODE a token (as input)
            is also used to SCORE that token (as output prediction).
            Encourages consistent, high-quality token representations.

            Saves: vocab_size × d_model = 512 × 160 = 81,920 parameters.
            """

        # ── Weight Initialization ────────────────────────────────────────────
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """
        Initialize model weights.

        Good initialization is crucial -- bad init leads to vanishing/exploding
        gradients in early training.

        Rules:
        - Linear layers: small normal distribution (std=0.02, from GPT-2 paper).
          This keeps activations and gradients in a reasonable range at layer 0.
        - Embeddings: same small normal (std=0.02).
        - LayerNorm: weight=1 (identity scale), bias=0 (no shift). Standard.
        - Output projections (attn out + ffn down): scaled by 1/√(2 × n_layers).
          This "scaled init" (from GPT-2) prevents the residual stream from
          growing too large with depth. Each block adds to the residual, so
          deeper networks need smaller initial contributions per block.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

        # Special scaled init for residual output projections
        for name, param in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("down.weight"):
                nn.init.normal_(
                    param,
                    mean=0.0,
                    std=0.02 / math.sqrt(2 * self.cfg.n_layers)
                )

    # ──────────────────────────────────────────────────────────────────────────
    # Forward Pass
    # ──────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor = None,
    ):
        """
        Forward pass.

        Args:
            idx:     Token ID tensor [B, T] -- integers in [0, vocab_size)
            targets: Target token IDs [B, T] for computing loss.
                     Can contain -100 for positions to ignore (SFT loss masking).
                     If None, no loss is computed (inference mode).

        Returns:
            logits: [B, T, vocab_size] -- raw prediction scores
            loss:   scalar tensor if targets provided, else None
        """
        B, T = idx.shape
        assert T <= self.cfg.block_size, (
            f"Sequence length {T} exceeds block_size {self.cfg.block_size}. "
            "Truncate your input."
        )

        # ── 1. Token Embeddings ─────────────────────────────────────────────
        tok_emb = self.token_embedding(idx)  # [B, T, d_model]

        # ── 2. Positional Embeddings ────────────────────────────────────────
        # Create position indices [0, 1, 2, ..., T-1] for the current sequence.
        # These are looked up in the positional embedding table.
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # [T]
        pos_emb = self.position_embedding(pos)  # [T, d_model] -> broadcasts over B

        # ── 3. Combine + Dropout ────────────────────────────────────────────
        # Element-wise addition: each token's embedding now encodes both
        # its identity (what it is) and its position (where it is).
        x = self.embed_dropout(tok_emb + pos_emb)  # [B, T, d_model]

        # ── 4. Transformer Blocks ────────────────────────────────────────────
        for block in self.blocks:
            x = block(x)  # [B, T, d_model] -> [B, T, d_model] (shape preserved)

        # ── 5. Final LayerNorm ───────────────────────────────────────────────
        x = self.final_ln(x)  # [B, T, d_model]

        # ── 6. LM Head -> Logits ─────────────────────────────────────────────
        logits = self.lm_head(x)  # [B, T, vocab_size]

        # ── 7. Loss (optional) ───────────────────────────────────────────────
        loss = None
        if targets is not None:
            # Standard cross-entropy: compare each position's prediction to its target.
            # ignore_index=-100: positions masked in SFT are excluded from the loss.
            #
            # Reshape for cross_entropy:
            #   logits:  [B, T, V] -> [B*T, V]
            #   targets: [B, T]    -> [B*T]
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    # ──────────────────────────────────────────────────────────────────────────
    # Text Generation
    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        eos_id: int = None,
    ) -> torch.Tensor:
        """
        Autoregressive text generation.

        Generates one token at a time by repeatedly feeding the current sequence
        back into the model and sampling from the predicted distribution.

        Args:
            idx:            Starting token IDs [B, T] (the prompt)
            max_new_tokens: Maximum number of tokens to generate
            temperature:    Sampling temperature.
                            - 1.0: sample from true distribution
                            - < 1.0 (e.g. 0.7): sharper, more conservative
                            - -> 0.0: greedy (always pick the highest-prob token)
                            - > 1.0: more random/creative
            eos_id:         If provided, stop generation when this token is produced

        Returns:
            idx: [B, T + max_new_tokens] -- original prompt + generated tokens

        How it works (one step):
            1. Crop sequence to block_size (model can't process longer inputs)
            2. Forward pass -> logits [B, T, vocab_size]
            3. Take logits at the LAST position only (that's our next-token prediction)
            4. Apply temperature scaling
            5. Convert to probabilities via softmax
            6. Sample from the probability distribution
            7. Append sampled token to the sequence
            8. Repeat
        """
        self.eval()  # Disable dropout during generation

        for _ in range(max_new_tokens):
            # Crop to block_size if needed (circular truncation)
            idx_cond = idx[:, -self.cfg.block_size:]  # [B, T_cond]

            # Forward pass (no loss computation needed at inference)
            logits, _ = self.forward(idx_cond)

            # Take logits at the final position -> [B, vocab_size]
            logits = logits[:, -1, :]

            # Apply temperature:
            # Dividing logits by temperature < 1 makes the distribution sharper
            # (higher peaks, lower tails -> more confident predictions).
            # Dividing by temperature > 1 flattens the distribution (more random).
            if temperature == 0.0:
                # Greedy decoding: always pick the argmax
                idx_next = logits.argmax(dim=-1, keepdim=True)  # [B, 1]
            else:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)        # [B, vocab_size]
                idx_next = torch.multinomial(probs, 1)   # [B, 1] -- sample

            # Append to sequence and continue
            idx = torch.cat([idx, idx_next], dim=1)      # [B, T+1]

            # Stop early if EOS token generated
            if eos_id is not None and (idx_next == eos_id).all():
                break

        return idx

    # ──────────────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────────────

    def count_params(self, non_embedding: bool = False) -> int:
        """
        Count trainable parameters.

        Args:
            non_embedding: If True, exclude embedding params from the count.
                           Useful because embeddings scale with vocab_size,
                           which is somewhat arbitrary.

        Returns:
            Total number of trainable parameters.
        """
        params = [p for p in self.parameters() if p.requires_grad]
        if non_embedding:
            params = [
                p for p in params
                if p is not self.token_embedding.weight
                and p is not self.position_embedding.weight
            ]
        n = sum(p.numel() for p in params)
        print(f"  Total {'(excl. embeddings) ' if non_embedding else ''}params: {n:,} ({n/1e6:.3f}M)")
        return n

    def get_num_params(self) -> dict:
        """Detailed parameter count by component."""
        breakdown = {}
        breakdown["token_embedding"] = self.token_embedding.weight.numel()
        breakdown["position_embedding"] = self.position_embedding.weight.numel()
        for i, block in enumerate(self.blocks):
            attn_params = sum(p.numel() for p in block.attn.parameters())
            ffn_params = sum(p.numel() for p in block.ffn.parameters())
            ln_params = sum(p.numel() for p in block.ln1.parameters()) + \
                        sum(p.numel() for p in block.ln2.parameters())
            breakdown[f"block_{i}_attention"] = attn_params
            breakdown[f"block_{i}_ffn"] = ffn_params
            breakdown[f"block_{i}_layernorm"] = ln_params
        breakdown["final_layernorm"] = sum(p.numel() for p in self.final_ln.parameters())
        breakdown["lm_head"] = 0 if self.cfg.tie_weights else self.lm_head.weight.numel()
        return breakdown

    def __repr__(self) -> str:
        total = self.count_params()
        return (
            f"TinyGPT(\n"
            f"  vocab={self.cfg.vocab_size}, d_model={self.cfg.d_model}, "
            f"n_heads={self.cfg.n_heads}, n_layers={self.cfg.n_layers},\n"
            f"  block_size={self.cfg.block_size}, ffn_dim={self.cfg.ffn_dim}, "
            f"tie_weights={self.cfg.tie_weights}\n"
            f"  total_params={total:,}\n"
            f")"
        )
