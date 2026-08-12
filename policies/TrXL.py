"""
TrXLExtractor with proper Transformer-XL relative positional encoding,
opt-in diagnostic capture (attention weights, hidden states, gate
activations) used by trxl_memory_diagnostics.py, and three independent
opt-out switches for the architectural extras layered on top of vanilla
TrXL: gating, CNN spatial biasing, and hyperconnections.

WHAT CHANGED vs the previous version (this pass):
  - ADDED three constructor booleans on TrXLExtractor, all default True
    so existing checkpoints/configs behave IDENTICALLY unless a caller
    explicitly opts out:
      * use_gating           -- GRU-style gate (GTrXL) vs plain residual
      * use_spatial_bias     -- sigmoid(pos)-gated CNN features vs raw CNN
      * use_hyperconnections -- learned multi-input mixing vs plain
                                 pass-through residual stream
    Each flag is threaded down to the module that actually implements the
    behavior (GatingUnit, HyperConnection, and the CNN fusion step) rather
    than being handled by conditionals scattered through forward(), so the
    "disabled" path is a real structural fallback, not a no-op multiply.
  - GatingUnit and HyperConnection now take an `enabled: bool` kwarg. When
    disabled, they skip building their learnable parameters entirely
    (no dead weights sitting unused in the state_dict) and their forward()
    reduces to the mathematically standard substitute:
      * GatingUnit(enabled=False)      -> x + sublayer_out   (plain residual)
      * HyperConnection(enabled=False) -> sublayer_out        (plain pass-through)
    Composed together (gating off + hyperconnections off), a block reduces
    to a standard pre-norm Transformer-XL block: x = x + sublayer(norm(x)).
  - GatingUnit.last_gate is now Optional and is explicitly set to None when
    gating is disabled (rather than left stale from a previous call), so
    diagnostic code can distinguish "no gate value because gating is off"
    from "forward() hasn't run yet."
  - use_spatial_bias=False skips constructing pos_to_cnn_bias altogether
    (not just skipping its use), so no unused parameters are created.

  Everything else -- relative positional encoding, memory update/indexing,
  NaN/Inf checks, diagnostic return shapes -- is unchanged from the
  previous version.

COMPATIBILITY:
  - Drop-in, non-breaking: TrXLExtractor(...) with no new kwargs behaves
    exactly as before (all three new flags default True).
  - Attention weight shape returned to diagnostics is unchanged:
    (B, n_heads, 1, memory_len+1) per layer.
  - Checkpoints trained with all three flags at their defaults load
    exactly as before. Checkpoints trained with any flag set to False are
    only loadable into an extractor constructed with that same flag value
    (parameter set differs -- this is expected and intentional, not a bug).
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

import numpy as np
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class GatingUnit(nn.Module):
    """
    GTrXL-style gated residual combination of x and a sublayer's output.

    enabled=True  (default, original behavior):
        g = sigmoid(Linear([x, sublayer_out]))
        out = g * sublayer_out + (1 - g) * x
    enabled=False:
        out = x + sublayer_out   (standard Transformer residual connection)

    When disabled, no gate parameters are created at all -- this isn't a
    bypass flag on an otherwise-live module, it's a real fallback to the
    vanilla residual that gating is meant to replace.
    """
    def __init__(self, d_model: int, gate_bias: float = 0.0, enabled: bool = True):
        super().__init__()
        self.enabled = enabled
        if self.enabled:
            self.gate_linear = nn.Linear(d_model * 2, d_model)
            nn.init.constant_(self.gate_linear.bias, gate_bias)
        else:
            self.gate_linear = None
        # Diagnostic-only: stores the most recent gate activation (detached,
        # no grad/memory cost beyond one tensor) so training code can inspect
        # "is this gate actually open" directly, without a separate forward
        # pass or hooks. Overwritten every forward call; read it right after.
        # Explicitly None when gating is disabled -- there is no gate value.
        self.last_gate = None

    def forward(self, x: torch.Tensor, sublayer_out: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            self.last_gate = None
            return x + sublayer_out

        combined = torch.cat([x, sublayer_out], dim=-1)
        g        = torch.sigmoid(self.gate_linear(combined))
        self.last_gate = g.detach()
        return g * sublayer_out + (1.0 - g) * x


class HyperConnection(nn.Module):
    """
    Learned mixing of all prior hidden states plus the current sublayer
    output (dynamic dense residual connections).

    enabled=True (default, original behavior):
        mixed = softmax(alpha, dim=0)-weighted sum of hidden_states
        out   = mixed + beta * sublayer_out
    enabled=False:
        out = sublayer_out   (plain pass-through; hidden_states is ignored)

    When disabled, no alpha/beta parameters are created. Combined with a
    plain-residual GatingUnit, a block then behaves as a standard TrXL
    block: x <- x + sublayer(norm(x)), with no cross-layer mixing.
    """
    def __init__(self, n_layers: int, d_model: int, layer_idx: int, enabled: bool = True):
        super().__init__()
        self.enabled   = enabled
        self.layer_idx = layer_idx
        n_inputs       = layer_idx + 1

        if self.enabled:
            self.alpha = nn.Parameter(torch.zeros(n_inputs, d_model))
            self.beta  = nn.Parameter(torch.ones(1, d_model))
        else:
            self.alpha = None
            self.beta  = None

    def forward(self, hidden_states: list, sublayer_out: torch.Tensor):
        if not self.enabled:
            return sublayer_out

        assert len(hidden_states) == self.alpha.shape[0], \
            f"Expected {self.alpha.shape[0]} hidden states, got {len(hidden_states)}"

        weights = torch.softmax(self.alpha, dim=0)
        stacked = torch.stack(hidden_states, dim=0)

        mixed = (weights.unsqueeze(1).unsqueeze(1) * stacked).sum(dim=0)

        return mixed + self.beta * sublayer_out


# 
# Relative positional encoding (TrXL-style)
# 

class RelativeSinusoidalEncoding(nn.Module):
    """
    Generates a fixed sinusoidal embedding R_r for every relative distance
    r = 0 (current token) .. memory_len (oldest cached token).

    Row j of the returned tensor corresponds to kv sequence index j, i.e.
    it's pre-aligned to match kv = torch.cat([memory, x_norm], dim=1):
      j=0                -> oldest memory slot  -> distance = memory_len
      j=memory_len-1      -> newest memory slot  -> distance = 1
      j=memory_len        -> current token       -> distance = 0

    This is a fixed, non-trainable buffer computed once at construction.
    Crucially it does NOT depend on absolute step count anywhere -- the
    query is always "now" by construction, so these relative distances
    are valid on every single forward call, forever, with no drift.
    """
    def __init__(self, d_model: int, memory_len: int):
        super().__init__()
        distances = torch.arange(memory_len, -1, -1).unsqueeze(1).float()  # (memory_len+1, 1)
        div_term  = torch.exp(
            torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model)
        )
        pe = torch.zeros(memory_len + 1, d_model)
        pe[:, 0::2] = torch.sin(distances * div_term)
        pe[:, 1::2] = torch.cos(distances * div_term)
        self.register_buffer('pe', pe)  # (memory_len+1, d_model)

    def forward(self):
        return self.pe


class RelativeMultiHeadAttention(nn.Module):
    """
    Transformer-XL relative attention (Dai et al. 2019), specialised for
    a single-token query (the "current" token) attending over a fixed-length
    key/value sequence (memory + current, length memory_len+1).

    score(j) = (q + u)^T k_content_j        content-content + global content bias
             + (q + v)^T k_position_j       content-position + global position bias

    k_content_j comes from the actual token content at kv position j.
    k_position_j comes from the FIXED relative-distance embedding at j,
    shared across all layers (each layer has its own projection W_k_pos
    to interpret it, and its own u/v bias vectors).
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = self.d_head ** -0.5

        self.q_proj       = nn.Linear(d_model, d_model, bias=False)
        self.k_content_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_pos_proj    = nn.Linear(d_model, d_model, bias=False)
        self.v_proj        = nn.Linear(d_model, d_model, bias=False)
        self.out_proj       = nn.Linear(d_model, d_model)

        # Global bias vectors (Dai et al.'s u, v) -- one pair per layer,
        # shared across all query positions within this layer.
        self.u = nn.Parameter(torch.zeros(n_heads, self.d_head))
        self.v = nn.Parameter(torch.zeros(n_heads, self.d_head))
        nn.init.xavier_uniform_(self.u)
        nn.init.xavier_uniform_(self.v)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x, kv_content, rel_pos_emb, need_weights: bool = False):
        """
        x:           (B, 1, D)              query -- the current token
        kv_content:  (B, L, D)               key/value content, L = memory_len+1
        rel_pos_emb: (L, D)                  fixed relative-distance embedding,
                                              pre-aligned to kv_content's index order
        Returns: (attn_out (B,1,D), attn_weights (B,n_heads,1,L) or None)
        """
        B, L, _ = kv_content.shape
        H, Dh   = self.n_heads, self.d_head

        q   = self.q_proj(x).view(B, 1, H, Dh).permute(0, 2, 1, 3)               # (B,H,1,Dh)
        k_e = self.k_content_proj(kv_content).view(B, L, H, Dh).permute(0, 2, 1, 3)  # (B,H,L,Dh)
        v   = self.v_proj(kv_content).view(B, L, H, Dh).permute(0, 2, 1, 3)          # (B,H,L,Dh)
        k_r = self.k_pos_proj(rel_pos_emb).view(L, H, Dh).permute(1, 0, 2)           # (H,L,Dh)

        u = self.u.unsqueeze(0).unsqueeze(2)  # (1,H,1,Dh)
        v_bias = self.v.unsqueeze(0).unsqueeze(2)  # (1,H,1,Dh)

        # content-content + global content bias
        AC = torch.matmul(q + u, k_e.transpose(-2, -1))          # (B,H,1,L)
        # content-position + global position bias (position term is
        # layer-shared across batch, so broadcast over B via einsum)
        BD = torch.einsum('bhqd,hld->bhql', q + v_bias, k_r)     # (B,H,1,L)

        scores = (AC + BD) * self.scale
        attn_weights = torch.softmax(scores, dim=-1)              # (B,H,1,L)
        attn_weights_dropped = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights_dropped, v)                # (B,H,1,Dh)
        out = out.permute(0, 2, 1, 3).reshape(B, 1, self.d_model)
        out = self.out_proj(out)

        return out, (attn_weights.detach() if need_weights else None)


class TrXLSplitBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, gate_bias=-2.0, use_gating: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = RelativeMultiHeadAttention(d_model, n_heads, dropout=dropout)

        self.ff1  = spectral_norm(nn.Linear(d_model, d_ff))
        self.ff2  = spectral_norm(nn.Linear(d_ff, d_model))
        self.act  = nn.ReLU()
        self.drop = nn.Dropout(dropout)

        self.attn_gate = GatingUnit(d_model, gate_bias, enabled=use_gating)
        self.ff_gate   = GatingUnit(d_model, gate_bias, enabled=use_gating)

    def _check(self, tensor, name):
        if torch.isnan(tensor).any():
            raise RuntimeError(f"NaN detected in TrXLBlock at: {name}")
        if torch.isinf(tensor).any():
            raise RuntimeError(f"Inf detected in TrXLBlock at: {name}")

    def attn_sublayer(self, x, memory=None, rel_pos_emb=None, need_weights=False):
        """
        need_weights=True additionally returns per-head attention weights,
        shape (B, n_heads, 1, memory_len+1) -- only set this during
        diagnostic passes, not during normal rollout/PPO-update forward calls.
        rel_pos_emb: (memory_len+1, D) from RelativeSinusoidalEncoding,
        required -- passed down from TrXLExtractor.forward().
        """
        x_norm      = self.norm1(x)
        self._check(x_norm, "norm1 output")
        kv          = torch.cat([memory, x_norm], dim=1) if memory is not None else x_norm

        attn_out, attn_weights = self.attn(
            x_norm, kv, rel_pos_emb, need_weights=need_weights
        )
        attn_out = torch.clamp(self.drop(attn_out), -10.0, 10.0)
        self._check(attn_out, "attn output")

        gated = self.attn_gate(x, attn_out)
        return gated, attn_weights

    def ff_sublayer(self, x):
        ff_out = self.ff2(self.act(self.ff1(self.norm2(x))))
        ff_out = torch.clamp(ff_out, -10.0, 10.0)
        self._check(ff_out, "ff output")

        gated = self.ff_gate(x, ff_out)
        return gated


class TrXLExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 256,
        memory_len: int = 32,
        n_layers: int = 2,
        n_heads: int = 4,
        d_ff_multiplier: int = 2,
        dropout: float = 0.1,
        gate_bias: float = -2.0,
        cnn_only=False,
        use_gating: bool = True,
        use_spatial_bias: bool = True,
        use_hyperconnections: bool = True,
    ):
        """
        use_gating:
            True  (default) -- GTrXL gated residual combination (original
                  behavior) for both the attention and feed-forward
                  sublayers in every block.
            False -- plain residual connections (x + sublayer_out) instead.
                  No gate parameters are created.
        use_spatial_bias:
            True  (default) -- CNN features are modulated by
                  sigmoid(Linear(pos)) before fusion (original behavior).
            False -- raw CNN features are used unmodified; pos_to_cnn_bias
                  is not constructed at all.
        use_hyperconnections:
            True  (default) -- each block's attn/ff output is combined with
                  a learned softmax-weighted mix of ALL prior hidden states
                  (original behavior).
            False -- each block's attn/ff output passes straight through as
                  the new hidden state (plain residual stream, no cross-
                  layer mixing). No alpha/beta parameters are created.

        All three flags default to True, so TrXLExtractor(...) with no new
        kwargs is functionally and numerically identical to the previous
        version of this class.
        """
        super().__init__(observation_space, features_dim)

        self.memory_len       = memory_len
        self.n_layers         = n_layers
        self.n_heads          = n_heads
        self._d_model         = features_dim
        self.memory           = None
        self._segment_hiddens = None

        self.use_gating           = use_gating
        self.use_spatial_bias     = use_spatial_bias
        self.use_hyperconnections = use_hyperconnections

        n_channels = observation_space["viewport"].shape[0]
        pos_dim    = observation_space["positions"].shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32,         64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64,         64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(64,        128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
        )
        with torch.no_grad():
            cnn_out = self.cnn(
                torch.zeros(1, *observation_space["viewport"].shape)
            ).shape[1]

        self.pos_mlp = nn.Sequential(
            nn.Linear(pos_dim, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128,     128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128,      64),                    nn.ReLU(),
        )

        # Only constructed when spatial biasing is enabled -- disabling the
        # flag removes these parameters entirely rather than just skipping
        # their use in forward().
        if self.use_spatial_bias:
            self.pos_to_cnn_bias = nn.Linear(pos_dim, cnn_out)
        else:
            self.pos_to_cnn_bias = None

        self.fusion = nn.Sequential(
            nn.Linear(cnn_out + 64, self._d_model * 2),
            nn.LayerNorm(self._d_model * 2),
            nn.ReLU(),
            nn.Linear(self._d_model * 2, self._d_model),
            nn.LayerNorm(self._d_model),
            nn.ReLU(),
        )

        # Spatial (world x,y) position encoding -- unrelated to the temporal
        # encoding below or to the use_spatial_bias flag above, this still
        # gets added to the current token's content unconditionally, exactly
        # as before.
        self.token_spatial_encoding = nn.Linear(pos_dim, self._d_model)

        # Relative TEMPORAL positional encoding. Fixed buffer, shared across
        # layers -- each block's RelativeMultiHeadAttention has its own
        # k_pos_proj/u/v to interpret it independently.
        self.rel_pos_encoding = RelativeSinusoidalEncoding(self._d_model, memory_len)

        d_ff = self._d_model * d_ff_multiplier
        self.blocks = nn.ModuleList([
            TrXLSplitBlock(self._d_model, n_heads, d_ff, dropout, gate_bias, use_gating=use_gating)
            for _ in range(n_layers)
        ])

        self.hyper_connections_attn = nn.ModuleList([
            HyperConnection(n_layers, self._d_model, layer_idx=i, enabled=use_hyperconnections)
            for i in range(n_layers)
        ])
        self.hyper_connections_ff = nn.ModuleList([
            HyperConnection(n_layers, self._d_model, layer_idx=i, enabled=use_hyperconnections)
            for i in range(n_layers)
        ])

        self.output_norm = nn.LayerNorm(self._d_model)

    # ── Memory management ─────────────────────────────────────────────────────

    def init_memory(self, batch_size, device):
        self.memory = [
            torch.zeros(batch_size, self.memory_len, self._d_model, device=device)
            for _ in range(self.n_layers)
        ]
        self._segment_hiddens = [
            torch.zeros(batch_size, self.memory_len, self._d_model, device=device)
            for _ in range(self.n_layers)
        ]

    def reset_memory(self, env_indices):
        if self.memory is None:
            return
        for layer_mem in self.memory:
            for idx in env_indices:
                layer_mem[idx] = 0.0
        if self._segment_hiddens is not None:
            for h in self._segment_hiddens:
                for idx in env_indices:
                    h[idx] = 0.0

    def _update_memory(self, new_hiddens):
        # Each layer's sliding window is built from its own previous memory
        # slice (self.memory[i]), matching how _segment_hiddens is indexed.
        new_memory = []
        for i in range(self.n_layers):
            updated = torch.cat([self.memory[i][:, 1:, :], new_hiddens[i].detach()], dim=1)
            new_memory.append(updated)

        self.memory = new_memory

        self._segment_hiddens = [
            torch.cat([self._segment_hiddens[i], new_hiddens[i].detach()], dim=1
            )[:, -self.memory_len:, :]
            for i in range(self.n_layers)
        ]

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        observations,
        memory_override=None,
        return_attn_weights: bool = False,
        return_hidden_states: bool = False,
    ):
        """
        return_attn_weights / return_hidden_states are diagnostic-only flags.
        Leave both False (the default) for normal rollout/PPO-update forward
        passes -- output is then just `out`, unchanged from before.

        When either is True, returns (out, attn_weights, hidden_states),
        where the unrequested item is None. attn_weights is a list of
        n_layers tensors, each (B, n_heads, 1, memory_len+1). hidden_states
        is a list of n_layers+1 tensors, each (B, 1, D) -- index 0 is the
        input embedding, index i is block i's output.
        """
        vp  = observations["viewport"]
        pos = observations["positions"]
        B   = vp.shape[0]

        def _check(tensor, name):
            if torch.isnan(tensor).any():
                raise RuntimeError(f"NaN in TrXLExtractor at: {name}")
            if torch.isinf(tensor).any():
                raise RuntimeError(f"Inf in TrXLExtractor at: {name}")

        cnn_feat = self.cnn(vp)
        if self.use_spatial_bias:
            spatial_bias = self.pos_to_cnn_bias(pos)
            cnn_feat     = cnn_feat * torch.sigmoid(spatial_bias)

        pos_feat = self.pos_mlp(pos)
        current  = self.fusion(torch.cat([cnn_feat, pos_feat], dim=1))
        current  = current + self.token_spatial_encoding(pos)
        current  = current.unsqueeze(1)

        is_update_pass = memory_override is not None
        if is_update_pass:
            active_memory = memory_override
        else:
            if self.memory is None or self.memory[0].shape[0] != B:
                self.init_memory(B, vp.device)
            active_memory = self.memory

        cur_token = current

        # Relative positional embedding -- fixed, recomputed identically
        # every forward call (no dependence on absolute step count, so no
        # drift as tokens age through the memory buffer).
        rel_pos_emb = self.rel_pos_encoding().to(vp.device)  # (memory_len+1, D)

        current_hiddens = []
        x               = cur_token
        hidden_states   = [x]
        layer_attn_weights = [] if return_attn_weights else None

        for i, (block, hyper_attn, hyper_ff) in enumerate(
            zip(self.blocks, self.hyper_connections_attn, self.hyper_connections_ff)
        ):
            mem_input = active_memory[i][:B]

            attn_gated, attn_w = block.attn_sublayer(
                x, mem_input, rel_pos_emb=rel_pos_emb, need_weights=return_attn_weights
            )
            _check(attn_gated, f"attn_gate block {i}")
            if return_attn_weights:
                layer_attn_weights.append(attn_w.detach())

            x_attn = hyper_attn(hidden_states, attn_gated)
            _check(x_attn, f"hyper_attn {i}")

            ff_gated = block.ff_sublayer(x_attn)
            _check(ff_gated, f"ff_gate block {i}")

            x = hyper_ff(hidden_states, ff_gated)
            _check(x, f"hyper_ff {i}")

            hidden_states.append(x)
            current_hiddens.append(x)

        if not is_update_pass:
            self._update_memory(current_hiddens)

        out = self.output_norm(x.squeeze(1))
        _check(out, "output_norm")

        if not return_attn_weights and not return_hidden_states:
            return out

        return out, layer_attn_weights, (hidden_states if return_hidden_states else None)