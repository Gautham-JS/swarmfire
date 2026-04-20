import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

import numpy as np
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TrXLBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(
            d_model, n_heads,
            dropout=dropout,
            batch_first=True,
        )
        # Spectral norm on FF layers — constrains weight growth
        self.ff1  = spectral_norm(nn.Linear(d_model, d_ff))
        self.ff2  = spectral_norm(nn.Linear(d_ff, d_model))
        self.act  = nn.ReLU()
        self.drop = nn.Dropout(dropout)
    
    def _check(self, tensor, name):
        if torch.isnan(tensor).any():
            raise RuntimeError(f"NaN detected in TrXLBlock at: {name}")
        if torch.isinf(tensor).any():
            raise RuntimeError(f"Inf detected in TrXLBlock at: {name}")

    def forward(self, x, memory=None):
        x_norm      = self.norm1(x)
        self._check(x_norm, "X norm1 output")
        kv          = torch.cat([memory, x_norm], dim=1) if memory is not None else x_norm

        attn_out, _ = self.attn(query=x_norm, key=kv, value=kv)
        attn_out    = torch.clamp(attn_out, -10.0, 10.0)
        self._check(attn_out, "Attention output")
        x           = x + self.drop(attn_out)

        ff_out      = self.ff2(self.act(self.ff1(self.norm2(x))))
        self._check(x, "X first FF layer")
        ff_out      = torch.clamp(ff_out, -10.0, 10.0)
        x           = x + self.drop(ff_out)
        self._check(x, "X dropout layer")
        return x


class SinusoidalTemporalEncoding(nn.Module):
    def __init__(self, features_dim, memory_len):
        super().__init__()
        # +1 to account for the current token appended during forward
        pe        = torch.zeros(memory_len + 1, features_dim)
        positions = torch.arange(memory_len + 1).unsqueeze(1).float()
        div_term  = torch.exp(
            torch.arange(0, features_dim, 2).float()
            * -(np.log(10000.0) / features_dim)
        )
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        # Not a trainable parameter, but moves with .to(device)
        self.register_buffer('pe', pe)

    def forward(self, tokens):
        # tokens: (B, memory_len+1, D)
        return tokens + self.pe.unsqueeze(0)


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
    ):
        super().__init__(observation_space, features_dim)

        self.memory_len = memory_len
        self.n_layers   = n_layers
        self._d_model   = features_dim   # renamed to avoid clash with SB3 property
        self.memory     = None           # initialized lazily on first forward pass

        n_channels = observation_space["viewport"].shape[0]
        pos_dim    = observation_space["positions"].shape[0]

        # ── CNN encoder (84x84) ───────────────────────────────────────────
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

        # ── Position / velocity MLP ───────────────────────────────────────
        self.pos_mlp = nn.Sequential(
            nn.Linear(pos_dim, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128,     128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128,      64),                    nn.ReLU(),
        )

        # ── CNN spatial bias ──────────────────────────────────────────────
        # Softly gates CNN features based on world position before fusion,
        # telling the CNN where in the 512x512 world it is looking
        self.pos_to_cnn_bias = nn.Linear(pos_dim, cnn_out)

        # ── Fusion ────────────────────────────────────────────────────────
        self.fusion = nn.Sequential(
            nn.Linear(cnn_out + 64, self._d_model * 2),
            nn.LayerNorm(self._d_model * 2), nn.ReLU(),
            nn.Linear(self._d_model * 2, self._d_model),
            nn.LayerNorm(self._d_model),     nn.ReLU(),
        )

        # ── Spatial token encoding ────────────────────────────────────────
        # Tags each memory token with its world XY position so the
        # transformer can learn to retrieve spatially relevant memories
        self.token_spatial_encoding = nn.Linear(pos_dim, self._d_model)

        # ── Sinusoidal temporal encoding ──────────────────────────────────
        # Tags tokens with recency: index 0 = oldest, index memory_len = current
        self.temporal_encoding = SinusoidalTemporalEncoding(self._d_model, memory_len)

        # ── Transformer-XL blocks ─────────────────────────────────────────
        d_ff = self._d_model * d_ff_multiplier
        self.blocks = nn.ModuleList([
            TrXLBlock(self._d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.output_norm = nn.LayerNorm(self._d_model)

    # ── Memory management ─────────────────────────────────────────────────

    def init_memory(self, batch_size, device):
        """Initialise per-layer memory caches with zeros."""
        self.memory = [
            torch.zeros(batch_size, self.memory_len, self._d_model, device=device)
            for _ in range(self.n_layers)
        ]

    def reset_memory(self, env_indices):
        """
        Zero out memory for specific envs on episode done.
        Called by MemoryResetCallback.
        """
        if self.memory is None:
            return
        for layer_mem in self.memory:
            for idx in env_indices:
                layer_mem[idx] = 0.0

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, observations):
        vp  = observations["viewport"]
        pos = observations["positions"]
        B   = vp.shape[0]

        def _check(tensor, name):
            if torch.isnan(tensor).any():
                raise RuntimeError(f"NaN detected in TrXLExtractor at: {name}")
            if torch.isinf(tensor).any():
                raise RuntimeError(f"Inf detected in TrXLExtractor at: {name}")

        # 1. CNN with spatial gating
        cnn_feat     = self.cnn(vp)
        _check(cnn_feat, "cnn_feat")

        spatial_bias = self.pos_to_cnn_bias(pos)
        cnn_feat     = cnn_feat * torch.sigmoid(spatial_bias)
        _check(cnn_feat, "cnn_feat after spatial gate")

        # 2. Position MLP
        pos_feat = self.pos_mlp(pos)
        _check(pos_feat, "pos_feat")

        # 3. Fusion
        current = self.fusion(torch.cat([cnn_feat, pos_feat], dim=1))
        _check(current, "fusion output")

        # 4. Spatial encoding
        current = current + self.token_spatial_encoding(pos)
        _check(current, "after spatial encoding")
        current = current.unsqueeze(1)

        # 5. Memory logic (unchanged from before)
        use_memory = (
            self.memory is not None
            and self.memory[0].shape[0] == B
        )
        if self.memory is None:
            self.init_memory(B, vp.device)
            use_memory = True

        if use_memory:
            tokens     = torch.cat([self.memory[0][:B], current], dim=1)
            tokens     = self.temporal_encoding(tokens)
            _check(tokens, "after temporal encoding")
            mem_tokens = tokens[:, :-1, :]
            cur_token  = tokens[:, -1:, :]
        else:
            zero_mem   = torch.zeros(B, self.memory_len, self._d_model, device=vp.device)
            tokens     = self.temporal_encoding(torch.cat([zero_mem, current], dim=1))
            mem_tokens = None
            cur_token  = tokens[:, -1:, :]

        # 6. TrXL blocks
        new_memory = []
        x = cur_token
        for i, block in enumerate(self.blocks):
            mem_input = (mem_tokens if i == 0 else self.memory[i][:B]) if use_memory else None
            x = block(x, mem_input)
            _check(x, f"TrXLBlock {i} output")

            if use_memory:
                updated = torch.cat(
                    [self.memory[i][:B], x.detach()], dim=1
                )[:, -self.memory_len:, :]
                new_memory.append(updated)

        if use_memory:
            self.memory = new_memory

        out = self.output_norm(x.squeeze(1))
        _check(out, "output_norm")
        return out