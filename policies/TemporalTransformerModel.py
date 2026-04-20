

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


from collections import deque




class TransformerExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256, seq_len=16):
        super().__init__(observation_space, features_dim)

        self.seq_len = seq_len
        self.d_model = 256

        # CNN for viewport
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.cnn_fc = nn.Linear(3136, 256)

        # positions encoder
        self.pos_mlp = nn.Sequential(
            nn.Linear(observation_space["positions"].shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )

        # projection
        self.input_proj = nn.Linear(256 + 128, self.d_model)

        # positional encoding
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, self.d_model))

        # transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=4,
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        # gating (stability)
        self.gate = nn.Parameter(torch.tensor(0.5))

        # runtime buffer (NOT part of model params)
        self._buffers = None

    def reset_buffer(self, n_envs):
        self._buffers = [
            deque(maxlen=self.seq_len) for _ in range(n_envs)
        ]

    def _encode_single(self, obs):
        vp = obs["viewport"]
        pos = obs["positions"]

        vp_emb = self.cnn_fc(self.cnn(vp))
        pos_emb = self.pos_mlp(pos)

        return torch.cat([vp_emb, pos_emb], dim=-1)

    def forward(self, observations):
        """
        observations:
            dict of tensors (B, ...)
        """

        B = observations["viewport"].shape[0]

        if self._buffers is None:
            self.reset_buffer(B)

        tokens = []

        for i in range(B):
            obs_i = {
                "viewport": observations["viewport"][i].unsqueeze(0),
                "positions": observations["positions"][i].unsqueeze(0)
            }

            token = self._encode_single(obs_i).squeeze(0)

            self._buffers[i].append(token.detach())  # store without grad

            seq = list(self._buffers[i])

            # pad sequence
            while len(seq) < self.seq_len:
                seq.insert(0, seq[0])

            seq = torch.stack(seq, dim=0)  # (T, D)
            tokens.append(seq)

        x = torch.stack(tokens, dim=0)  # (B, T, D)

        x = self.input_proj(x)
        x = x + self.pos_embedding[:, :x.size(1)]

        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()

        x_trans = self.transformer(x, mask=mask)

        x = self.gate * x_trans + (1 - self.gate) * x

        return x[:, -1]  # last token




class TemporalTransformerExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256, n_heads=4,
                 n_layers=2, memory_len=8, n_envs=1):
        super().__init__(observation_space, features_dim)
        self.memory_len = memory_len
        self.token_dim  = 128
        self.n_envs     = n_envs

        n_channels = observation_space["viewport"].shape[0]
        pos_dim    = observation_space["positions"].shape[0]

        self.patch_embed = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(64, 128,    kernel_size=4, stride=2),     nn.ReLU(),
            nn.Conv2d(128, self.token_dim, kernel_size=3, stride=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        self.time_pos = nn.Parameter(
            torch.randn(1, memory_len, self.token_dim) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.token_dim, nhead=n_heads,
            dim_feedforward=256, dropout=0.0, batch_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.pos_mlp = nn.Sequential(
            nn.Linear(pos_dim, 64), nn.ReLU(),
            nn.Linear(64, 64),     nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.token_dim + 64, features_dim), nn.ReLU(),
        )

        # One memory slot per env
        self.register_buffer(
            "memory", torch.zeros(n_envs, memory_len, self.token_dim)
        )

    def reset_memory(self, env_indices=None):
        """Reset memory for specific envs (or all if env_indices is None)."""
        if env_indices is None:
            self.memory.zero_()
        else:
            for idx in env_indices:
                self.memory[idx].zero_()

    def forward(self, observations):
        viewport  = observations["viewport"]    # (B, C, 84, 84)
        positions = observations["positions"]   # (B, pos_dim)
        B = viewport.shape[0]

        current_token = self.patch_embed(viewport)  # (B, token_dim)

        # Use the stored memory for the current batch
        # During rollout B == n_envs; during update B == batch_size (memory is ignored)
        if B == self.n_envs:
            mem = self.memory.clone()
        else:
            # During PPO update with large batch — no meaningful per-env memory
            # Use a zeroed buffer, attention still runs over the sequence
            mem = torch.zeros(B, self.memory_len, self.token_dim,
                              device=viewport.device)

        mem = torch.roll(mem, -1, dims=1)
        mem[:, -1, :] = current_token

        if B == self.n_envs:
            self.memory = mem.detach()

        mem = mem + self.time_pos.expand(B, -1, -1)
        attended = self.temporal_transformer(mem)
        summary  = attended[:, -1, :]

        pos_feat = self.pos_mlp(positions)
        return self.fusion(torch.cat([summary, pos_feat], dim=1))