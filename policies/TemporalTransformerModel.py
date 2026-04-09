


import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor




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