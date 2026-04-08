


import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor





class TemporalTransformerExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256, n_heads=4,
                 n_layers=2, memory_len=8):
        super().__init__(observation_space, features_dim)
        self.memory_len = memory_len
        self.token_dim = 128

        n_channels = observation_space["viewport"].shape[0]
        pos_dim = observation_space["positions"].shape[0]

        # Spatial CNN stem — same as before
        self.patch_embed = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, self.token_dim, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),   # collapse spatial → single token per frame
            nn.Flatten(),              # (B, token_dim)
        )

        # Learnable positional encoding over the time window
        self.time_pos = nn.Parameter(
            torch.randn(1, memory_len, self.token_dim) * 0.02
        )

        # Temporal transformer — attends across the memory window
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.token_dim,
            nhead=n_heads,
            dim_feedforward=256,
            dropout=0.0,
            batch_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        self.pos_mlp = nn.Sequential(
            nn.Linear(pos_dim, 64), nn.ReLU(),
            nn.Linear(64, 64),     nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(self.token_dim + 64, features_dim),
            nn.ReLU(),
        )

        # Rolling memory buffer — persists across steps within an episode
        # Shape: (1, memory_len, token_dim) — batch dim kept for easy concat
        self.register_buffer(
            "memory", torch.zeros(1, memory_len, self.token_dim)
        )

    def reset_memory(self):
        """Call this at the start of each episode."""
        self.memory.zero_()

    def forward(self, observations):
        viewport   = observations["viewport"]    # (B, 2, 84, 84)
        positions  = observations["positions"]   # (B, n_agents*2)
        B = viewport.shape[0]

        # Encode current frame → (B, token_dim)
        current_token = self.patch_embed(viewport)

        # Expand memory to batch size, shift left, append current token
        mem = self.memory.expand(B, -1, -1).clone()   # (B, memory_len, token_dim)
        mem = torch.roll(mem, -1, dims=1)
        mem[:, -1, :] = current_token

        # Store back (only from first batch item — works for DummyVecEnv)
        self.memory = mem[:1].detach()

        # Temporal attention over the window
        mem = mem + self.time_pos.expand(B, -1, -1)
        attended = self.temporal_transformer(mem)   # (B, memory_len, token_dim)

        # Use only the most recent token's output as the summary
        summary = attended[:, -1, :]                # (B, token_dim)

        pos_feat = self.pos_mlp(positions)
        return self.fusion(torch.cat([summary, pos_feat], dim=1))