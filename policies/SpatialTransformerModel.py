import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor






class SpatialTransformerExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256, n_heads=4, n_layers=2):
        super().__init__(observation_space, features_dim)

        n_channels = observation_space["viewport"].shape[0]  # 2
        pos_dim = observation_space["positions"].shape[0]

        # CNN stem — converts (2, 84, 84) into a sequence of patch tokens
        self.patch_embed = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=8, stride=4),  # → (64, 20, 20)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),          # → (128, 9, 9)
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),         # → (128, 7, 7)
            nn.ReLU(),
        )
        # Each spatial location (7x7 = 49) becomes a token of dim 128
        self.token_dim = 128
        self.n_tokens = 7 * 7  # 49 patch tokens

        # Learnable positional encoding for the 7x7 grid
        self.pos_encoding = nn.Parameter(torch.randn(1, self.n_tokens, self.token_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.token_dim,
            nhead=n_heads,
            dim_feedforward=256,
            dropout=0.0,       # no dropout during RL — it hurts value estimation
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Position MLP (agent coordinates)
        self.pos_mlp = nn.Sequential(
            nn.Linear(pos_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Fusion: mean-pool transformer output + position features → features_dim
        self.fusion = nn.Sequential(
            nn.Linear(self.token_dim + 64, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        viewport = observations["viewport"]    # (B, 2, 84, 84)
        positions = observations["positions"]  # (B, n_agents*2)

        # Patch tokens: (B, 128, 7, 7) → (B, 49, 128)
        x = self.patch_embed(viewport)
        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)  # (B, 49, 128)
        x = x + self.pos_encoding           # add positional encoding

        # Transformer: each patch attends to all others
        x = self.transformer(x)             # (B, 49, 128)

        # Global average pool over tokens → (B, 128)
        x = x.mean(dim=1)

        pos_feat = self.pos_mlp(positions)  # (B, 64)

        return self.fusion(torch.cat([x, pos_feat], dim=1))  # (B, 256)