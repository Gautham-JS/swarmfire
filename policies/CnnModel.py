import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym


class PlainCNNExtractor(BaseFeaturesExtractor):
    """
    Stateless CNN+MLP extractor for use with RecurrentPPO.
    RecurrentPPO's own LSTM sits on top and handles all temporal memory.
    No rolling buffer needed here.
    """
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 256,
    ):
        super().__init__(observation_space, features_dim)

        n_channels = observation_space["viewport"].shape[0]   # 3
        pos_dim    = observation_space["positions"].shape[0]  # n_agents * 7

        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=8, stride=4, padding=0), nn.ReLU(),  # → 20x20
            nn.Conv2d(32,         64, kernel_size=4, stride=2, padding=0), nn.ReLU(),  # → 9x9
            nn.Conv2d(64,         64, kernel_size=3, stride=1, padding=1), nn.ReLU(),  # → 9x9 (padding keeps size)
            nn.Conv2d(64,        128, kernel_size=3, stride=1, padding=1), nn.ReLU(),  # → 9x9 (extra capacity)
            nn.AdaptiveAvgPool2d(4),  # → 4x4, handles any input size cleanly
            nn.Flatten(),             # → 128*16 = 2048
        )

        with torch.no_grad():
            sample = torch.zeros(1, *observation_space["viewport"].shape)
            cnn_out_dim = self.cnn(sample).shape[1]

        self.pos_mlp = nn.Sequential(
            nn.Linear(pos_dim, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 128),     nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 64),                         nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(cnn_out_dim + 64, features_dim * 2), nn.LayerNorm(features_dim * 2), nn.ReLU(),
            nn.Linear(features_dim * 2, features_dim),     nn.LayerNorm(features_dim),     nn.ReLU(),
        )

        self.pos_to_cnn_bias = nn.Linear(pos_dim, cnn_out_dim)

    def forward(self, observations):
        vp  = observations["viewport"]
        pos = observations["positions"]

        cnn_feat = self.cnn(vp)
        pos_feat = self.pos_mlp(pos)
        
        spatial_bias = self.pos_to_cnn_bias(pos)          # positional context
        gated = cnn_feat * torch.sigmoid(spatial_bias)    # soft gating [TODO : Write about the effects of gating]
        
        return self.fusion(torch.cat([gated, pos_feat], dim=1))
