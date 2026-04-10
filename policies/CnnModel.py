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
            nn.Conv2d(n_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32,         64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64,         64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample = torch.zeros(1, *observation_space["viewport"].shape)
            cnn_out_dim = self.cnn(sample).shape[1]

        self.pos_mlp = nn.Sequential(
            nn.Linear(pos_dim, 64), nn.ReLU(),
            nn.Linear(64, 64),      nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(cnn_out_dim + 64, features_dim), nn.ReLU(),
        )

    def forward(self, observations):
        vp  = observations["viewport"]
        pos = observations["positions"]
        return self.fusion(
            torch.cat([self.cnn(vp), self.pos_mlp(pos)], dim=1)
        )
