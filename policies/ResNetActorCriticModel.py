import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy


class PositionEncoder(nn.Module):
    def __init__(self, n_agents: int, out_dim: int = 64):
        super().__init__()
        self.n_agents = n_agents
        # input: raw positions (n_agents*2,) + pairwise deltas (n_agents*(n_agents-1),)
        in_dim = n_agents * 2 + n_agents * (n_agents - 1)
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
            nn.ReLU()
        )

    def forward(self, pos):
        # pos: (B, n_agents*2)
        coords = pos.view(-1, self.n_agents, 2)       # (B, N, 2)

        # pairwise relative positions — captures inter-agent geometry
        deltas = []
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if i != j:
                    deltas.append(coords[:, i] - coords[:, j])  # (B, 2)

        if deltas:
            deltas = torch.cat(deltas, dim=-1)        # (B, N*(N-1)*2) -- wait, each delta is 2D
            # fix: each delta is (B,2), so cat across dim=1 gives (B, N*(N-1)*2)
            # but we want scalar distances too
            x = torch.cat([pos, deltas], dim=-1)
        else:
            x = pos

        return self.net(x)

class FireScoutCNN(nn.Module):
    """Spatial encoder for (C, 84, 84) float map."""
    def __init__(self, in_channels: int, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            # wide first kernel to capture large-scale structure
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.LayerNorm([32, 42, 42]),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.LayerNorm([64, 21, 21]),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm([128, 11, 11]),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((4, 4)),  # → (128, 4, 4)
            nn.Flatten(),                  # → 2048
        )
        self.proj = nn.Sequential(
            nn.Linear(128 * 4 * 4, out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.proj(self.net(x))
    


class FireScoutExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, n_agents: int = 1,
                 cnn_out_dim: int = 256, pos_out_dim: int = 64):
        
        features_dim = cnn_out_dim + pos_out_dim
        super().__init__(observation_space, features_dim=features_dim)

        vp_shape = observation_space["viewport"].shape   # (C, 84, 84)
        in_channels = vp_shape[0]

        self.cnn = FireScoutCNN(in_channels, out_dim=cnn_out_dim)
        self.pos_enc = PositionEncoder(n_agents, out_dim=pos_out_dim)

    def forward(self, obs):
        vp_feat  = self.cnn(obs["viewport"])        # (B, cnn_out_dim)
        pos_feat = self.pos_enc(obs["positions"])   # (B, pos_out_dim)
        return torch.cat([vp_feat, pos_feat], dim=-1)
    


class FireScoutPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            # separate nets for actor and critic
            net_arch=dict(
                pi=[256, 128],   # policy head
                vf=[512, 256]    # value head — wider, better for credit assignment
            )
        )