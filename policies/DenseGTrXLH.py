"""
Gated Transformer XL Hyperconnected Architecture.

Stability changes vs. previous version:
  1. HyperConnections: controller input is LayerNormed, controller output is
     bounded with tanh and multiplied by a small learnable scale.
  2. TrXLExtractor: the residual stream is LayerNormed after every block
     (self.stream_norm[i]) instead of being an unbounded mean of H_ff.
  3. Extra _check calls on H_attn / H_ff so any blow-up is caught where it starts.
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

import numpy as np
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class GatingUnit(nn.Module):
    def __init__(self, d_model: int, gate_bias: float = 0.0, enabled: bool = True):
        super().__init__()
        self.enabled = enabled
        if self.enabled:
            self.gate_linear = nn.Linear(d_model * 2, d_model)
            nn.init.constant_(self.gate_linear.bias, gate_bias)
        else:
            self.gate_linear = None
        self.last_gate = None

    def forward(self, x: torch.Tensor, sublayer_out: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            self.last_gate = None
            return x + sublayer_out

        combined = torch.cat([x, sublayer_out], dim=-1)
        g        = torch.sigmoid(self.gate_linear(combined))
        self.last_gate = g.detach()
        return g * sublayer_out + (1.0 - g) * x


class HyperConnections(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_streams: int = 4,
        dynamic: bool = True,
        enabled: bool = True,
        ctrl_scale_init: float = 0.01,
    ):
        super().__init__()
        self.enabled   = enabled
        self.n_streams = n_streams if enabled else 1
        self.dynamic   = dynamic and enabled and self.n_streams > 1

        n = self.n_streams
        if self.n_streams > 1:
            self.static_alpha = nn.Parameter(torch.full((n,), 1.0 / n))
            self.static_width = nn.Parameter(torch.eye(n))
            self.static_beta  = nn.Parameter(torch.ones(n))

            if self.dynamic:
                # FIX 1: normalize controller input, bound + scale controller output
                self.ctrl_norm  = nn.LayerNorm(d_model)
                self.ctrl_scale = nn.Parameter(torch.full((1,), ctrl_scale_init))
                self.controller = nn.Linear(d_model, n + n * n + n)
                nn.init.zeros_(self.controller.weight)
                nn.init.zeros_(self.controller.bias)
            else:
                self.ctrl_norm  = None
                self.ctrl_scale = None
                self.controller = None
        else:
            self.static_alpha = None
            self.static_width = None
            self.static_beta  = None
            self.ctrl_norm    = None
            self.ctrl_scale   = None
            self.controller   = None

    def init_streams(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_streams == 1:
            return x
        return x.expand(-1, self.n_streams, -1).contiguous()

    def forward(self, H: torch.Tensor, sublayer_fn):
        if self.n_streams == 1:
            y = sublayer_fn(H)
            return y, y

        B, n, D = H.shape

        alpha = self.static_alpha.unsqueeze(0).expand(B, -1)
        width = self.static_width.unsqueeze(0).expand(B, -1, -1)
        beta  = self.static_beta.unsqueeze(0).expand(B, -1)

        if self.controller is not None:
            pooled = self.ctrl_norm(H.mean(dim=1))                       # (B, D), normalized
            ctrl   = torch.tanh(self.controller(pooled)) * self.ctrl_scale  # bounded, small
            d_alpha, d_width, d_beta = torch.split(ctrl, [n, n * n, n], dim=-1)
            alpha = alpha + d_alpha
            width = width + d_width.reshape(B, n, n)
            beta  = beta + d_beta

        x = torch.einsum('bn,bnd->bd', alpha, H).unsqueeze(1)   # (B,1,D) depth-connection
        y = sublayer_fn(x)                                        # (B,1,D) sublayer output

        carried   = torch.einsum('bnm,bnd->bmd', width, H)               # (B,n,D)
        broadcast = torch.einsum('bn,bd->bnd', beta, y.squeeze(1))       # (B,n,D)
        H_new = carried + broadcast
        return H_new, y


class DenseNetMix(nn.Module):
    def __init__(self, d_model: int, layer_idx: int, init_concentration: float = 4.0):
        super().__init__()
        n_candidates = layer_idx + 1  # input embedding + `layer_idx` prior block outputs
        self.n_candidates = n_candidates
        self.raw_weights = nn.Parameter(torch.zeros(n_candidates))
        with torch.no_grad():
            self.raw_weights[-1] = init_concentration

    def forward(self, candidates):
        assert len(candidates) == self.n_candidates, (
            f"DenseNetMix at layer_idx implying {self.n_candidates} candidates "
            f"got {len(candidates)} -- candidate list must be exactly "
            f"[input embedding, block_0_out, ..., block_{{layer_idx-1}}_out]."
        )
        stacked = torch.cat(candidates, dim=1)          # (B, n_candidates, D)
        w       = torch.softmax(self.raw_weights, dim=0)  # (n_candidates,)
        mixed   = torch.einsum('c,bcd->bd', w, stacked).unsqueeze(1)  # (B,1,D)
        return mixed


class RelativeSinusoidalEncoding(nn.Module):
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
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = self.d_head ** -0.5

        self.q_proj         = nn.Linear(d_model, d_model, bias=False)
        self.k_content_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_pos_proj     = nn.Linear(d_model, d_model, bias=False)
        self.v_proj         = nn.Linear(d_model, d_model, bias=False)
        self.out_proj       = nn.Linear(d_model, d_model)

        self.u = nn.Parameter(torch.zeros(n_heads, self.d_head))
        self.v = nn.Parameter(torch.zeros(n_heads, self.d_head))
        nn.init.xavier_uniform_(self.u)
        nn.init.xavier_uniform_(self.v)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x, kv_content, rel_pos_emb, need_weights: bool = False):
        B, L, _ = kv_content.shape
        H, Dh   = self.n_heads, self.d_head

        q   = self.q_proj(x).view(B, 1, H, Dh).permute(0, 2, 1, 3)                   # (B,H,1,Dh)
        k_e = self.k_content_proj(kv_content).view(B, L, H, Dh).permute(0, 2, 1, 3)  # (B,H,L,Dh)
        v   = self.v_proj(kv_content).view(B, L, H, Dh).permute(0, 2, 1, 3)          # (B,H,L,Dh)
        k_r = self.k_pos_proj(rel_pos_emb).view(L, H, Dh).permute(1, 0, 2)           # (H,L,Dh)

        u      = self.u.unsqueeze(0).unsqueeze(2)  # (1,H,1,Dh)
        v_bias = self.v.unsqueeze(0).unsqueeze(2)  # (1,H,1,Dh)

        AC = torch.matmul(q + u, k_e.transpose(-2, -1))          # (B,H,1,L)
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
        x_norm = self.norm1(x)
        self._check(x_norm, "norm1 output")
        kv = torch.cat([memory, x_norm], dim=1) if memory is not None else x_norm

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
        hc_n_streams: int = 4,
        hc_dynamic: bool = True,
        use_densenet_mix: bool = True,
    ):
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
        self.use_densenet_mix     = use_densenet_mix

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

        self.token_spatial_encoding = nn.Linear(pos_dim, self._d_model)

        self.rel_pos_encoding = RelativeSinusoidalEncoding(self._d_model, memory_len)

        d_ff = self._d_model * d_ff_multiplier
        self.blocks = nn.ModuleList([
            TrXLSplitBlock(self._d_model, n_heads, d_ff, dropout, gate_bias, use_gating=use_gating)
            for _ in range(n_layers)
        ])

        self.hyper_connections_attn = nn.ModuleList([
            HyperConnections(self._d_model, n_streams=hc_n_streams, dynamic=hc_dynamic,
                             enabled=use_hyperconnections)
            for _ in range(n_layers)
        ])
        self.hyper_connections_ff = nn.ModuleList([
            HyperConnections(self._d_model, n_streams=hc_n_streams, dynamic=hc_dynamic,
                             enabled=use_hyperconnections)
            for _ in range(n_layers)
        ])

        # FIX 2: normalize the residual stream between blocks
        self.stream_norm = nn.ModuleList([
            nn.LayerNorm(self._d_model) for _ in range(n_layers)
        ])

        if self.use_densenet_mix:
            self.densenet_mix = nn.ModuleList([
                DenseNetMix(self._d_model, layer_idx=i) for i in range(n_layers)
            ])
        else:
            self.densenet_mix = None

        self.output_norm = nn.LayerNorm(self._d_model)

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

    def forward(
        self,
        observations,
        memory_override=None,
        return_attn_weights: bool = False,
        return_hidden_states: bool = False,
    ):
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
        current  = current.unsqueeze(1)  # (B,1,D)

        is_update_pass = memory_override is not None
        if is_update_pass:
            active_memory = memory_override
        else:
            if self.memory is None or self.memory[0].shape[0] != B:
                self.init_memory(B, vp.device)
            active_memory = self.memory

        rel_pos_emb = self.rel_pos_encoding().to(vp.device)  # (memory_len+1, D)

        current_hiddens    = []
        hidden_states      = [current]
        layer_attn_weights = [] if return_attn_weights else None

        x_stream = current  # (B,1,D) single-stream carrier between blocks

        for i, (block, hyper_attn, hyper_ff) in enumerate(
            zip(self.blocks, self.hyper_connections_attn, self.hyper_connections_ff)
        ):
            if self.use_densenet_mix:
                block_input = self.densenet_mix[i](hidden_states)
            else:
                block_input = x_stream

            mem_input  = active_memory[i][:B]
            attn_w_box = {}

            def _attn_fn(xin, _block=block, _mem=mem_input, _rp=rel_pos_emb, _box=attn_w_box):
                gated, w = _block.attn_sublayer(xin, _mem, rel_pos_emb=_rp, need_weights=return_attn_weights)
                _box["w"] = w
                return gated

            H_attn = hyper_attn.init_streams(block_input)
            H_attn, y_attn = hyper_attn(H_attn, _attn_fn)
            _check(H_attn, f"H_attn (hyper-connection) block {i}")
            _check(y_attn, f"attn output block {i}")
            if return_attn_weights:
                layer_attn_weights.append(attn_w_box["w"].detach())

            H_ff = hyper_ff.init_streams(y_attn)
            H_ff, y_ff = hyper_ff(H_ff, block.ff_sublayer)
            _check(H_ff, f"H_ff (hyper-connection) block {i}")
            _check(y_ff, f"ff output block {i}")

            # FIX 2: bound the residual stream before it feeds the next block / output
            x_stream = self.stream_norm[i](H_ff.mean(dim=1, keepdim=True))  # (B,1,D)
            _check(x_stream, f"stream_norm block {i}")

            hidden_states.append(x_stream)
            current_hiddens.append(y_ff)  # (B,1,D) -- written into temporal memory

        if not is_update_pass:
            self._update_memory(current_hiddens)

        out = self.output_norm(x_stream.squeeze(1))
        out = torch.clamp(out, -10.0, 10.0)
        _check(out, "output_norm")

        if not return_attn_weights and not return_hidden_states:
            return out

        return out, layer_attn_weights, (hidden_states if return_hidden_states else None)