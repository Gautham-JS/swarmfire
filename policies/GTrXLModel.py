"""
GTrXLFeaturesExtractor -- swaps the custom TrXLExtractor for OpenDILab
DI-engine's GTrXL (https://opendilab.github.io/DI-engine/12_policies/gtrxl.html),
while keeping the exact interface the rest of the training script (and the
memory diagnostics module) already relies on:

    extractor.memory                        -> list[Tensor(bs, memory_len, D)] or None
    extractor.memory = [...]                -> setter (used to detach after PPO update)
    extractor.init_memory(batch_size, device)
    extractor.reset_memory(env_indices)      -> zero out specific envs' memory slice
    extractor(obs)                           -> features                         (default)
    extractor(obs, memory_override=...)      -> features, using a *frozen* snapshot
                                                 instead of live rolling memory
                                                 (used during PPO minibatch replay)
    extractor(obs, return_attn_weights=True,
                   return_hidden_states=True) -> (features, attn_weights, hidden_states)

Install: pip install DI-engine   (imported as `ding`)

--------------------------------------------------------------------------
Why the subclassing below exists
--------------------------------------------------------------------------
Stock `ding.torch_utils.network.gtrxl.GTrXL`:
  1. Always reads/writes `self.memory` internally -- there is no way to run
     a forward pass against an arbitrary memory snapshot without mutating
     the live rolling memory. Your PPO update loop (replaying stored
     per-(step,env) memory snapshots in shuffled minibatches) and the CKA
     memory-ablation diagnostic both need exactly that. `DiagGTrXL` below
     adds an optional `memory_override` argument that is entirely read-only
     with respect to `self.memory`.
  2. Doesn't expose attention weights or GRU-gate activations -- both are
     computed internally and thrown away. `DiagAttentionXL` and
     `DiagGRUGatingUnit` are thin subclasses that stash the post-softmax
     attention tensor / gate tensor as an instance attribute as a side
     effect of the (otherwise unmodified) forward computation, and
     `DiagGTrXL.forward` can optionally collect and return them.

Everything else (the actual math) is untouched, copy-for-copy, from
DI-engine's implementation -- these are additive hooks, not a reimplementation.
"""

import torch
import torch.nn as nn

from ding.torch_utils.network.gtrxl import (
    GTrXL,
    GatedTransformerXLLayer,
    AttentionXL,
    GRUGatingUnit,
    Memory,
)
from ding.torch_utils.network.nn_module import fc_block, build_normalization


# 
# Diagnostic-hook subclasses (math unchanged from DI-engine, just add
# instance-attribute side effects so we can read internal state afterwards)
# 

class DiagAttentionXL(AttentionXL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_attn = None  # (bs, head_num, cur_seq, full_seq), post-softmax, pre-dropout

    def forward(self, inputs, pos_embedding, full_input, u, v, mask=None):
        bs, cur_seq, full_seq = inputs.shape[1], inputs.shape[0], full_input.shape[0]
        prev_seq = full_seq - cur_seq

        kv = self.attention_kv(full_input)
        key, value = torch.chunk(kv, 2, dim=-1)
        query = self.attention_q(inputs)
        r = self.project_pos(pos_embedding)

        key = key.view(full_seq, bs, self.head_num, self.head_dim)
        query = query.view(cur_seq, bs, self.head_num, self.head_dim)
        value = value.view(cur_seq + prev_seq, bs, self.head_num, self.head_dim)
        r = r.view(full_seq, self.head_num, self.head_dim)

        q_u = query + u
        content_attn = q_u.permute(1, 2, 0, 3) @ key.permute(1, 2, 3, 0)

        q_v = query + v
        position_attn = q_v.permute(1, 2, 0, 3) @ r.permute(1, 2, 0)
        position_attn = self._rel_shift(position_attn)

        attn = content_attn + position_attn
        attn.mul_(self.scale)

        if mask is not None and mask.any().item():
            mask_ = mask.permute(2, 0, 1).unsqueeze(1)
            attn = attn.masked_fill(mask_, -float("inf")).type_as(attn)

        attn = torch.softmax(attn, dim=-1)
        # --- diagnostics hook -------------------------------------------------
        self.last_attn = attn.detach()
        # -----------------------------------------------------------------------

        attn = self.dropout(attn)
        attn_vec = attn @ value.permute(1, 2, 0, 3)
        attn_vec = attn_vec.permute(2, 0, 1, 3)
        attn_vec = attn_vec.contiguous().view(cur_seq, bs, self.head_num * self.head_dim)
        output = self.dropout(self.project(attn_vec))
        return output


class DiagGRUGatingUnit(GRUGatingUnit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_gate = None  # z: 0 -> keep old (identity), 1 -> take new content

    def forward(self, x, y):
        r = self.sigmoid(self.Wr(y) + self.Ur(x))
        z = self.sigmoid(self.Wz(y) + self.Uz(x) - self.bg)
        h = self.tanh(self.Wg(y) + self.Ug(torch.mul(r, x)))
        g = torch.mul(1 - z, x) + torch.mul(z, h)
        # --- diagnostics hook -------------------------------------------------
        self.last_gate = z.detach()
        # -----------------------------------------------------------------------
        return g


class DiagGatedTransformerXLLayer(GatedTransformerXLLayer):
    """
    Identical to GatedTransformerXLLayer, except it's built from the Diag*
    sub-modules above. `forward` is inherited unchanged -- it calls
    `self.attention(...)` / `self.gate1(...)` / `self.gate2(...)`, which
    Python dispatches to the Diag versions via normal polymorphism, so no
    override is needed there.
    """

    def __init__(
        self, input_dim, head_dim, hidden_dim, head_num, mlp_num, dropout, activation,
        gru_gating: bool = True, gru_bias: float = 2.,
    ):
        nn.Module.__init__(self)
        self.dropout = dropout
        self.gating = gru_gating
        if self.gating:
            self.gate1 = DiagGRUGatingUnit(input_dim, gru_bias)
            self.gate2 = DiagGRUGatingUnit(input_dim, gru_bias)
        self.attention = DiagAttentionXL(input_dim, head_dim, head_num, dropout)
        layers = []
        dims = [input_dim] + [hidden_dim] * (mlp_num - 1) + [input_dim]
        for i in range(mlp_num):
            layers.append(fc_block(dims[i], dims[i + 1], activation=activation))
            if i != mlp_num - 1:
                layers.append(self.dropout)
        layers.append(self.dropout)
        self.mlp = nn.Sequential(*layers)
        self.layernorm1 = build_normalization('LN')(input_dim)
        self.layernorm2 = build_normalization('LN')(input_dim)
        self.activation = activation


class DiagGTrXL(GTrXL):
    """
    Same as GTrXL, built from DiagGatedTransformerXLLayer blocks, with a
    forward() that additionally supports:
      - memory_override: read-only forward against a supplied memory
        snapshot; self.memory is left completely untouched.
      - return_attn_weights / return_hidden_states: collect diagnostics
        computed during this forward call for free (no extra passes).
    """

    def __init__(
        self,
        input_dim: int,
        head_dim: int = 128,
        embedding_dim: int = 256,
        head_num: int = 2,
        mlp_num: int = 2,
        layer_num: int = 3,
        memory_len: int = 64,
        dropout_ratio: float = 0.,
        activation: nn.Module = nn.ReLU(),
        gru_gating: bool = True,
        gru_bias: float = 2.,
        use_embedding_layer: bool = True,
    ) -> None:
        nn.Module.__init__(self)
        assert embedding_dim % 2 == 0, f'embedding_dim={embedding_dim} should be even'
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        if isinstance(input_dim, list):
            import numpy as np
            input_dim = int(np.prod(input_dim))
        self.use_embedding_layer = use_embedding_layer
        if use_embedding_layer:
            self.embedding = fc_block(input_dim, embedding_dim, activation=activation)
        self.activation = activation
        from ding.torch_utils.network.gtrxl import PositionalEmbedding
        self.pos_embedding = PositionalEmbedding(embedding_dim)
        self.memory = None
        self.memory_len = memory_len
        layers = []
        dims = [embedding_dim] + [embedding_dim] * layer_num
        self.dropout = nn.Dropout(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        for i in range(layer_num):
            layers.append(
                DiagGatedTransformerXLLayer(
                    dims[i], head_dim, embedding_dim, head_num, mlp_num, self.dropout, self.activation,
                    gru_gating, gru_bias,
                )
            )
        self.layers = nn.Sequential(*layers)
        self.embedding_dim = embedding_dim
        self.u, self.v = (
            torch.nn.Parameter(torch.zeros(self.head_num, self.head_dim)),
            torch.nn.Parameter(torch.zeros(self.head_num, self.head_dim)),
        )
        self.att_mask = {}
        self.pos_embedding_dict = {}

    def forward(
        self,
        x: torch.Tensor,
        batch_first: bool = False,
        return_mem: bool = True,
        memory_override: "torch.Tensor | None" = None,
        return_attn_weights: bool = False,
        return_hidden_states: bool = False,
    ):
        if batch_first:
            x = torch.transpose(x, 1, 0)
        cur_seq, bs = x.shape[:2]

        using_override = memory_override is not None
        if using_override:
            # Read-only: build a throwaway Memory wrapper around the given
            # snapshot. self.memory is never touched below.
            mem_obj = Memory(
                memory_len=self.memory_len, layer_num=self.layer_num,
                embedding_dim=self.embedding_dim, memory=memory_override.to(x.device),
            )
        else:
            memory = None if self.memory is None else self.memory.get()
            if memory is None or memory.shape[-2] != bs or memory.shape[-1] != self.embedding_dim:
                self.reset_memory(bs)
            self.memory.to(x.device)
            mem_obj = self.memory

        memory = mem_obj.get()

        if self.use_embedding_layer:
            x = self.dropout(self.embedding(x))
        prev_seq = self.memory_len
        full_seq = cur_seq + prev_seq

        if cur_seq in self.att_mask.keys():
            attn_mask = self.att_mask[cur_seq]
        else:
            attn_mask = (
                torch.triu(torch.ones((cur_seq, full_seq)), diagonal=1 + prev_seq)
                .bool().unsqueeze(-1).to(x.device)
            )
            self.att_mask[cur_seq] = attn_mask

        if cur_seq in self.pos_embedding_dict.keys():
            pos_embedding = self.pos_embedding_dict[cur_seq]
        else:
            pos_ips = torch.arange(full_seq - 1, -1, -1.0, dtype=torch.float)
            pos_embedding = self.pos_embedding(pos_ips.to(x.device))
            self.pos_embedding_dict[cur_seq] = pos_embedding
        pos_embedding = self.dropout(pos_embedding)

        # `hidden_state` is only needed to (a) commit the live memory update
        # (Memory.update() consumes it) or (b) satisfy return_hidden_states.
        # During PPO minibatch replay (using_override=True,
        # return_hidden_states=False -- the hot path, called n_epochs *
        # n_minibatches times per rollout) neither applies, so skip building
        # it entirely: the old unconditional `hidden_state.append(out.clone())`
        # was cloning every layer's full activation on every single minibatch
        # forward for no reason, which is exactly the kind of extra
        # short-lived allocation that fragments the CUDA allocator and can
        # tip you into OOM well before you're actually out of memory.
        # `.clone()` was also redundant even when the list is needed: `out`
        # is rebound to a new tensor object each iteration, never mutated
        # in place, so appending the reference is sufficient.
        need_hidden = return_hidden_states or not using_override
        hidden_state = [x] if need_hidden else None
        attn_weights = []
        out = x
        for i in range(self.layer_num):
            layer = self.layers[i]
            out = layer(out, pos_embedding, self.u, self.v, mask=attn_mask, memory=memory[i])
            if need_hidden:
                hidden_state.append(out)
            if return_attn_weights:
                attn_weights.append(layer.attention.last_attn)

        out = self.dropout(out)

        if using_override:
            pass  # deliberately do not commit -- this pass must not affect live memory
        else:
            self.memory.update(hidden_state)

        if batch_first:
            out = torch.transpose(out, 1, 0)

        output = {"logit": out}
        if return_mem:
            output["memory"] = memory
        if return_hidden_states:
            output["hidden_states"] = hidden_state
        if return_attn_weights:
            output["attn_weights"] = attn_weights
        return output


# 
# Observation encoder + GTrXL wrapper exposing the buffer/PPO-script interface
# 

class GTrXLFeaturesExtractor(nn.Module):
    """
    CNN(viewport) + MLP(positions) -> DiagGTrXL, presenting the same public
    surface your PPO script and diagnostics module already call into.
    """

    def __init__(
        self,
        observation_space,
        features_dim: int,
        memory_len: int,
        n_layers: int,
        n_heads: int,
        d_ff_multiplier: float = 2.0,
        dropout: float = 0.0,
        head_dim: int = 128,
        mlp_num: int = 2,
        gru_bias: float = 2.0,
        cnn_out_dim: int = 512,
        pos_out_dim: int = 32,
    ):
        super().__init__()
        vp_shape = observation_space.spaces["viewport"].shape   # (C, 84, 84)
        pos_shape = observation_space.spaces["positions"].shape  # (4,)
        in_channels = vp_shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, *vp_shape)
            flat_dim = self.cnn(dummy).shape[1]
        self.cnn_proj = nn.Sequential(nn.Linear(flat_dim, cnn_out_dim), nn.ReLU())
        self.pos_mlp = nn.Sequential(nn.Linear(pos_shape[0], pos_out_dim), nn.ReLU())

        gtrxl_input_dim = cnn_out_dim + pos_out_dim
        d_ff = int(features_dim * d_ff_multiplier)

        self.core = DiagGTrXL(
            input_dim=gtrxl_input_dim,
            head_dim=head_dim,
            embedding_dim=features_dim,
            head_num=n_heads,
            mlp_num=mlp_num,
            layer_num=n_layers,
            memory_len=memory_len,
            dropout_ratio=dropout,
            activation=nn.ReLU(),
            gru_gating=True,
            gru_bias=gru_bias,
            use_embedding_layer=True,
        )
        # d_ff is absorbed into DiagGTrXL's internal MLP dim (== embedding_dim
        # there, per DI-engine's own design); kept as a ctor arg only so cfg
        # wiring in the training script doesn't need special-casing.
        del d_ff

        self.features_dim = features_dim
        self.memory_len = memory_len
        self.n_layers = n_layers                # transformer blocks
        self.n_memory_slices = n_layers + 1      # + the embedding-level slice
        self._segment_hiddens = None             # unused by GTrXL; kept so the rest
                                                  # of the (TrXL-authored) script doesn't
                                                  # need special-casing around this attr.

    # ---- memory <-> buffer format adapters --------------------------------
    # DI-engine stores memory as (n_memory_slices, memory_len, bs, D).
    # The rollout buffer / PPO loop expect a list of n_memory_slices tensors
    # each shaped (bs, memory_len, D) -- the same convention the original
    # TrXLExtractor used, so no other file needs to change.

    @property
    def memory(self):
        if self.core.memory is None:
            return None
        mem = self.core.memory.get()  # (n_memory_slices, memory_len, bs, D)
        return [mem[i].permute(1, 0, 2) for i in range(mem.shape[0])]

    @memory.setter
    def memory(self, layer_list):
        if layer_list is None:
            self.core.memory = None
            return
        stacked = torch.stack([m.permute(1, 0, 2) for m in layer_list], dim=0)
        self.core.reset_memory(state=stacked)

    def init_memory(self, batch_size, device):
        self.core.reset_memory(batch_size=batch_size)
        self.core.memory.to(device)

    def reset_memory(self, env_indices):
        """Zero only the given envs' slice of the rolling memory, leaving the
        rest of the batch untouched -- used on episode boundaries."""
        if self.core.memory is None or len(env_indices) == 0:
            return
        mem = self.core.memory.get()  # (n_memory_slices, memory_len, bs, D) -- live ref
        for idx in env_indices:
            mem[:, :, idx, :] = 0.0

    # ---- forward ------------------------------------------------------------

    def _encode(self, obs):
        vp = self.cnn_proj(self.cnn(obs["viewport"]))
        pos = self.pos_mlp(obs["positions"])
        x = torch.cat([vp, pos], dim=-1)   # (bs, input_dim)
        return x.unsqueeze(0)              # (cur_seq=1, bs, input_dim)

    def forward(self, obs, memory_override=None, return_attn_weights=False, return_hidden_states=False):
        x = self._encode(obs)

        internal_override = None
        if memory_override is not None:
            internal_override = torch.stack([m.permute(1, 0, 2) for m in memory_override], dim=0)

        out = self.core(
            x,
            memory_override=internal_override,
            return_attn_weights=return_attn_weights,
            return_hidden_states=return_hidden_states,
            return_mem=True,
        )
        features = out["logit"].squeeze(0)  # (bs, features_dim)

        if not return_attn_weights and not return_hidden_states:
            return features

        attn_weights = out.get("attn_weights", None)  # list[(bs, n_heads, 1, memory_len+1)]
        hidden_states = None
        if return_hidden_states:
            # (cur_seq=1, bs, D) -> (bs, 1, D), matching what the diagnostics
            # module expects (it does `.squeeze(1)` on each element).
            hidden_states = [h.transpose(0, 1) for h in out["hidden_states"]]

        return features, attn_weights, hidden_states