"""
GTrXL memory diagnostics: recency profile, attention entropy, head
specialization, layerwise representational similarity (CKA) under a
memory-ablation counterfactual, a full layer-vs-layer CKA similarity
matrix, and GRU-gate activation statistics.

This is a port of the original TrXL diagnostics module onto
policies.gtrxl_extractor.GTrXLFeaturesExtractor (DI-engine's GTrXL under
the hood). Only compute_gate_statistics() actually changes in substance --
it now reads gate activations off DiagGatedTransformerXLLayer.gate1/gate2
instead of a bespoke block.attn_gate/ff_gate pair, since GTrXL's own gating
scheme is a single GRU gate around attention and a second around the MLP
(DiagGRUGatingUnit.last_gate), analogous in spirit but not identically
named to the original architecture's gates. Everything else -- the shape
conventions for attn_weights (B, n_heads, 1, memory_len+1) and
hidden_states (B, 1, D), and the return-tuple contract of the extractor's
forward() -- lines up with GTrXLFeaturesExtractor as built, so those
functions are unchanged from the original.

All passes here are DIAGNOSTIC-ONLY: they call extractor(..., memory_override=...)
with a frozen snapshot of the live memory, so none of them advance or
otherwise disturb the rolling memory state used by the actual rollout.
"""

import matplotlib
matplotlib.use("Agg")  # headless -- no display available in a training process
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb


# 
# 1. Recency profile
# 

def compute_recency_profile(attn_weights: list, memory_len: int):
    """
    attn_weights: list of n_layers tensors, each (B, n_heads, 1, memory_len+1)
    KV sequence order is [oldest_memory ... newest_memory, current_token],
    so kv index j corresponds to lag = memory_len - j (current token -> lag 0).

    Returns: dict {
        "lags": np.ndarray (memory_len+1,)         -- 0 (current) .. memory_len (oldest)
        "profile_per_layer": np.ndarray (n_layers, memory_len+1)  -- avg over B, heads
    }
    """
    n_layers = len(attn_weights)
    lags = np.arange(memory_len, -1, -1)
    order = np.argsort(lags)
    lags_sorted = lags[order]

    profile_per_layer = np.zeros((n_layers, memory_len + 1), dtype=np.float32)
    for i, w in enumerate(attn_weights):
        avg = w.mean(dim=(0, 1, 2)).cpu().numpy()  # (memory_len+1,)
        profile_per_layer[i] = avg[order]

    return {"lags": lags_sorted, "profile_per_layer": profile_per_layer}


# 
# 2. Attention entropy
# 

def compute_attention_entropy(attn_weights: list, eps: float = 1e-8):
    """
    attn_weights: list of n_layers tensors, each (B, n_heads, 1, memory_len+1)
    Returns: dict {
        "entropy_per_layer": np.ndarray (n_layers,)
        "entropy_per_layer_head": np.ndarray (n_layers, n_heads)
    }
    """
    n_layers = len(attn_weights)
    n_heads = attn_weights[0].shape[1]

    entropy_per_layer = np.zeros(n_layers, dtype=np.float32)
    entropy_per_layer_head = np.zeros((n_layers, n_heads), dtype=np.float32)

    for i, w in enumerate(attn_weights):
        p = w.squeeze(2)  # (B, n_heads, memory_len+1)
        ent = -(p * (p + eps).log()).sum(dim=-1)
        entropy_per_layer_head[i] = ent.mean(dim=0).cpu().numpy()
        entropy_per_layer[i] = ent.mean().item()

    return {
        "entropy_per_layer": entropy_per_layer,
        "entropy_per_layer_head": entropy_per_layer_head,
    }


# 
# 3. Head specialization
# 

def compute_head_specialization(attn_weights: list, memory_len: int, memory_mass_threshold: float = 0.5):
    """
    Returns: dict {
        "memory_mass_per_layer_head": np.ndarray (n_layers, n_heads)
        "is_memory_head": np.ndarray (n_layers, n_heads) bool
        "fraction_memory_heads": float
    }
    """
    n_layers = len(attn_weights)
    n_heads = attn_weights[0].shape[1]

    memory_mass = np.zeros((n_layers, n_heads), dtype=np.float32)
    for i, w in enumerate(attn_weights):
        p = w.squeeze(2)
        mem_mass = p[:, :, :memory_len].sum(dim=-1)
        memory_mass[i] = mem_mass.mean(dim=0).cpu().numpy()

    is_memory_head = memory_mass > memory_mass_threshold
    return {
        "memory_mass_per_layer_head": memory_mass,
        "is_memory_head": is_memory_head,
        "fraction_memory_heads": float(is_memory_head.mean()),
    }


# 
# 4. Linear CKA + layerwise memory-ablation comparison
# 

def linear_cka(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-8) -> float:
    """
    X, Y: (N, D) feature matrices, same N. Standard linear CKA (Kornblith
    et al. 2019) via the HSIC-linear-kernel identity.
    """
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    xty_f = torch.norm(Y.T @ X, p="fro")
    xtx_f = torch.norm(X.T @ X, p="fro")
    yty_f = torch.norm(Y.T @ Y, p="fro")

    return ((xty_f ** 2) / (xtx_f * yty_f + eps)).item()


def compute_layerwise_cka_memory_ablation(agent, obs_batch: dict, device):
    """
    Runs the extractor twice on the same obs batch -- once with the real
    (live) memory, once with memory forcibly zeroed -- and computes linear
    CKA between the two runs' hidden states at each layer.

    extractor(obs_batch, memory_override=..., return_hidden_states=True)
    is read-only w.r.t. the rolling memory (see DiagGTrXL.forward), so
    both calls below leave live training memory untouched.

    Returns: dict {
        "cka_per_layer":       np.ndarray (n_layers+1,)  -- index 0 is the
                                input embedding (should be ~1.0, sanity check)
        "l2_diff_per_layer":   np.ndarray (n_layers+1,)
        "suspiciously_identical": list of layer indices (>0) where
                                hidden_real and hidden_zero are numerically
                                identical -- a routing-bug signal, not a
                                training-dynamics signal
        "real_memory_norm": float
    }
    """
    extractor = agent.extractor
    B = next(iter(obs_batch.values())).shape[0]

    with torch.no_grad():
        if extractor.memory is None or extractor.memory[0].shape[0] != B:
            extractor.init_memory(B, device)
        real_memory = [m.clone() for m in extractor.memory]

        _, _, hidden_real = extractor(
            obs_batch, memory_override=real_memory, return_hidden_states=True
        )

        zero_memory = [torch.zeros_like(m) for m in real_memory]
        _, _, hidden_zero = extractor(
            obs_batch, memory_override=zero_memory, return_hidden_states=True
        )

    n_points = len(hidden_real)
    cka_per_layer = np.zeros(n_points, dtype=np.float32)
    l2_diff_per_layer = np.zeros(n_points, dtype=np.float32)
    suspiciously_identical = []

    real_memory_norm = float(torch.stack([m.norm() for m in real_memory]).mean())

    for i in range(n_points):
        X = hidden_real[i].squeeze(1)  # (B, D)
        Y = hidden_zero[i].squeeze(1)  # (B, D)
        cka_per_layer[i] = linear_cka(X, Y)
        l2_diff_per_layer[i] = (X - Y).norm(dim=-1).mean().item()

        if i > 0 and torch.allclose(X, Y, atol=1e-5, rtol=1e-4):
            suspiciously_identical.append(i)

    if suspiciously_identical:
        print(
            f"[CKA-DIAG] WARNING: hidden states at layer(s) {suspiciously_identical} "
            f"are numerically IDENTICAL between real and zeroed memory. This is NOT "
            f"the expected 'gated down' behavior -- check that memory_override is "
            f"actually reaching DiagAttentionXL.forward (i.e. DiagGTrXL.forward is "
            f"indexing into the overridden memory tensor, not self.memory)."
        )
    if real_memory_norm < 1e-4:
        print(
            f"[CKA-DIAG] WARNING: live extractor.memory has near-zero norm "
            f"({real_memory_norm:.6f}). The 'real vs zeroed' comparison is "
            f"meaningless if real memory is already close to zero -- this can "
            f"happen right after a batch of envs reset."
        )

    return {
        "cka_per_layer": cka_per_layer,
        "l2_diff_per_layer": l2_diff_per_layer,
        "suspiciously_identical": suspiciously_identical,
        "real_memory_norm": real_memory_norm,
    }


# 
# 4b. Full layer x layer CKA similarity matrix
# 

def compute_full_layer_cka_matrix(hidden_states: list) -> np.ndarray:
    """
    hidden_states: list of L tensors (L = n_layers+1), each (B, 1, D).
    Answers "how similar is layer i's representation to layer j's" for
    every pair, both under the same (real-memory) condition.

    Note on interpreting absolute values: DiagGatedTransformerXLLayer's
    GRU gating (a highway-style residual) structurally biases adjacent-
    layer CKA toward 1.0 regardless of training progress. The TREND over
    training and non-adjacent pairs (e.g. input vs. final block) are more
    informative than any single adjacent-pair snapshot value.
    """
    L = len(hidden_states)
    feats = [h.squeeze(1) for h in hidden_states]

    mat = np.ones((L, L), dtype=np.float32)
    for i in range(L):
        for j in range(i + 1, L):
            c = linear_cka(feats[i], feats[j])
            mat[i, j] = c
            mat[j, i] = c
    return mat


def plot_cka_matrix_heatmap(cka_matrix: np.ndarray, title: str = "Layer-wise CKA Similarity"):
    """
    Renders the (L, L) matrix as an annotated matplotlib heatmap. Caller
    wraps it: wandb.Image(fig), and should plt.close(fig) afterward.
    """
    L = cka_matrix.shape[0]
    labels = ["input"] + [f"block{i}" for i in range(1, L)]

    fig, ax = plt.subplots(figsize=(1.2 * L + 2, 1.2 * L + 2))
    vmin = max(0.0, float(cka_matrix.min()) - 0.02)
    im = ax.imshow(cka_matrix, cmap="viridis", vmin=vmin, vmax=1.0)

    ax.set_xticks(range(L))
    ax.set_yticks(range(L))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(L):
        for j in range(L):
            val = cka_matrix[i, j]
            color = "white" if val < (vmin + 1.0) / 2 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=color, fontsize=9)

    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Linear CKA")
    fig.tight_layout()
    return fig


def cka_matrix_to_scalar_dict(cka_matrix: np.ndarray) -> dict:
    """
    Flattens the upper triangle (excluding the diagonal) into scalars keyed
    so wandb auto-plots each pair as its own line over global_step.
    """
    L = cka_matrix.shape[0]
    labels = ["input"] + [f"block{i}" for i in range(1, L)]

    out = {}
    for i in range(L):
        for j in range(i + 1, L):
            out[f"memory/cka_pair_{labels[i]}_{labels[j]}"] = float(cka_matrix[i, j])
    return out


# 
# 5. Gate activation statistics
#    GTrXL uses one DiagGRUGatingUnit around attention (gate1) and one
#    around the MLP (gate2) per block; DiagGRUGatingUnit.last_gate holds
#    the update-gate tensor `z` as a side effect of the forward pass
#    already performed above -- no extra forward call needed. With
#    gru_bias=2.0 (DI-engine default), gates start at sigmoid(-2) ~= 0.12
#    and are EXPECTED to be near that early in training -- watch this
#    trend upward, not just its value at one checkpoint.
# 

def compute_gate_statistics(agent) -> dict:
    """
    Must be called AFTER a forward pass (e.g. right after the call inside
    run_full_diagnostics) so DiagGRUGatingUnit.last_gate is populated.
    Returns: dict {
        "attn_gate_mean_per_layer": np.ndarray (n_layers,)
        "ff_gate_mean_per_layer":   np.ndarray (n_layers,)
    }
    """
    layers = agent.extractor.core.layers  # nn.Sequential[DiagGatedTransformerXLLayer]
    n_layers = len(layers)
    attn_gate_means = np.zeros(n_layers, dtype=np.float32)
    ff_gate_means = np.zeros(n_layers, dtype=np.float32)

    for i, block in enumerate(layers):
        if getattr(block, "gate1", None) is not None and block.gate1.last_gate is not None:
            attn_gate_means[i] = block.gate1.last_gate.mean().item()
        if getattr(block, "gate2", None) is not None and block.gate2.last_gate is not None:
            ff_gate_means[i] = block.gate2.last_gate.mean().item()

    return {
        "attn_gate_mean_per_layer": attn_gate_means,
        "ff_gate_mean_per_layer": ff_gate_means,
    }


# 
# 6. Top-level: run everything, return a wandb-loggable dict
# 

def run_full_diagnostics(agent, obs_batch: dict, device, memory_len: int) -> dict:
    """
    obs_batch: dict of (B, *shape) tensors on `device`. Use a batch drawn
    from the current rollout so memory reflects real training-time state.

    Returns a flat dict of wandb-loggable values -- pass straight into
    wandb.log({**diagnostics, "global_step": global_step}).
    """
    extractor = agent.extractor
    B = next(iter(obs_batch.values())).shape[0]

    with torch.no_grad():
        if extractor.memory is None or extractor.memory[0].shape[0] != B:
            extractor.init_memory(B, device)
        # Frozen snapshot so this diagnostic pass doesn't slide the live
        # rolling memory window an extra, off-schedule step.
        memory_snapshot = [m.clone() for m in extractor.memory]

        _, attn_weights, hidden_states = extractor(
            obs_batch,
            memory_override=memory_snapshot,
            return_attn_weights=True,
            return_hidden_states=True,
        )

    recency = compute_recency_profile(attn_weights, memory_len)
    entropy = compute_attention_entropy(attn_weights)
    heads = compute_head_specialization(attn_weights, memory_len)
    cka = compute_layerwise_cka_memory_ablation(agent, obs_batch, device)
    gates = compute_gate_statistics(agent)

    cka_matrix = compute_full_layer_cka_matrix(hidden_states)

    log_dict = {}

    for i, layer_profile in enumerate(recency["profile_per_layer"]):
        table = wandb.Table(
            columns=["lag", "attention_weight"],
            data=[[int(l), float(w)] for l, w in zip(recency["lags"], layer_profile)],
        )
        log_dict[f"memory/recency_profile_layer{i}"] = wandb.plot.line(
            table, x="lag", y="attention_weight",
            title=f"Recency Profile - Layer {i}",
        )

    for i, e in enumerate(entropy["entropy_per_layer"]):
        log_dict[f"memory/attn_entropy_layer{i}"] = float(e)
    log_dict["memory/attn_entropy_mean"] = float(entropy["entropy_per_layer"].mean())

    log_dict["memory/fraction_memory_heads"] = heads["fraction_memory_heads"]
    for i in range(heads["memory_mass_per_layer_head"].shape[0]):
        log_dict[f"memory/memory_mass_layer{i}"] = float(heads["memory_mass_per_layer_head"][i].mean())

    for i, c in enumerate(cka["cka_per_layer"]):
        log_dict[f"memory/cka_layer{i}"] = float(c)
    log_dict["memory/cka_mean_blocks"] = float(cka["cka_per_layer"][1:].mean())

    for i, d in enumerate(cka["l2_diff_per_layer"]):
        log_dict[f"memory/cka_l2_diff_layer{i}"] = float(d)
    log_dict["memory/live_memory_norm"] = cka["real_memory_norm"]
    log_dict["memory/n_suspiciously_identical_layers"] = len(cka["suspiciously_identical"])

    fig = plot_cka_matrix_heatmap(cka_matrix)
    log_dict["memory/cka_layer_matrix"] = wandb.Image(fig)
    plt.close(fig)
    log_dict.update(cka_matrix_to_scalar_dict(cka_matrix))

    for i, g in enumerate(gates["attn_gate_mean_per_layer"]):
        log_dict[f"memory/attn_gate_mean_layer{i}"] = float(g)
    for i, g in enumerate(gates["ff_gate_mean_per_layer"]):
        log_dict[f"memory/ff_gate_mean_layer{i}"] = float(g)
    log_dict["memory/attn_gate_mean"] = float(gates["attn_gate_mean_per_layer"].mean())
    log_dict["memory/ff_gate_mean"] = float(gates["ff_gate_mean_per_layer"].mean())

    return log_dict