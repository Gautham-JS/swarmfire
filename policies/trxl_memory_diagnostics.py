"""
TrXL memory diagnostics: recency profile, attention entropy, head
specialization, layerwise representational similarity (CKA) under a
memory-ablation counterfactual, and a full layer-vs-layer CKA similarity
matrix (heatmap + scalar trend lines).

These are all DIAGNOSTIC-ONLY passes -- they call extractor.forward()
with return_attn_weights=True / return_hidden_states=True, which does
not touch or advance self.memory differently than a normal forward call
would (memory still updates on non-update-pass calls, same as always).
If you don't want a diagnostic pass to disturb the rolling memory state
during training, pass memory_override explicitly (see run_full_diagnostics
below, which uses the buffer's stored memory snapshot rather than the
live extractor.memory).

WHAT CHANGED vs the previous version:
  - ADDED: compute_full_layer_cka_matrix() -- unlike
    compute_layerwise_cka_memory_ablation() (which compares layer i under
    real memory vs. layer i under zeroed memory -- "does this layer use
    memory?"), this compares layer i vs. layer j, both under real memory
    -- "how similar are representations at different depths?" Reuses the
    hidden_states already computed in run_full_diagnostics's main forward
    call, so no extra forward pass.
  - ADDED: plot_cka_matrix_heatmap() -- renders that matrix as an
    annotated matplotlib heatmap, logged via wandb.Image (wandb has no
    native heatmap-from-arbitrary-matrix plot, so this is the standard
    approach for this kind of visualization).
  - ADDED: cka_matrix_to_scalar_dict() -- flattens the same matrix's
    upper triangle into plain scalars keyed per-pair, so wandb
    auto-plots each layer-pair's CKA as a line over global_step. This is
    the more useful of the two views for judging whether depth is doing
    anything, since absolute CKA between adjacent layers tends to sit
    high (0.9+) in gated/residual architectures like this one regardless
    of training progress -- the trend is the signal, not any single
    snapshot's value.
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
    lags = np.arange(memory_len, -1, -1)  # kv index 0 -> lag=memory_len ... index memory_len -> lag=0
    # reorder so profile[i] corresponds to lags[i] ascending (0..memory_len) for easier plotting
    order = np.argsort(lags)  # ascending lag order
    lags_sorted = lags[order]

    profile_per_layer = np.zeros((n_layers, memory_len + 1), dtype=np.float32)
    for i, w in enumerate(attn_weights):
        # w: (B, n_heads, 1, memory_len+1) -> avg over B, heads, query dim
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
        "entropy_per_layer": np.ndarray (n_layers,)              -- avg over B, heads
        "entropy_per_layer_head": np.ndarray (n_layers, n_heads) -- avg over B only
    }
    """
    n_layers = len(attn_weights)
    n_heads  = attn_weights[0].shape[1]

    entropy_per_layer      = np.zeros(n_layers, dtype=np.float32)
    entropy_per_layer_head = np.zeros((n_layers, n_heads), dtype=np.float32)

    for i, w in enumerate(attn_weights):
        p = w.squeeze(2)  # (B, n_heads, memory_len+1)
        ent = -(p * (p + eps).log()).sum(dim=-1)  # (B, n_heads)
        entropy_per_layer_head[i] = ent.mean(dim=0).cpu().numpy()
        entropy_per_layer[i]      = ent.mean().item()

    return {
        "entropy_per_layer": entropy_per_layer,
        "entropy_per_layer_head": entropy_per_layer_head,
    }


# 
# 3. Head specialization: local (attends mostly to current token) vs
#    memory (attends mostly into the cached region)
# 

def compute_head_specialization(attn_weights: list, memory_len: int, memory_mass_threshold: float = 0.5):
    """
    Returns: dict {
        "memory_mass_per_layer_head": np.ndarray (n_layers, n_heads)  -- avg over B
        "is_memory_head": np.ndarray (n_layers, n_heads) bool
        "fraction_memory_heads": float
    }
    """
    n_layers = len(attn_weights)
    n_heads  = attn_weights[0].shape[1]

    memory_mass = np.zeros((n_layers, n_heads), dtype=np.float32)
    for i, w in enumerate(attn_weights):
        p = w.squeeze(2)                      # (B, n_heads, memory_len+1)
        mem_mass = p[:, :, :memory_len].sum(dim=-1)  # exclude current-token index (last)
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
    X, Y: (N, D) feature matrices, same N (same samples), D can differ.
    Standard linear CKA (Kornblith et al. 2019), computed via the
    HSIC-linear-kernel identity: CKA = ||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F)
    """
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    xty_f = torch.norm(Y.T @ X, p="fro")
    xtx_f = torch.norm(X.T @ X, p="fro")
    yty_f = torch.norm(Y.T @ Y, p="fro")

    return ((xty_f ** 2) / (xtx_f * yty_f + eps)).item()


def compute_layerwise_cka_memory_ablation(agent, obs_batch: dict, device):
    """
    Runs the extractor twice on the same obs batch:
      - once with the real (live) memory
      - once with memory forcibly zeroed (counterfactual: "no history")
    and computes linear CKA between the two runs' hidden states at each
    layer. Low CKA at a layer = that layer's representation depends
    heavily on memory content; CKA near 1 = memory barely changes what
    that layer computes.

    Also computes raw (non-scale-invariant) L2 difference and flags cases
    where hidden_real and hidden_zero come out numerically IDENTICAL --
    CKA is scale/rotation invariant, so a tiny genuine difference can
    still register as CKA~1. A truly identical pair (within float
    tolerance) is a different, more serious signal: it means
    memory_override isn't actually reaching the computation (a routing
    bug), not just that its effect has been gated down to near-zero.

    obs_batch: dict of (B, *shape) tensors, already on `device`.
    Returns: dict {
        "cka_per_layer":       np.ndarray (n_layers+1,)  -- index 0 is the
                                input embedding (should be ~1.0, sanity
                                check), index i is block i's output
        "l2_diff_per_layer":   np.ndarray (n_layers+1,)  -- mean per-sample
                                L2 norm of (hidden_real - hidden_zero)
        "suspiciously_identical": list of layer indices (>0) where
                                hidden_real and hidden_zero are within
                                float tolerance of each other -- a bug
                                signal, not a training-dynamics signal
    }
    """
    extractor = agent.extractor
    B = next(iter(obs_batch.values())).shape[0]

    with torch.no_grad():
        # Real memory: use whatever's currently live (must already be initialised)
        if extractor.memory is None or extractor.memory[0].shape[0] != B:
            extractor.init_memory(B, device)
        real_memory = [m.clone() for m in extractor.memory]

        # NOTE: forward() with memory_override does NOT advance/mutate
        # self.memory (that only happens on the live, non-override path),
        # so both calls below are read-only w.r.t. the rolling memory state.
        _, _, hidden_real = extractor(
            obs_batch, memory_override=real_memory, return_hidden_states=True
        )

        zero_memory = [torch.zeros_like(m) for m in real_memory]
        _, _, hidden_zero = extractor(
            obs_batch, memory_override=zero_memory, return_hidden_states=True
        )

    n_points = len(hidden_real)  # n_layers + 1
    cka_per_layer     = np.zeros(n_points, dtype=np.float32)
    l2_diff_per_layer = np.zeros(n_points, dtype=np.float32)
    suspiciously_identical = []

    real_memory_norm = float(torch.stack([m.norm() for m in real_memory]).mean())

    for i in range(n_points):
        X = hidden_real[i].squeeze(1)  # (B, D)
        Y = hidden_zero[i].squeeze(1)  # (B, D)
        cka_per_layer[i]     = linear_cka(X, Y)
        l2_diff_per_layer[i] = (X - Y).norm(dim=-1).mean().item()

        if i > 0 and torch.allclose(X, Y, atol=1e-5, rtol=1e-4):
            suspiciously_identical.append(i)

    if suspiciously_identical:
        print(
            f"[CKA-DIAG] WARNING: hidden states at layer(s) {suspiciously_identical} "
            f"are numerically IDENTICAL between real and zeroed memory. This is NOT "
            f"the expected 'gated down' behavior (which would show a small but "
            f"nonzero l2_diff) -- check that memory_override is actually being "
            f"routed into block.attn_sublayer's key/value concat."
        )
    if real_memory_norm < 1e-4:
        print(
            f"[CKA-DIAG] WARNING: live extractor.memory has near-zero norm "
            f"({real_memory_norm:.6f}). The 'real vs zeroed' comparison is "
            f"meaningless if real memory is already close to zero -- this can "
            f"happen right after a batch of envs reset, or if _update_memory "
            f"isn't accumulating meaningful content. Check memory norm over "
            f"a full rollout, not just this one diagnostic snapshot."
        )

    return {
        "cka_per_layer": cka_per_layer,
        "l2_diff_per_layer": l2_diff_per_layer,
        "suspiciously_identical": suspiciously_identical,
        "real_memory_norm": real_memory_norm,
    }


# 
# 4b. Full layer x layer CKA similarity matrix (depth-wise representational
#     drift, under normal real-memory operation -- a different question
#     from the memory-ablation comparison above)
# 

def compute_full_layer_cka_matrix(hidden_states: list) -> np.ndarray:
    """
    hidden_states: list of L tensors (L = n_layers+1), each (B, 1, D).
    D can differ across layers only if you've changed the model to vary
    width per layer -- with a constant d_model (as in TrXLExtractor) all
    entries are directly comparable.

    Answers "how similar is layer i's representation to layer j's
    representation" for every pair, all under the SAME (real-memory)
    condition -- as opposed to compute_layerwise_cka_memory_ablation,
    which fixes the layer and varies the memory condition instead.

    Returns: (L, L) symmetric matrix, diagonal == 1.0 by construction
    (CKA of a representation with itself is always 1).

    Note on interpreting absolute values: TrXLSplitBlock's GatingUnit
    (highway-style residual) and HyperConnection (explicit weighted mix
    of ALL earlier hidden states) both structurally bias adjacent-layer
    CKA toward 1.0 regardless of training progress -- this is expected
    for gated/residual architectures, not a bug. The TREND over training
    (see cka_matrix_to_scalar_dict) and non-adjacent pairs (e.g. input
    vs. final block) are more informative than any single adjacent-pair
    snapshot value.

    Cost: O(L^2) linear_cka calls, each O(B*D^2) -- negligible next to a
    forward pass for typical L (n_layers+1 ~ 3-5) and B (rollout batch).
    """
    L = len(hidden_states)
    feats = [h.squeeze(1) for h in hidden_states]  # each (B, D)

    mat = np.ones((L, L), dtype=np.float32)  # diagonal defaults to 1.0
    for i in range(L):
        for j in range(i + 1, L):
            c = linear_cka(feats[i], feats[j])
            mat[i, j] = c
            mat[j, i] = c  # CKA is symmetric
    return mat


def plot_cka_matrix_heatmap(cka_matrix: np.ndarray, title: str = "Layer-wise CKA Similarity"):
    """
    Renders the (L, L) matrix as an annotated matplotlib heatmap and
    returns the Figure. Caller wraps it: wandb.Image(fig), and should
    plt.close(fig) afterward to avoid leaking figures over a long run.

    Labels: index 0 = "input", index i>0 = "block{i}" -- matches the
    hidden_states convention used throughout this file.
    """
    L = cka_matrix.shape[0]
    labels = ["input"] + [f"block{i}" for i in range(1, L)]

    fig, ax = plt.subplots(figsize=(1.2 * L + 2, 1.2 * L + 2))
    # Values here typically sit in the 0.9-1.0 range for this architecture
    # (see note in compute_full_layer_cka_matrix), so vmin=0 would wash out
    # all the interesting variation into a single color. Anchor to the
    # matrix's own observed range instead.
    vmin = max(0.0, float(cka_matrix.min()) - 0.02)
    im = ax.imshow(cka_matrix, cmap="viridis", vmin=vmin, vmax=1.0)

    ax.set_xticks(range(L))
    ax.set_yticks(range(L))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(L):
        for j in range(L):
            val = cka_matrix[i, j]
            # White text on dark cells, black text on light cells, so
            # annotations stay legible across the whole colormap range.
            color = "white" if val < (vmin + 1.0) / 2 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                     color=color, fontsize=9)

    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Linear CKA")
    fig.tight_layout()
    return fig


def cka_matrix_to_scalar_dict(cka_matrix: np.ndarray) -> dict:
    """
    Flattens the upper triangle (excluding the diagonal, which is always
    1.0 by construction and carries no information) into plain scalars,
    keyed so wandb auto-plots each pair as its own line over global_step --
    this is what actually lets you watch the TREND, which matters more
    than any single snapshot's absolute value.

    Labels match plot_cka_matrix_heatmap: index 0 = "input", index i>0 =
    "block{i}".

    Returns: dict {"memory/cka_pair_input_block1": 0.987, ...} -- merge
    directly into run_full_diagnostics' log_dict.

    Worth watching in particular: the widest-gap pair (input vs. final
    block) -- non-adjacent pairs are less structurally biased toward high
    CKA by the hyperconnection mixing than adjacent pairs are, so movement
    there is a cleaner signal of the network actually doing something with
    depth.
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
#    Reads GatingUnit.last_gate (set as a side effect of the forward pass
#    already performed above -- no extra forward call needed). Directly
#    answers "are the GTrXL gates actually open" rather than inferring it
#    indirectly from CKA. With gate_bias=-2.0, gates start at sigmoid(-2)
#    ~= 0.12 and are EXPECTED to be near that early in training -- watch
#    this trend upward over training, not just its value at one checkpoint.
# 

def compute_gate_statistics(agent) -> dict:
    """
    Must be called AFTER a forward pass (e.g. right after the call inside
    run_full_diagnostics) so GatingUnit.last_gate is populated.
    Returns: dict {
        "attn_gate_mean_per_layer": np.ndarray (n_layers,)
        "ff_gate_mean_per_layer":   np.ndarray (n_layers,)
    }
    """
    n_layers = len(agent.extractor.blocks)
    attn_gate_means = np.zeros(n_layers, dtype=np.float32)
    ff_gate_means   = np.zeros(n_layers, dtype=np.float32)

    for i, block in enumerate(agent.extractor.blocks):
        if block.attn_gate.last_gate is not None:
            attn_gate_means[i] = block.attn_gate.last_gate.mean().item()
        if block.ff_gate.last_gate is not None:
            ff_gate_means[i] = block.ff_gate.last_gate.mean().item()

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
    from the current rollout (e.g. vec_obs_to_tensor(obs_dict, device))
    so memory reflects real training-time state, not a cold-started one.

    Returns a flat dict of wandb-loggable values -- pass straight into
    wandb.log({**diagnostics, "global_step": global_step}).
    """
    extractor = agent.extractor
    B = next(iter(obs_batch.values())).shape[0]

    with torch.no_grad():
        if extractor.memory is None or extractor.memory[0].shape[0] != B:
            extractor.init_memory(B, device)
        # Use a frozen snapshot so this diagnostic pass doesn't slide the
        # live rolling memory window an extra, "off-schedule" step.
        memory_snapshot = [m.clone() for m in extractor.memory]

        _, attn_weights, hidden_states = extractor(
            obs_batch,
            memory_override=memory_snapshot,
            return_attn_weights=True,
            return_hidden_states=True,
        )

    recency = compute_recency_profile(attn_weights, memory_len)
    entropy = compute_attention_entropy(attn_weights)
    heads   = compute_head_specialization(attn_weights, memory_len)
    cka     = compute_layerwise_cka_memory_ablation(agent, obs_batch, device)
    gates   = compute_gate_statistics(agent)  # reads GatingUnit.last_gate set by the forward call above

    # Full layer x layer CKA matrix -- reuses hidden_states from the main
    # forward call above (real memory, no ablation), no extra forward pass.
    cka_matrix = compute_full_layer_cka_matrix(hidden_states)

    log_dict = {}

    # Recency profile as a wandb line plot: weight vs lag, one line per layer
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
        log_dict[f"memory/memory_mass_layer{i}"] = float(
            heads["memory_mass_per_layer_head"][i].mean()
        )

    for i, c in enumerate(cka["cka_per_layer"]):
        # index 0 = input embedding CKA (near 1.0 is expected/sanity-check,
        # since it doesn't depend on memory at all); index i>0 = block i's output
        log_dict[f"memory/cka_layer{i}"] = float(c)
    log_dict["memory/cka_mean_blocks"] = float(cka["cka_per_layer"][1:].mean())

    # Raw (non-scale-invariant) difference, to sanity-check CKA -- a small
    # but nonzero l2_diff alongside CKA~1 is consistent with "gated down",
    # whereas l2_diff~0 (== a value in suspiciously_identical) means the
    # memory_override isn't reaching the computation at all.
    for i, d in enumerate(cka["l2_diff_per_layer"]):
        log_dict[f"memory/cka_l2_diff_layer{i}"] = float(d)
    log_dict["memory/live_memory_norm"] = cka["real_memory_norm"]
    log_dict["memory/n_suspiciously_identical_layers"] = len(cka["suspiciously_identical"])

    # Full layer x layer CKA matrix: heatmap snapshot (structure at this
    # point in training) + scalar trend lines (change over training).
    fig = plot_cka_matrix_heatmap(cka_matrix)
    log_dict["memory/cka_layer_matrix"] = wandb.Image(fig)
    plt.close(fig)  # avoid leaking figures over a long training run
    log_dict.update(cka_matrix_to_scalar_dict(cka_matrix))

    # Gate activations -- with gate_bias=-2.0, expect these to start near
    # sigmoid(-2) ~= 0.12 and trend upward over training if the model is
    # learning to open the gates. Stuck near ~0.12 late into training,
    # together with cka_mean_blocks ~= 1, points at closed/stuck gates
    # rather than a code bug.
    for i, g in enumerate(gates["attn_gate_mean_per_layer"]):
        log_dict[f"memory/attn_gate_mean_layer{i}"] = float(g)
    for i, g in enumerate(gates["ff_gate_mean_per_layer"]):
        log_dict[f"memory/ff_gate_mean_layer{i}"] = float(g)
    log_dict["memory/attn_gate_mean"] = float(gates["attn_gate_mean_per_layer"].mean())
    log_dict["memory/ff_gate_mean"]   = float(gates["ff_gate_mean_per_layer"].mean())

    return log_dict