"""
Belief-state probing for TrXL.

Same idea as probe_belief_state.py (RPPO), adapted for TrXL's architecture.
Key difference: TrXL has no single compact hidden state like LSTM's h_t.
Instead there's a cache of past segment activations (extractor.memory) that
the model attends over. We probe TWO representations separately:

  - features    : the extractor's output after attention -- what the actor
                  and critic heads actually condition on. Closest analog
                  to h_t in the LSTM script.
  - mem_summary : the pooled memory cache alone (memory[-1].mean(dim=1)),
                  same quantity TrXLActorCritic._get_critic_features() uses.
                  A purer view of "what's in the cache" vs. "what the model
                  is doing with current input + cache combined".

If features decode a target well but mem_summary doesn't, the model is
leaning on immediate context rather than genuinely cached history -- a
distinction the single-h_t LSTM probe can't separate as cleanly.

IMPORTANT: extractor.forward() is stateful (it advances the memory cache on
every call). Do NOT call agent.get_action_and_value() here -- it calls the
extractor internally, and calling it a second time per step would advance
the cache twice per real timestep and corrupt it. This script calls the
extractor exactly once per step and derives the action manually.
"""

import argparse
import numpy as np
import torch
from torch.distributions import Categorical

from train.probe_belief_state import (
    LinearProbe, train_probe, episode_split, r_squared,
)
from train.trxl_train_single_agent import (  # rename to match your actual TrXL script's module name
    TrXLActorCritic, single_obs_to_tensor, make_env_fn,
)


# 1. Data collection (TrXL-specific: single extractor call per step,
#    memory reset per episode, capture both features and mem_summary)
#    -- see sanity_check_probe_data() below for validating this output
#       before trusting any downstream R^2 numbers.
# 
 
def collect_probe_data_trxl(agent, cfg, device, n_episodes: int, target_keys: list):
    env = make_env_fn(cfg, rank=98)()
    agent.eval()
 
    feat_list, mem_list, obs_list = [], [], []
    target_lists = {k: [] for k in target_keys}
    episode_ids  = []
 
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done   = False
 
        # Full memory reset, matching the TrXL script's evaluate()
        agent.extractor.memory           = None
        agent.extractor._segment_hiddens = None
 
        while not done:
            obs_t = single_obs_to_tensor(obs, device)
 
            with torch.no_grad():
                features = agent.extractor(obs_t)                    # single call, advances cache
                logits_list = [head(features) for head in agent.actor_heads]
                dists  = [Categorical(logits=l) for l in logits_list]
                action = torch.stack([d.sample() for d in dists], dim=1)
 
                if agent.extractor.memory is not None:
                    mem_summary = agent.extractor.memory[-1].mean(dim=1)  # (1, d_model)
                else:
                    mem_summary = torch.zeros_like(features)
 
            feat_list.append(features.squeeze(0).cpu().numpy())
            mem_list.append(mem_summary.squeeze(0).cpu().numpy())
            obs_flat = np.concatenate([np.asarray(v).flatten() for v in obs.values()])
            obs_list.append(obs_flat)
            episode_ids.append(ep)
 
            obs, reward, terminated, truncated, info = env.step(
                action.squeeze(0).cpu().numpy()
            )
            for k in target_keys:
                target_lists[k].append(info.get(k, np.nan))
 
            done = terminated or truncated
 
    env.close()
 
    return {
        "features":   np.stack(feat_list),   # (N, features_dim)  -- current + attended history
        "mem":        np.stack(mem_list),    # (N, features_dim)  -- pooled cache alone
        "obs":        np.stack(obs_list),    # (N, obs_dim)       -- baseline, no memory at all
        "episode_id": np.array(episode_ids),
        "targets":    {k: np.array(v, dtype=np.float32) for k, v in target_lists.items()},
    }
 
 
# 
# 2. Sanity checks -- run BEFORE trusting any probe R^2 number.
#    A probe hitting R^2 ~= 1.000 on the obs-only baseline is almost never
#    a genuine result; it's usually one of the three issues below. These
#    checks print warnings but don't stop execution, since a positive
#    real result can occasionally look similar -- use judgement, but treat
#    any WARNING here as a reason to distrust the R^2 numbers until fixed.
# 
 
def sanity_check_probe_data(data: dict, target_keys: list, obs_corr_threshold: float = 0.98):
    print("\n[PROBE-TrXL] Running sanity checks on collected data...")
    warnings_raised = False
 
    # --- Check 1: duplicate / near-identical episodes ---
    # If env resets deterministically (e.g. cfg.seed fixed, no randomized
    # start state), "held-out" val episodes may just be replays of train
    # episodes, making any representation look like it generalizes.
    episode_ids = data["episode_id"]
    unique_eps  = np.unique(episode_ids)
    first_obs_hashes = {}
    dup_groups = {}
    for ep in unique_eps:
        ep_mask  = episode_ids == ep
        ep_obs   = data["obs"][ep_mask]
        ep_len   = int(ep_mask.sum())
        key      = (hash(ep_obs[0].tobytes()), ep_len)
        dup_groups.setdefault(key, []).append(int(ep))
        first_obs_hashes[int(ep)] = key
 
    n_dup_groups = sum(1 for eps in dup_groups.values() if len(eps) > 1)
    if n_dup_groups > 0:
        warnings_raised = True
        dup_examples = [eps for eps in dup_groups.values() if len(eps) > 1][:3]
        print(
            f"  [WARNING] {n_dup_groups} group(s) of episodes share an identical "
            f"first observation + length (likely deterministic replays). "
            f"Example groups: {dup_examples}. "
            f"If train/val episodes fall in the same group, val R^2 is inflated "
            f"by memorization, not generalization. Randomize env start state "
            f"(spawn position, ignition point, seed) across probe episodes."
        )
    else:
        print(f"  [OK] All {len(unique_eps)} episodes have distinct start states.")
 
    # --- Check 2: degenerate / near-constant targets ---
    # A target with ~zero variance can make R^2 spuriously collapse to ~1
    # regardless of whether the probe learned anything real, because both
    # the residual and total sum-of-squares shrink toward zero together.
    for k in target_keys:
        y = data["targets"][k]
        y_valid = y[~np.isnan(y)]
        if len(y_valid) == 0:
            warnings_raised = True
            print(f"  [WARNING] target '{k}': all values are NaN -- info key is never populated. "
                  f"Check the env is actually setting info['{k}'] on every step.")
            continue
 
        std      = float(np.std(y_valid))
        n_unique = len(np.unique(y_valid))
        print(f"  [INFO] target '{k}': mean={np.mean(y_valid):.3f}  std={std:.3f}  "
              f"n_unique={n_unique}  n_valid={len(y_valid)}/{len(y)}")
 
        if std < 1e-3 or n_unique <= 2:
            warnings_raised = True
            print(
                f"  [WARNING] target '{k}' has near-zero variance (std={std:.4f}, "
                f"{n_unique} unique values). R^2 is not meaningful here -- fix the "
                f"env's tracking/logging for this target before trusting any probe result."
            )
 
    # --- Check 3: target trivially reconstructable from raw obs ---
    # High correlation between a target and ANY single raw-obs dimension is
    # a strong sign the "privileged" variable isn't actually privileged --
    # e.g. the fire is still directly visible when you're scoring
    # "last seen position", so you're testing perception, not memory.
    for k in target_keys:
        y = data["targets"][k]
        valid = ~np.isnan(y)
        if valid.sum() < 2:
            continue
        y_valid   = y[valid]
        obs_valid = data["obs"][valid]
        if np.std(y_valid) < 1e-6:
            continue  # already flagged by check 2
 
        # Correlate against every obs dimension, report the strongest hit
        obs_std = obs_valid.std(axis=0)
        safe    = obs_std > 1e-6
        if not safe.any():
            continue
        corrs = np.zeros(obs_valid.shape[1])
        corrs[safe] = [
            np.corrcoef(obs_valid[:, i], y_valid)[0, 1] for i in np.where(safe)[0]
        ]
        max_idx  = int(np.nanargmax(np.abs(corrs)))
        max_corr = corrs[max_idx]
 
        if abs(max_corr) >= obs_corr_threshold:
            warnings_raised = True
            print(
                f"  [WARNING] target '{k}' correlates {max_corr:+.3f} with raw obs "
                f"dim {max_idx} -- this target may be directly visible in the current "
                f"observation rather than requiring memory. If this is a 'last seen' "
                f"style target, restrict scoring to steps where the source is "
                f"currently OUT of view (e.g. mask by an is_visible flag from info)."
            )
        else:
            print(f"  [OK] target '{k}': strongest single-obs-dim correlation = {max_corr:+.3f}")
 
    if warnings_raised:
        print(
            "\n[PROBE-TrXL] One or more sanity checks failed above. Proceeding with "
            "probe training, but treat any R^2 == obs_r2 or R^2 ~= 1.000 results as "
            "suspect until these are resolved.\n"
        )
    else:
        print("[PROBE-TrXL] All sanity checks passed.\n")
 
    return warnings_raised
 
 
# 
# 3. Run probing for each target, across all three representations
# 
 
def run_probing_trxl(checkpoint_path: str, cfg, target_keys: list, n_episodes: int = 30):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
 
    probe_env   = make_env_fn(cfg, rank=97)()
    obs_space   = probe_env.observation_space
    action_nvec = probe_env.action_space.nvec.tolist()
    probe_env.close()
 
    agent = TrXLActorCritic(obs_space, action_nvec, cfg).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    agent.load_state_dict(ckpt["agent"])
    agent.eval()
 
    print(f"[PROBE-TrXL] Collecting {n_episodes} episodes from {checkpoint_path}")
    data = collect_probe_data_trxl(agent, cfg, device, n_episodes, target_keys)
 
    sanity_check_probe_data(data, target_keys)
 
    train_mask, val_mask = episode_split(data["episode_id"], val_frac=0.2)
 
    results = {}
    for k in target_keys:
        y = data["targets"][k]
        valid = ~np.isnan(y)
        tm, vm = train_mask & valid, val_mask & valid
 
        _, feat_r2 = train_probe(data["features"][tm], y[tm], data["features"][vm], y[vm], device=device)
        _, mem_r2  = train_probe(data["mem"][tm],      y[tm], data["mem"][vm],      y[vm], device=device)
        _, obs_r2  = train_probe(data["obs"][tm],      y[tm], data["obs"][vm],      y[vm], device=device)
 
        results[k] = {
            "features_probe_r2": feat_r2,
            "mem_probe_r2":      mem_r2,
            "obs_probe_r2":      obs_r2,
            "memory_gain_features": feat_r2 - obs_r2,
            "memory_gain_mem":      mem_r2 - obs_r2,
        }
        print(
            f"[PROBE-TrXL] {k:35s} | features R^2={feat_r2:.3f}  "
            f"mem R^2={mem_r2:.3f}  obs R^2={obs_r2:.3f}  "
            f"gain(features)={feat_r2 - obs_r2:+.3f}  gain(mem)={mem_r2 - obs_r2:+.3f}"
        )
 
    return results



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Belief-state probing for TrXL checkpoints")
    parser.add_argument("-c", "--checkpoint", type=str, required=True)
    parser.add_argument("-n", "--n_episodes", type=int, default=30)
    parser.add_argument(
        "-t", "--targets", nargs="+",
        default=[
            "probe/fire_last_seen_x",
            "probe/fire_last_seen_y",
            "probe/steps_since_revisit",
        ],
    )
    args = parser.parse_args()

    from config.Config import EnvConfig
    cfg = EnvConfig()  # fill in / load whatever config the checkpoint was trained with

    run_probing_trxl(args.checkpoint, cfg, args.targets, n_episodes=args.n_episodes)