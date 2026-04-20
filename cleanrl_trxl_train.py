"""
CleanRL-style PPO with TrXL memory snapshot support.

Key difference from SB3: memory snapshots are stored in the rollout buffer
at collection time and replayed during evaluate_actions, so gradients are
computed on features that match what the policy actually experienced.
"""

import os
import time
import argparse
import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import wandb

from gymnasium.wrappers import TimeLimit
from envs.SingleAgentEnv import SingleAgentEnv
from policies.TrXL import TrXLExtractor


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # Environment
    world_size:       tuple = (512, 512)
    n_agents:         int   = 1
    iter_limit:       int   = 500
    seed:             int   = 34

    # TrXL
    features_dim:     int   = 256
    memory_len:       int   = 128
    n_layers:         int   = 2
    n_heads:          int   = 4
    d_ff_multiplier:  int   = 2
    dropout:          float = 0.1

    # PPO
    total_timesteps:  int   = 5_000_000
    n_steps:          int   = 512        # steps collected per rollout
    batch_size:       int   = 256        # minibatch size for PPO update
    n_epochs:         int   = 10         # PPO update epochs per rollout
    learning_rate:    float = 1e-4
    gamma:            float = 0.99
    gae_lambda:       float = 0.95
    clip_coef:        float = 0.2        # PPO clip epsilon
    ent_coef:         float = 0.0001
    vf_coef:          float = 0.5
    max_grad_norm:    float = 0.3
    target_kl:        float = 0.03       # early stop if KL exceeds this

    # Checkpointing
    checkpoint_freq:  int   = 50_000
    checkpoint_dir:   str   = "./checkpoints"
    best_model_dir:   str   = "./best_model"

    # Evaluation
    eval_freq:        int   = 50_000
    n_eval_episodes:  int   = 5

    # WandB
    wandb_project:    str   = "thesis-drl"
    wandb_api_key:    str   = "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"


# ─────────────────────────────────────────────────────────────────────────────
# Actor-Critic policy
# ─────────────────────────────────────────────────────────────────────────────

class TrXLActorCritic(nn.Module):
    """
    Wraps TrXLExtractor with separate actor and critic heads.
    Both heads share the same extractor (and therefore the same memory).
    """

    def __init__(self, observation_space, action_nvec, cfg: Config):
        super().__init__()

        self.extractor = TrXLExtractor(
            observation_space,
            features_dim    = cfg.features_dim,
            memory_len      = cfg.memory_len,
            n_layers        = cfg.n_layers,
            n_heads         = cfg.n_heads,
            d_ff_multiplier = cfg.d_ff_multiplier,
            dropout         = cfg.dropout,
        )

        # action_nvec: e.g. [3, 3] for MultiDiscrete([3, 3])
        self.action_nvec = action_nvec

        # Separate linear heads per action dimension
        self.actor_heads = nn.ModuleList([
            nn.Linear(cfg.features_dim, n) for n in action_nvec
        ])
        self.critic_head = nn.Linear(cfg.features_dim, 1)

        # Weight init — conservative gain for transformer stability
        for head in self.actor_heads:
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.critic_head.bias)

    def get_value(self, obs):
        features = self.extractor(obs)
        return self.critic_head(features)

    def get_action_and_value(self, obs, action=None):
        features = self.extractor(obs)

        # One Categorical distribution per action dimension
        logits_list = [head(features) for head in self.actor_heads]
        dists       = [Categorical(logits=logits) for logits in logits_list]

        if action is None:
            # Sample mode (collection)
            action = torch.stack([d.sample() for d in dists], dim=1)

        # action shape: (B, n_dims)
        log_prob = sum(d.log_prob(action[:, i]) for i, d in enumerate(dists))
        entropy  = sum(d.entropy() for d in dists)
        value    = self.critic_head(features)

        return action, log_prob, entropy, value


# ─────────────────────────────────────────────────────────────────────────────
# Rollout buffer with memory snapshot storage
# ─────────────────────────────────────────────────────────────────────────────

class TrXLRolloutBuffer:
    """
    Stores transitions AND per-step TrXL memory snapshots.

    During collection:  add_step() saves the memory state BEFORE each forward pass.
    During PPO update:  get_minibatches() yields (transition, memory_snapshot) pairs
                        so evaluate_actions sees the same context as collection.

    This is the fix that SB3 cannot do cleanly — it's why TrXL failed there.
    """

    def __init__(self, n_steps, obs_space, action_nvec,
                 n_layers, memory_len, d_model, device, gamma, gae_lambda):
        self.n_steps    = n_steps
        self.n_layers   = n_layers
        self.memory_len = memory_len
        self.d_model    = d_model
        self.device     = device
        self.gamma      = gamma
        self.gae_lambda = gae_lambda
        self.action_dims = len(action_nvec)

        self.ptr  = 0
        self.full = False

        # Observation buffers — Dict obs: one tensor per key
        self.obs_keys = list(obs_space.spaces.keys())
        self.obs_bufs = {
            k: np.zeros((n_steps, *obs_space.spaces[k].shape), dtype=np.float32)
            for k in self.obs_keys
        }

        # Transition buffers
        self.actions   = np.zeros((n_steps, len(action_nvec)), dtype=np.int64)
        self.rewards   = np.zeros(n_steps, dtype=np.float32)
        self.dones     = np.zeros(n_steps, dtype=np.float32)
        self.values    = np.zeros(n_steps, dtype=np.float32)
        self.log_probs = np.zeros(n_steps, dtype=np.float32)

        # GAE targets (filled after collection)
        self.advantages = np.zeros(n_steps, dtype=np.float32)
        self.returns    = np.zeros(n_steps, dtype=np.float32)

        # Memory snapshots: list of n_layers tensors, each (n_steps, memory_len, d_model)
        # Stored on CPU to avoid filling GPU memory across the full rollout
        self.memory_snapshots = [
            torch.zeros(n_steps, memory_len, d_model, dtype=torch.float32)
            for _ in range(n_layers)
        ]

    def add_step(self, obs, action, reward, done, value, log_prob, memory):
        """
        Store one transition. memory is the extractor's state BEFORE this step's
        forward pass — i.e. the context that produced the action.
        """
        for k in self.obs_keys:
            self.obs_bufs[k][self.ptr] = obs[k]

        self.actions[self.ptr]   = action
        self.rewards[self.ptr]   = reward
        self.dones[self.ptr]     = done
        self.values[self.ptr]    = value
        self.log_probs[self.ptr] = log_prob

        # Store memory snapshot — detach and move to CPU
        if memory is not None:
            for layer_idx, layer_mem in enumerate(memory):
                # layer_mem: (1, memory_len, d_model) for single env
                self.memory_snapshots[layer_idx][self.ptr] = layer_mem[0].detach().cpu()

        self.ptr = (self.ptr + 1) % self.n_steps
        self.full = self.full or (self.ptr == 0)

    def compute_gae(self, last_value, last_done):
        """Generalised Advantage Estimation over the stored rollout."""
        last_gae = 0.0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_value        = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value        = self.values[t + 1]

            delta              = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            last_gae           = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values

    def get_minibatches(self, batch_size):
        """
        Yield minibatches of (obs_dict, actions, log_probs, advantages,
        returns, values, memory_snapshots) for the PPO update.

        Crucially, memory_snapshots are the STORED states from collection,
        not zeros — this is what fixes the evaluate_actions mismatch.
        """
        indices = np.random.permutation(self.n_steps)

        for start in range(0, self.n_steps, batch_size):
            idx = indices[start : start + batch_size]

            obs_batch = {
                k: torch.tensor(self.obs_bufs[k][idx], dtype=torch.float32).to(self.device)
                for k in self.obs_keys
            }
            memory_batch = [
                self.memory_snapshots[layer][idx].to(self.device)
                for layer in range(self.n_layers)
            ]

            yield (
                obs_batch,
                torch.tensor(self.actions[idx],   dtype=torch.long).to(self.device),
                torch.tensor(self.log_probs[idx], dtype=torch.float32).to(self.device),
                torch.tensor(self.advantages[idx],dtype=torch.float32).to(self.device),
                torch.tensor(self.returns[idx],   dtype=torch.float32).to(self.device),
                torch.tensor(self.values[idx],    dtype=torch.float32).to(self.device),
                memory_batch,
            )

    def reset(self):
        self.ptr  = 0
        self.full = False



class RunningMeanStd:
    """Tracks running mean and variance for normalisation."""
    def __init__(self, epsilon=1e-4):
        self.mean  = 0.0
        self.var   = 1.0
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x)
        batch_var  = np.var(x)
        batch_count = 1

        total = self.count + batch_count
        self.mean  = (self.count * self.mean + batch_count * batch_mean) / total
        self.var   = (self.count * self.var  + batch_count * batch_var
                      + (batch_mean - self.mean)**2
                      * self.count * batch_count / total) / total
        self.count = total

    def normalise(self, x, clip=10.0):
        normed = (x - self.mean) / (np.sqrt(self.var) + 1e-8)
        return np.clip(normed, -clip, clip)


# ─────────────────────────────────────────────────────────────────────────────
# Environment helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_env(cfg: Config, is_eval=False):
    env = SingleAgentEnv(
        n_agents        = cfg.n_agents,
        world_size      = cfg.world_size,
        start_positions = [(256, 256)],
        render_mode     = "rgb_array",
        sample_interval = 50,
        save_interval   = 50,
        seed            = cfg.seed,
        fixed_seed      = False,
        is_vid_out      = True,
        vid_id="firescout_smooth",
        vid_base_path="/home/s3400220/swarmfire/vids",
        phase_weights   = {
            "exploration":    5.0,
            "fire_discovery": 9.8,
            "fire_tracking":  2.0,
            "risk":           5.0,
        },
    )
    return TimeLimit(env, max_episode_steps=cfg.iter_limit)


def obs_to_tensor(obs_dict, device):
    """Convert a single numpy obs dict to tensors with batch dim."""
    return {
        k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(device)
        for k, v in obs_dict.items()
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(agent, cfg: Config, device, n_episodes=5):
    """
    Run n_episodes with deterministic (argmax) actions.
    Properly resets TrXL memory at each episode boundary.
    Returns mean episode reward.
    """
    eval_env    = make_env(cfg, is_eval=True)
    ep_rewards  = []

    for _ in range(n_episodes):
        obs, _    = eval_env.reset()
        done      = False
        ep_reward = 0.0

        # Fresh memory for each eval episode
        agent.extractor.memory = None

        while not done:
            obs_t = obs_to_tensor(obs, device)
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs_t)
            action_np = action.squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, _ = eval_env.step(action_np)
            done       = terminated or truncated
            ep_reward += reward

        ep_rewards.append(ep_reward)

    eval_env.close()

    # Clear memory after eval so it doesn't bleed into training
    agent.extractor.memory = None

    return float(np.mean(ep_rewards))


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: Config, checkpoint_path=None):

    # ── Setup ─────────────────────────────────────────────────────────────────
    os.environ["WANDB_API_KEY"] = cfg.wandb_api_key
    wandb.init(project=cfg.wandb_project, config=vars(cfg))

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.best_model_dir, exist_ok=True)

    reward_rms = RunningMeanStd()

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INIT] Using device: {device}")

    torch.backends.cudnn.benchmark        = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

    # ── Environment ───────────────────────────────────────────────────────────
    env        = make_env(cfg)
    obs, _     = env.reset()
    obs_space  = env.observation_space
    action_nvec = env.action_space.nvec.tolist()   # [3, 3]

    # ── Agent ─────────────────────────────────────────────────────────────────
    agent     = TrXLActorCritic(obs_space, action_nvec, cfg).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)

    global_step     = 0
    best_eval_reward = -np.inf

    # ── Checkpoint loading ────────────────────────────────────────────────────
    if checkpoint_path is not None:
        print(f"[INIT] Loading checkpoint: {checkpoint_path}")
        ckpt        = torch.load(checkpoint_path, map_location=device)
        agent.load_state_dict(ckpt["agent"])
        optimizer.load_state_dict(ckpt["optimizer"])
        global_step = ckpt.get("global_step", 0)
        best_eval_reward = ckpt.get("best_eval_reward", -np.inf)
        print(f"[INIT] Resumed from step {global_step}")

    # ── Rollout buffer ────────────────────────────────────────────────────────
    buffer = TrXLRolloutBuffer(
        n_steps    = cfg.n_steps,
        obs_space  = obs_space,
        action_nvec = action_nvec,
        n_layers   = cfg.n_layers,
        memory_len = cfg.memory_len,
        d_model    = cfg.features_dim,
        device     = device,
        gamma      = cfg.gamma,
        gae_lambda = cfg.gae_lambda,
    )

    # ── Training state ────────────────────────────────────────────────────────
    ep_reward       = 0.0
    norm_ep_reward  = 0.0
    ep_length       = 0
    recent_rewards  = deque(maxlen=10)
    next_ckpt_step  = cfg.checkpoint_freq
    next_eval_step  = cfg.eval_freq

    # Init memory for collection
    agent.extractor.init_memory(batch_size=1, device=device)

    print(f"[TRAIN] Starting training for {cfg.total_timesteps:,} steps")
    start_time = time.time()

    # ── Main loop ─────────────────────────────────────────────────────────────
    while global_step < cfg.total_timesteps:

        # ── Rollout collection ────────────────────────────────────────────────
        agent.eval()
        buffer.reset()

        for step in range(cfg.n_steps):

            obs_t = obs_to_tensor(obs, device)

            with torch.no_grad():
                # Snapshot memory BEFORE forward pass
                # This is the context that will produce the action
                memory_snapshot = (
                    [m.clone() for m in agent.extractor.memory]
                    if agent.extractor.memory is not None else None
                )

                action, log_prob, _, value = agent.get_action_and_value(obs_t)

            action_np = action.squeeze(0).cpu().numpy()
            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            reward_rms.update(reward)
            norm_reward = reward_rms.normalise(reward)

            buffer.add_step(
                obs       = obs,
                action    = action_np,
                reward    = norm_reward,
                done      = float(done),
                value     = value.item(),
                log_prob  = log_prob.item(),
                memory    = memory_snapshot,
            )

            obs        = next_obs
            global_step += 1
            ep_reward  += reward
            norm_ep_reward += norm_reward
            ep_length  += 1

            if done:
                recent_rewards.append(ep_reward)
                mean_reward = np.mean(recent_rewards) if recent_rewards else 0.0

                wandb.log({
                    "episode/reward":      ep_reward,
                    "episode/length":      ep_length,
                    "episode/reward_normalised": norm_ep_reward,  # normalised accumulation
                    "episode/mean_reward": mean_reward,
                    "global_step":         global_step,
                })

                ep_reward = 0.0
                ep_length = 0

                # Reset TrXL memory on episode end
                agent.extractor.reset_memory([0])

                obs, _ = env.reset()

        # ── Compute GAE ───────────────────────────────────────────────────────
        with torch.no_grad():
            obs_t      = obs_to_tensor(obs, device)
            last_value = agent.get_value(obs_t).item()

        buffer.compute_gae(
            last_value = last_value,
            last_done  = done,
        )

        # ── PPO update ────────────────────────────────────────────────────────
        # Detach memory between collection and training — the snapshots in the
        # buffer are what matter now, not the live extractor memory
        if agent.extractor.memory is not None:
            agent.extractor.memory = [m.detach() for m in agent.extractor.memory]

        agent.train()

        policy_losses, value_losses, entropies, kl_divs = [], [], [], []

        for epoch in range(cfg.n_epochs):
            for (obs_b, actions_b, old_log_probs_b,
                 advantages_b, returns_b, old_values_b,
                 memory_b) in buffer.get_minibatches(cfg.batch_size):

                # ── Inject stored memory snapshot ─────────────────────────────
                # This is THE fix: the extractor sees the memory context from
                # collection, not zeros. Gradients flow through the current step
                # only, not through the stored memory (which is detached).
                agent.extractor.memory = memory_b

                # Normalise advantages within minibatch
                advantages_b = (advantages_b - advantages_b.mean()) / (advantages_b.std() + 1e-8)

                _, new_log_probs, entropy, new_values = agent.get_action_and_value(
                    obs_b, action=actions_b
                )
                new_values = new_values.squeeze(-1)

                # PPO clipped policy loss
                log_ratio    = new_log_probs - old_log_probs_b
                ratio        = log_ratio.exp()
                approx_kl    = ((ratio - 1) - log_ratio).mean().item()

                pg_loss1     = -advantages_b * ratio
                pg_loss2     = -advantages_b * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                policy_loss  = torch.max(pg_loss1, pg_loss2).mean()

                # Clipped value loss
                v_clipped    = old_values_b + torch.clamp(new_values - old_values_b, -cfg.clip_coef, cfg.clip_coef)
                vf_loss1     = (new_values  - returns_b).pow(2)
                vf_loss2     = (v_clipped   - returns_b).pow(2)
                value_loss   = 0.5 * torch.max(vf_loss1, vf_loss2).mean()

                entropy_loss = entropy.mean()
                loss         = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy_loss.item())
                kl_divs.append(approx_kl)

            # Early stop if KL too large
            mean_kl = np.mean(kl_divs[-len(list(buffer.get_minibatches(cfg.batch_size))):]) \
                      if kl_divs else 0.0
            if cfg.target_kl is not None and mean_kl > cfg.target_kl:
                print(f"[PPO] Early stop at epoch {epoch+1}, KL={mean_kl:.4f}")
                break

        # Clear extractor memory after training phase — collection resumes
        # with the live memory that was being maintained during rollout
        agent.extractor.memory = None
        agent.extractor.init_memory(batch_size=1, device=device)

        # ── Logging ───────────────────────────────────────────────────────────
        elapsed   = time.time() - start_time
        sps       = global_step / elapsed if elapsed > 0 else 0

        mean_pl   = np.mean(policy_losses) if policy_losses else 0.0
        mean_vl   = np.mean(value_losses)  if value_losses  else 0.0
        mean_ent  = np.mean(entropies)     if entropies      else 0.0
        mean_kl   = np.mean(kl_divs)       if kl_divs        else 0.0

        wandb.log({
            "train/policy_loss":    mean_pl,
            "train/value_loss":     mean_vl,
            "train/entropy":        mean_ent,
            "train/approx_kl":      mean_kl,
            "train/steps_per_sec":  sps,
            "global_step":          global_step,
        })

        print(
            f"[{global_step:>8}] "
            f"pl={mean_pl:.4f} vl={mean_vl:.4f} "
            f"ent={mean_ent:.4f} kl={mean_kl:.4f} "
            f"sps={sps:.0f}"
        )

        # ── Checkpoint ────────────────────────────────────────────────────────
        if global_step >= next_ckpt_step:
            ckpt_path = os.path.join(
                cfg.checkpoint_dir, f"firescout_{global_step}_steps.pt"
            )
            torch.save({
                "agent":            agent.state_dict(),
                "optimizer":        optimizer.state_dict(),
                "global_step":      global_step,
                "best_eval_reward": best_eval_reward,
            }, ckpt_path)
            print(f"[CKPT] Saved checkpoint → {ckpt_path}")
            next_ckpt_step += cfg.checkpoint_freq

        # ── Evaluation ────────────────────────────────────────────────────────
        if global_step >= next_eval_step:
            eval_reward = evaluate(agent, cfg, device, cfg.n_eval_episodes)
            print(f"[EVAL] step={global_step} mean_reward={eval_reward:.3f}")

            wandb.log({
                "eval/mean_reward": eval_reward,
                "global_step":      global_step,
            })

            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                best_path = os.path.join(cfg.best_model_dir, "best_model.pt")
                torch.save({
                    "agent":            agent.state_dict(),
                    "optimizer":        optimizer.state_dict(),
                    "global_step":      global_step,
                    "best_eval_reward": best_eval_reward,
                }, best_path)
                print(f"[EVAL] New best model saved (reward={best_eval_reward:.3f})")

            # Re-init memory after eval (evaluate() clears it)
            agent.extractor.init_memory(batch_size=1, device=device)
            next_eval_step += cfg.eval_freq

    # ── End of training ───────────────────────────────────────────────────────
    final_path = "./firescout_final.pt"
    torch.save({
        "agent":            agent.state_dict(),
        "optimizer":        optimizer.state_dict(),
        "global_step":      global_step,
        "best_eval_reward": best_eval_reward,
    }, final_path)
    print(f"[DONE] Training complete. Final model saved → {final_path}")

    wandb.finish()
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CleanRL TrXL PPO — FireScout")
    parser.add_argument("-c", "--checkpoint", type=str, default=None,
                        help="Path to a .pt checkpoint file to resume from")
    args = parser.parse_args()

    cfg = Config()
    train(cfg, checkpoint_path=args.checkpoint)