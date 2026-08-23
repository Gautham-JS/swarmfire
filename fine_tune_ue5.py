"""
Fine-tune / domain-adapt a pretrained TrXL-PPO checkpoint (trained against
the synthetic Python SingleAgentEnv) against the LIVE UE5 environment.

Structural notes (see eval_trxl_ue5.py / run_eval.py for the pattern this
mirrors):

  1. THE PYTHON PROCESS OWNS THE WEBSOCKET SERVER.
     UE5 connects INTO this process as a client (see the commented-out
     `WebsocketManager::Get()->Connect("ws://localhost:8080/")` in
     DroneParent.cpp) — Python is the server, not the client. That server
     runs on an asyncio event loop, which has to stay unblocked to service
     the connection, so it runs on the MAIN thread via
     `asyncio.run(start_eval_server(...))`. The actual fine-tuning loop
     blocks repeatedly on websocket round trips (env.step() waiting for a
     UE5 response), so it CANNOT run on the same thread as the event loop
     — it runs on a background thread instead, exactly like
     run_evaluation() does in your eval script.

     Point the UE5 client's connection string at whatever host:port you
     pass via --port here (default 8091 — deliberately different from the
     eval script's 8090, so you can run an eval session and a fine-tune
     session without them fighting over the same port if both processes
     happen to be up at once).

  2. SINGLE ENV ONLY.
     UE5MapManager (MapManagerSingleton metaclass) and WSCommsHandler
     (WSCommsHandler.instance()) are both process-wide singletons — only
     one live UE5 connection can exist per process. Do NOT wrap this in
     SubprocVecEnv. DummyVecEnv (in-process, n_envs=1) is used purely to
     keep the existing (n_steps, n_envs) buffer plumbing unchanged.

  3. NO SEPARATE EVAL ENVIRONMENT DURING FINE-TUNING.
     A second env instance in this same process would share the same
     singleton comms handler as the training env, so it would be driving
     the same live drone as training at the same time. Model selection
     here uses a rolling mean of TRAINING episode reward instead of a
     held-out evaluate() call. If you want a true held-out eval, run your
     existing eval_trxl_ue5.run_evaluation() as a wholly separate process
     (own server, own port) against a checkpoint saved by this script.

  4. is_gt_visible MUST be False.
     SingleAgentEnv.render() calls agent_instance.get_GT_map() when True;
     UE5Agent.get_GT_map() raises NotImplementedError unconditionally.

  5. OBSERVATION-SHAPE PARITY WITH THE ORIGINAL TRAINING RUN IS ON YOU.
     world_size, vp_size, and is_recency_obs_disabled (see
     OBS_MUST_MATCH_TRAINING below) must match the checkpoint's original
     training config exactly, or the network's input shape won't line up.

  6. REAL UE5 STEP LATENCY IS FAR HIGHER THAN THE SYNTHETIC SIM.
     n_steps / total_timesteps default much smaller than a full training
     run — this is a short domain-adaptation pass on an already-competent
     policy, not training from scratch.

Reuses the exact model / buffer / helper classes from the original
multi-env trainer so the network architecture matches the checkpoint.
Adjust MULTIENV_TRAIN_MODULE (the import below) to wherever you saved
that script.
"""

import os
import time
import argparse
import random
import threading
import asyncio
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.WildfireSingleAgentEnv import SingleAgentEnv
from envs.RedisSingleAgentEnv import RedisRenderedEnv
from config.Config import VideoWriterConfig, EnvConfig
from comms.web_sockets.server import handle_client, WSCommsHandler, start_eval_server

# reuse whatever I used from training harness for eval
from train.trxl_train_single_agent import (
    TrXLActorCritic,
    TrXLRolloutBuffer,
    RunningMeanStd,
    vec_obs_to_tensor,
)


import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


# MUST match whatever the checkpoint was originally trained with. The
# multi-env trainer's make_env_fn hardcoded is_recency_obs_disabled=True,
# which feeds directly into observation_space shape.
OBS_MUST_MATCH_TRAINING = dict(
    is_recency_obs_disabled=False,
)


def make_ue5_env_fn(cfg: EnvConfig, video_dir: str, device: torch.device, is_eval_mode: bool = False):
    """Single UE5-backed env thunk - see module docstring, point 2."""
    video_config = VideoWriterConfig(
        is_enabled=True,
        sample_interval=1,
        save_interval=1,
        base_path=video_dir,
    )

    def _init():
        env = RedisRenderedEnv(
            world_size=cfg.world_size,                 # MUST match original training
            render_mode="rgb_array",
            seed=cfg.seed,
            video_conf=video_config,
            phase_weights=cfg.phase_weights,
            env_id=f"{cfg.run_id}_ue5_finetune",
            vp_size=cfg.vp_size,                        # MUST match original training
            is_gt_visible=False,                        # REQUIRED — module docstring, point 4
            is_recency_obs_disabled=OBS_MUST_MATCH_TRAINING["is_recency_obs_disabled"],
            is_ue5_mode=True,
            is_eval_mode=is_eval_mode,
            device=device,
            redis_host="localhost",
            redis_port=8090,
            redis_channel_prefix="ue5_train_env"
        )
        return TimeLimit(env, max_episode_steps=cfg.iter_limit)

    return _init


def freeze_extractor_backbone(agent: TrXLActorCritic, unfreeze_last_n_layers: int = 0):
    """
    Optionally freezes the TrXL feature extractor so fine-tuning only
    updates the actor/critic heads + critic aggregator.

    NOTE: only freezes/unfreezes at whole-extractor granularity.
    TrXLExtractor's internal layer-stack attribute name isn't visible in
    what's been shared so far, so precise last-N-layer unfreezing isn't
    wired up — tell me the attribute name (e.g. self.layers / self.blocks)
    if you want that.
    """
    for p in agent.extractor.parameters():
        p.requires_grad = False

    if unfreeze_last_n_layers > 0:
        logging.warning(
            "[FREEZE] unfreeze_last_n_layers > 0 requested but "
            "TrXLExtractor's layer attribute name is unknown here — "
            "extractor is fully frozen instead."
        )

    for p in agent.actor_heads.parameters():
        p.requires_grad = True
    for p in agent.critic_head.parameters():
        p.requires_grad = True
    for p in agent.critic_aggregator.parameters():
        p.requires_grad = True

    n_trainable = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in agent.parameters())
    logging.info(f"[FREEZE] Trainable params: {n_trainable:,} / {n_total:,}")


def run_finetuning(checkpoint_path: str, cfg: EnvConfig, args: argparse.Namespace):
    """
    Runs on a background thread (see module docstring, point 1) while the
    main thread hosts the websocket server UE5 connects into. Mirrors
    run_evaluation()'s role in your eval script.
    """
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.best_model_dir, exist_ok=True)
    os.makedirs(args.video_dir, exist_ok=True)

    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    r_count = 0
    while not WSCommsHandler.instance().is_clients_connected():
        logging.info(
            f"[INIT] [WAIT] Waiting for UE5 client to connect, n_retries : {r_count}..."
        )
        time.sleep(1)
        r_count+=1
    logging.info("[INIT] [OK] : Client connection established.")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logging.info(f"[INIT] Device: {device} | UE5 fine-tune (n_envs=1, forced)")

    envs = DummyVecEnv([make_ue5_env_fn(cfg, args.video_dir, device, is_eval_mode=False)])
    n_envs = 1
    obs_dict = envs.reset()

    obs_space = envs.observation_space
    action_nvec = envs.action_space.nvec.tolist()

    agent = TrXLActorCritic(obs_space, action_nvec, cfg).to(device)

    logging.info(f"[INIT] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    agent.load_state_dict(ckpt["agent"])

    if args.freeze_extractor:
        freeze_extractor_backbone(agent, unfreeze_last_n_layers=args.unfreeze_last_n_layers)

    trainable_params = [p for p in agent.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=args.learning_rate, eps=1e-5)

    if args.resume_optimizer_state and "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
            logging.info("[INIT] Restored optimizer state from checkpoint.")
        except ValueError as e:
            logging.warning(f"[INIT] Could not restore optimizer state ({e}); starting fresh.")

    global_step = 0
    best_train_reward = -np.inf
    reward_rms = RunningMeanStd()  # fresh — UE5 reward statistics are a new distribution

    buffer = TrXLRolloutBuffer(
        n_steps=args.n_steps,
        n_envs=n_envs,
        obs_space=obs_space,
        action_nvec=action_nvec,
        n_layers=cfg.n_layers,
        memory_len=cfg.memory_len,
        d_model=cfg.features_dim,
        device=device,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
    )

    ep_rewards = np.zeros(n_envs, dtype=np.float32)
    ep_lengths = np.zeros(n_envs, dtype=np.int32)
    recent_rewards = deque(maxlen=args.reward_window)

    agent.extractor.init_memory(batch_size=n_envs, device=device)

    logging.info(
        f"[FINETUNE] Starting — {args.total_timesteps:,} steps | "
        f"rollout size = {args.n_steps * n_envs:,} transitions | "
        f"lr={args.learning_rate}"
    )
    start_time = time.time()

    logging.info(f"[FINETUNE] : Total Trainable Parameters : {len(trainable_params)}")

    while global_step < args.total_timesteps:

        agent.eval()
        for step in range(args.n_steps):
            obs_t = vec_obs_to_tensor(obs_dict, device)

            with torch.no_grad():
                memory_snapshot = (
                    [m.clone() for m in agent.extractor.memory]
                    if agent.extractor.memory is not None else None
                )
                actions, log_probs, _, values = agent.get_action_and_value(obs_t)

            actions_np = actions.cpu().numpy()
            values_np = values.squeeze(-1).cpu().numpy()
            log_probs_np = log_probs.cpu().numpy()

            # Blocks here waiting on a UE5 websocket round trip — this is
            # exactly why this function must run off the asyncio thread.
            next_obs_dict, rewards, dones, infos = envs.step(actions_np)

            reward_rms.update(rewards)
            norm_rewards = reward_rms.normalise(rewards)

            buffer.add_step(
                step=step,
                obs_dict=obs_dict,
                actions=actions_np,
                rewards=norm_rewards,
                dones=dones.astype(np.float32),
                values=values_np,
                log_probs=log_probs_np,
                memory=memory_snapshot,
            )

            obs_dict = next_obs_dict
            global_step += n_envs
            ep_rewards += rewards
            ep_lengths += 1

            done_envs = np.where(dones)[0]
            for env_idx in done_envs:
                recent_rewards.append(float(ep_rewards[env_idx]))
                mean_reward = np.mean(recent_rewards) if recent_rewards else 0.0
                logging.info(
                    f"[EPISODE] step={global_step} reward={ep_rewards[env_idx]:.2f} "
                    f"length={int(ep_lengths[env_idx])} rolling_mean={mean_reward:.2f}"
                )
                agent.extractor.reset_memory([env_idx])
                ep_rewards[env_idx] = 0.0
                ep_lengths[env_idx] = 0

        with torch.no_grad():
            obs_t = vec_obs_to_tensor(obs_dict, device)
            last_values = agent.get_value(obs_t).squeeze(-1).cpu().numpy()
        buffer.compute_gae(last_values=last_values, last_dones=dones)

        agent.train()
        policy_losses, value_losses, entropies, kl_divs = [], [], [], []
        stop_early = False

        for epoch in range(args.n_epochs):
            if stop_early:
                break
            epoch_kls = []

            for (obs_b, actions_b, old_log_probs_b,
                 advantages_b, returns_b, old_values_b,
                 memory_b) in buffer.get_minibatches(args.batch_size):

                advantages_b = (advantages_b - advantages_b.mean()) / (advantages_b.std() + 1e-8)

                _, new_log_probs, entropy, new_values = agent.get_action_and_value(
                    obs_b, action=actions_b, memory_override=memory_b,
                )
                new_values = new_values.squeeze(-1)

                log_ratio = new_log_probs - old_log_probs_b
                ratio = log_ratio.exp()
                approx_kl = ((ratio - 1) - log_ratio).mean().item()

                pg_loss1 = -advantages_b * ratio
                pg_loss2 = -advantages_b * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_clipped = old_values_b + torch.clamp(new_values - old_values_b, -cfg.clip_coef, cfg.clip_coef)
                vf_loss1 = (new_values - returns_b).pow(2)
                vf_loss2 = (v_clipped - returns_b).pow(2)
                value_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()

                entropy_loss = entropy.mean()
                loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(trainable_params, cfg.max_grad_norm)
                optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy_loss.item())
                kl_divs.append(approx_kl)
                epoch_kls.append(approx_kl)

                if approx_kl > cfg.target_kl * 1.5:
                    logging.info(
                        f"[PPO] Early stop mid-epoch {epoch + 1}, "
                        f"minibatch KL={approx_kl:.4f} (> {cfg.target_kl * 1.5:.4f})"
                    )
                    stop_early = True
                    break

            if not stop_early and epoch_kls and np.mean(epoch_kls) > cfg.target_kl:
                logging.info(f"[PPO] Early stop at epoch {epoch + 1}, KL={np.mean(epoch_kls):.4f}")
                stop_early = True

        agent.extractor.memory = [m.detach() for m in agent.extractor.memory]
        if agent.extractor._segment_hiddens is not None:
            agent.extractor._segment_hiddens = [h.detach() for h in agent.extractor._segment_hiddens]

        elapsed = time.time() - start_time
        sps = global_step / elapsed if elapsed > 0 else 0
        mean_pl = np.mean(policy_losses) if policy_losses else 0.0
        mean_vl = np.mean(value_losses) if value_losses else 0.0
        mean_ent = np.mean(entropies) if entropies else 0.0
        mean_kl = np.mean(kl_divs) if kl_divs else 0.0

        logging.info(
            f"[{global_step:>8}] pl={mean_pl:.4f} vl={mean_vl:.4f} "
            f"ent={mean_ent:.4f} kl={mean_kl:.4f} sps={sps:.2f}"
        )

        # Model selection on rolling training reward — module docstring, point 3.
        if recent_rewards:
            current_mean = float(np.mean(recent_rewards))
            if current_mean > best_train_reward:
                best_train_reward = current_mean
                torch.save({
                    "agent": agent.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "global_step": global_step,
                    "best_train_reward": best_train_reward,
                    "base_checkpoint": checkpoint_path,
                }, os.path.join(cfg.best_model_dir, "best_model_ue5_finetuned.pt"))
                logging.info(f"[CKPT] New best rolling-train-reward model: {best_train_reward:.3f}")

        if global_step % args.checkpoint_freq < n_envs * args.n_steps:
            ckpt_path = os.path.join(cfg.checkpoint_dir, f"ue5_finetune_{global_step}_steps.pt")
            torch.save({
                "agent": agent.state_dict(),
                "optimizer": optimizer.state_dict(),
                "global_step": global_step,
                "best_train_reward": best_train_reward,
                "base_checkpoint": checkpoint_path,
            }, ckpt_path)
            logging.info(f"[CKPT] Saved: {ckpt_path}")

    torch.save({
        "agent": agent.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": global_step,
        "best_train_reward": best_train_reward,
        "base_checkpoint": checkpoint_path,
    }, args.final_checkpoint_path)
    logging.info(f"[DONE] Fine-tuning complete. Saved to {args.final_checkpoint_path}")
    envs.close()


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune a pretrained TrXL-PPO checkpoint against live UE5")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to the pretrained .pt checkpoint")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--port", type=int, default=8091,
                   help="Port the websocket server listens on — point UE5's connection at this (module docstring, point 1)")
    p.add_argument("--learning-rate", type=float, default=1e-5,
                   help="Lower than the original training LR by default — fine-tuning, not training from scratch")
    p.add_argument("--n-steps", type=int, default=128,
                   help="Rollout length per update; kept small since each step is a real websocket round trip")
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--total-timesteps", type=int, default=20_000)
    p.add_argument("--checkpoint-freq", type=int, default=2_000)
    p.add_argument("--reward-window", type=int, default=10,
                   help="Episodes averaged for the rolling best-model signal")
    p.add_argument("--freeze-extractor", action="store_true",
                   help="Freeze the TrXL backbone, fine-tune only actor/critic heads")
    p.add_argument("--unfreeze-last-n-layers", type=int, default=0,
                   help="Not yet wired up precisely — see freeze_extractor_backbone() docstring")
    p.add_argument("--resume-optimizer-state", action="store_true",
                   help="Restore optimizer state from the checkpoint (usually NOT what you want when freezing params)")
    p.add_argument("--video-dir", type=str, default="./vids_ue5_finetune/")
    p.add_argument("--final-checkpoint-path", type=str, default="./ue5_finetuned_final.pt")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # =========================================================================
    # IMPORTANT: same as your eval script — this should be the same EnvConfig
    # used during original training. Replace this with however your project
    # actually constructs EnvConfig if it's not just the bare default.
    # =========================================================================
    cfg = EnvConfig()
    cfg.run_id = f"ue5_finetune_{int(time.time())}"

    # Fine-tuning loop blocks on websocket round trips -> runs on a
    # background thread. The main thread hosts the asyncio server UE5
    # connects into, exactly mirroring run_eval.py's structure.
    t = threading.Thread(target=run_finetuning, args=(args.checkpoint, cfg, args))
    t.start()

    asyncio.run(start_eval_server("0.0.0.0", args.port))