
from envs.MultiAgentEnv import MultiAgentEnv
from policies import ResNetActorCriticModel, TemporalTransformerModel

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

import torch.nn as nn
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback






N_ENVS = 4


def make_env(rank):
    def _init():
        return MultiAgentEnv(
            n_agents=1,
            world_size=(512, 512),
            start_positions=[(128, 128), (256, 128), (128, 256), (256, 256)],
            render_mode="human",  # only rank 0 renders
            sample_interval=200 if rank == 0 else 999999,
            save_interval=200 if rank == 0 else 999999,
            seed=34,
            fixed_seed=False,
            is_vid_out=True if rank == 0 else False,
            iter_limit=500,
            vid_id="no_swarming_global_reward",
            vid_base_path="./vids/"
        )
    return _init

 





class MemoryResetCallback(BaseCallback):
    def _on_step(self):
        # Check if any env just reset (done=True)
        for i, done in enumerate(self.locals["dones"]):
            if done:
                policy = self.model.policy
                policy.features_extractor.reset_memory()
        return True



checkpoint_cb = CheckpointCallback(
    save_freq=50_000,            # save every 50k timesteps
    save_path="./checkpoints/",
    name_prefix="drone_ppo",
    save_vecnormalize=True,      # saves the VecNormalize stats too
)

# Separate eval env to measure true performance without exploration noise
eval_env = DummyVecEnv([lambda: MultiAgentEnv(
    n_agents=4,
    world_size=(512, 512),
    start_positions=[(128, 128), (256, 128), (128, 256), (256, 256)],
    render_mode="human",
    sample_interval=999999,      # suppress rendering during eval
)])
eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False, training=False)

eval_cb = EvalCallback(
    eval_env,
    best_model_save_path="./best_model/",
    log_path="./logs/",
    eval_freq=50_000,            # evaluate every 50k timesteps
    n_eval_episodes=10,          # average over 10 episodes
    deterministic=True,          # no exploration during eval
)


























env = DummyVecEnv(
    [lambda: MultiAgentEnv(
            n_agents=1,
            world_size=(512, 512),
            start_positions=[(256, 256), (256, 128), (128, 256), (256, 256)],  # fixed grid
            render_mode="human",
            sample_interval=200,
            save_interval=200,
            seed=34,
            fixed_seed=False,
            is_vid_out=True,
            iter_limit=500,
            vid_id="no_swarming_global_reward",
            vid_base_path="/home/gjs/software/thesis/swarmfire/vids/"
        )
    ]
)

resnet_policy_kwargs = dict(
    features_extractor_class = ResNetActorCriticModel.FireScoutExtractor,
    features_extractor_kwargs=dict(
        n_agents=1,
        cnn_out_dim=256,
        pos_out_dim=64
    ),
    net_arch=dict(
        pi=[256, 128],
        vf=[512, 256]
    ),
    activation_fn=nn.ReLU,
    normalize_images=False,  # critical — your input is already float [0,1]
)

temporal_xformer_policy_kwargs = dict(
    features_extractor_class=TemporalTransformerModel.TemporalTransformerExtractor,
    features_extractor_kwargs=dict(
        features_dim=256,
        n_heads=4,
        n_layers=3,
        memory_len=8,   # how many past frames to attend over
    ),
)


model = PPO(
    "MultiInputPolicy",
    env,
    policy_kwargs=temporal_xformer_policy_kwargs,
    verbose=1,
    learning_rate=1e-4,
    batch_size=256,
    n_steps=2048 // N_ENVS,
    n_epochs=10,
    gamma=0.98,
    gae_lambda=0.95,
    ent_coef=0.05,
    vf_coef=0.5,
    max_grad_norm=0.5,
    clip_range=0.2,
)


if __name__ == "__main__":
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    model = PPO(
        "MultiInputPolicy", env,
        policy_kwargs=dict(
            features_extractor_class=TemporalTransformerModel.TemporalTransformerExtractor,
            features_extractor_kwargs=dict(features_dim=256, n_heads=4, n_layers=3),
        ),
        verbose=1,
        learning_rate=1e-4,
        n_steps=512,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.05,
    )

    model.learn(total_timesteps=5_000_000, callback=MemoryResetCallback())
    model.save("./checkpoints/final_model")
    env.close()






        
        




        
