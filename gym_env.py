
from envs.MultiAgentEnv import MultiAgentEnv
from policies import ResNetActorCriticModel, TemporalTransformerModel

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback


N_AGENTS=1
N_STEPS=500



class MemoryResetCallback(BaseCallback):
    def _on_step(self):
        # Check if any env just reset (done=True)
        for i, done in enumerate(self.locals["dones"]):
            if done:
                policy = self.model.policy
                policy.features_extractor.reset_memory()
        return True


class TrainingMonitorCallback(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.episode_count = 0
        self.timestep_at_last_episode = 0

    def _on_step(self):
        # Count episodes from the done flags
        for done in self.locals["dones"]:
            if done:
                self.episode_count += 1
                steps_this_ep = self.num_timesteps - self.timestep_at_last_episode
                self.timestep_at_last_episode = self.num_timesteps
                if self.verbose and self.episode_count % 50 == 0:
                    print(
                        f"Episode {self.episode_count} | "
                        f"Timestep {self.num_timesteps} | "
                        f"Steps this ep: {steps_this_ep}"
                    )
        return True

    def _on_training_end(self):
        print(f"\nTraining ended after {self.episode_count} episodes "
              f"and {self.num_timesteps} timesteps")


# Separate eval env to measure true performance without exploration noise
eval_env = DummyVecEnv([lambda: MultiAgentEnv(
    n_agents=N_AGENTS,
    world_size=(512, 512),
    start_positions=[(128, 128), (256, 128), (128, 256), (256, 256)],
    render_mode="human",
    sample_interval=999999,      # suppress rendering during eval,
    iter_limit=N_STEPS
)])
eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False, training=False)

env = DummyVecEnv(
    [lambda: MultiAgentEnv(
            n_agents=N_AGENTS,
            world_size=(512, 512),
            start_positions=[(256, 256), (256, 128), (128, 256), (256, 256)],  # fixed grid
            render_mode="human",
            sample_interval=100,
            save_interval=100,
            seed=34,
            fixed_seed=False,
            is_vid_out=True,
            iter_limit=N_STEPS,
            vid_id="no_swarming_global_reward",
            vid_base_path="/home/gjs/software/thesis/swarmfire/vids/"
        )
    ]
)
env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)







import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy

resnet_policy_kwargs = dict(
    features_extractor_class = ResNetActorCriticModel.FireScoutExtractor,
    features_extractor_kwargs=dict(
        n_agents=N_AGENTS,
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
    n_steps=2048,
    n_epochs=10,
    gamma=0.98,
    gae_lambda=0.95,
    ent_coef=0.05,
    vf_coef=0.5,
    max_grad_norm=0.5,
    clip_range=0.2,
)


# model = PPO(
#     "MultiInputPolicy", env,
#     policy_kwargs=dict(
#         features_extractor_class=SpatialTransformerModel.SpatialTransformerExtractor,
#         features_extractor_kwargs=dict(
#             features_dim=256,
#             n_heads=4,
#             n_layers=2,
#         ),
#         net_arch=dict(pi=[128, 64], vf=[128, 64]),
#     ),
#     verbose=1,
#     learning_rate=3e-4,
#     batch_size=256,
#     n_steps=2048,
#     n_epochs=10,
#     gamma=0.99,
#     gae_lambda=0.95,
#     ent_coef=0.05,
#     vf_coef=0.5,
#     max_grad_norm=0.5,
#     clip_range=0.2,
# )







# model = PPO(
#     "MultiInputPolicy", env,
#     verbose=1,
#     learning_rate=1e-4,          # lower — reward scale is high
#     batch_size=256,
#     n_steps=2048,                # ~20 full episodes per rollout
#     n_epochs=10,
#     gamma=0.99,
#     gae_lambda=0.95,
#     ent_coef=0.05,               # higher entropy to fight premature convergence
#     vf_coef=0.5,
#     max_grad_norm=0.5,
#     clip_range=0.2
# )


memory_reset_cb = MemoryResetCallback()
memory_reset_cb = MemoryResetCallback()
monitor_cb      = TrainingMonitorCallback()
eval_cb = EvalCallback(
    eval_env,
    best_model_save_path="./best_model/",
    log_path="./logs/",
    eval_freq=50_000,            # evaluate every 50k timesteps
    n_eval_episodes=10,          # average over 10 episodes
    deterministic=True,          # no exploration during eval
)
checkpoint_cb = CheckpointCallback(
    save_freq=50_000,            # save every 50k timesteps
    save_path="./checkpoints/",
    name_prefix="drone_ppo",
    save_vecnormalize=True,      # saves the VecNormalize stats too
)




model.learn(
    total_timesteps=5_000_000,
    callback=[memory_reset_cb, monitor_cb, checkpoint_cb, eval_cb],
    reset_num_timesteps=False,
)

model.save("./single_agent_xformer.zip")








        
        




        
