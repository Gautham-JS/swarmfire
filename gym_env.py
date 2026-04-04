
from envs.MultiAgentEnv import MultiAgentEnv


from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

env = DummyVecEnv(
    [lambda: MultiAgentEnv(
            n_agents=1,
            world_size=(512, 512),
            start_positions=[(256, 256), (256, 128), (128, 256), (256, 256)],  # fixed grid
            render_mode="human",
            sample_interval=25,
            save_interval=500,
            seed=13,
            is_vid_out=True,
            vid_id="no_swarming_global_reward",
            vid_base_path="/home/gjs/software/thesis/swarmfire/vids/"
        )
    ]
)
env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)

model = PPO(
    "MultiInputPolicy", env,
    verbose=1,
    learning_rate=1e-4,          # lower — reward scale is high
    batch_size=256,
    n_steps=2048,                # ~20 full episodes per rollout
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.05,               # higher entropy to fight premature convergence
    vf_coef=0.5,
    max_grad_norm=0.5,
    clip_range=0.2,
)



model.learn(total_timesteps=500_000)
model.save("./single_agent.zip")

# for episode in range(10):
#     obs, _ = env.reset()
#     episode_reward = 0.0
#     is_exit = False
#     terminated = False

#     while not terminated or not is_exit:  # PettingZoo: loop until no agents remain
#         actions = env.action_space.sample()
#         obs, rewards, terminated, truncated, infos = env.step(actions)
#         episode_reward += rewards
#         frame = env.render()
    
#     if is_exit:
#         break

#     print(f"Episode {episode} | rewards: {episode_reward}")






        
        




        
