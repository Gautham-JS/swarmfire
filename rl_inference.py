from envs.MultiAgentEnv import MultiAgentEnv

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

env = DummyVecEnv(
    [
        lambda: MultiAgentEnv(
            n_agents=1,
            world_size=(512, 512),
            start_positions=[(256, 256), (256, 128), (128, 256), (256, 256)],  # fixed grid
            render_mode="human",
            sample_interval=1,
            save_interval=100,
            seed=27,
            is_vid_out=True,
            vid_id="no_swarming_global_reward",
            vid_base_path="/home/gjs/software/thesis/swarmfire/vids/"
        )
    ]
)
env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)
model = PPO.load("./single_agent.zip", env=env)



o = env.reset()
print(o)


for _ in range(1000):
    action, _ = model.predict(o, deterministic=True)
    o, reward, done, info = env.step(action)
    env.render()
    if done.any():
        o, info = env.reset()

env.close()
