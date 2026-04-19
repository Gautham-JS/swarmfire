from envs.MultiAgentEnv import MultiAgentEnv

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO

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
            vid_base_path="/home/gjs/software/thesis/swarmfire/vids/",
            phase_weights={"exploration": 2.0, "fire_discovery": 3.8,
                           "fire_tracking": 2.0, "risk": 2.0}
        )
    ]
)
env = VecNormalize.load("checkpoints/firescout_vecnormalize_400000_steps.pkl", env)
model = RecurrentPPO.load("checkpoints/firescout_400000_steps.zip", env=env)



o = env.reset()
print(o)


for _ in range(1000):
    action, _ = model.predict(o, deterministic=True)
    o, reward, done, info = env.step(action)
    env.render()
    if done.any():
        o, info = env.reset()

env.close()
