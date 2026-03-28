import gymnasium as gym
import ale_py

gym.register_envs(ale_py)
env = gym.make("ALE/Pong-v5", render_mode="rgb_array", obs_type="grayscale", frameskip=4)

input_shape = env.observation_space.shape
action_space = env.action_space.n

print(f"Observation space shape: {input_shape}")

