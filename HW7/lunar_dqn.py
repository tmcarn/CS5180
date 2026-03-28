
from DQN import DQNAgent
from schedulers import ExponentialSchedule
from four_rooms_env import FourRooms
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from utils import run_training
import argparse

env = gym.make("LunarLander-v3", continuous=False, gravity=-9.81,
               enable_wind=False, wind_power=0, turbulence_power=0)

scheduler = ExponentialSchedule(start_value=1.0, end_value=0.01, num_steps=250_000)

total_training_time_step = 1_500_000

# create training parameters
train_parameters = {
    'observation_dim': env.observation_space.shape[0],
    'action_dim': env.action_space.n,
    'action_space': list(range(env.action_space.n)),
    'hidden_layer_num': 2,
    'hidden_layer_dim': 128,
    'gamma': 0.99,

    'total_training_time_step': total_training_time_step,

    'epsilon_scheduler': scheduler,

    'replay_buffer_size': 200_000,
    'start_training_step': 2_000,
    'behavior_update_freq': 4,
    'target_update_freq': 2_000,

    'save_freq': total_training_time_step / 4,

    'batch_size': 32,
    'learning_rate': 1e-3,

    'model_name': "lunar_lander"
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "play"], default="train")
    parser.add_argument("--checkpoint", type=str, default="models/lunar_lander.pt")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=10)

    args = parser.parse_args()

    if args.mode == "train":
        env = gym.make("LunarLander-v3", continuous=False, gravity=-9.81,
               enable_wind=False, wind_power=0, turbulence_power=0, render_mode=None)
        run_training(env, train_parameters, num_trials=args.trials, name="lunar_lander")

    elif args.mode == "play":
        env = gym.make("LunarLander-v3", continuous=False, gravity=-9.81,
               enable_wind=False, wind_power=0, turbulence_power=0, render_mode="human")
        agent = DQNAgent.from_checkpoint(args.checkpoint, env=env)
        agent.play(num_episodes=args.episodes)

