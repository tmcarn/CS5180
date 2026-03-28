from DQN import DQNAgent
from schedulers import ExponentialSchedule
from four_rooms_env import FourRooms
from qnets import MLPQNet

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from utils import run_training
import argparse

scheduler = ExponentialSchedule(start_value=1.0, end_value=0.01, num_steps=250_000)
env = gym.make("CartPole-v1")
total_training_time_step = 1_500_000

# create training parameters
train_parameters = {
    'observation_dim': env.observation_space.shape[0],
    'action_dim': env.action_space.n,
    'action_space': list(range(env.action_space.n)),
    'hidden_layer_num': 1,
    'hidden_layer_dim': 128,
    'gamma': 0.99,

    'total_training_time_step': total_training_time_step,

    'epsilon_scheduler': scheduler,
    'qnet_class': MLPQNet,

    'replay_buffer_size': 200_000,
    'start_training_step': 2_000,
    'behavior_update_freq': 4,
    'target_update_freq': 2_000,

    'save_freq': total_training_time_step / 4,

    'batch_size': 32,
    'learning_rate': 1e-3,

    'model_name': "cart_pole"
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "play"], default="train")
    parser.add_argument("--checkpoint", type=str, default="models/cart_pole.pt")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=10)

    args = parser.parse_args()

    if args.mode == "train":
        env = gym.make("CartPole-v1", render_mode=None)
        run_training(env, train_parameters, num_trials=args.trials, name="CartPole")

    elif args.mode == "play":
        env = gym.make("CartPole-v1", render_mode="human")
        agent = DQNAgent.from_checkpoint(args.checkpoint, env=env)
        agent.play(num_episodes=args.episodes)

