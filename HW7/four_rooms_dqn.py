import argparse

from DQN import DQNAgent
from four_rooms_env import FourRooms
import numpy as np
import matplotlib.pyplot as plt
from utils import run_training

env = FourRooms()

# create training parameters
train_parameters = {
    'observation_dim': 2,
    'action_dim': 4,
    'action_space': env.action_names,
    
    'hidden_layer_num': 2,
    'hidden_layer_dim': 128,
    'gamma': 1.0,

    'total_training_time_step': 500_000,

    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_duration': 400_000,

    'replay_buffer_size': 50000,
    'start_training_step': 2000,
    'behavior_update_freq': 4,
    'target_update_freq': 2000,

    'save_freq': 50000,

    'batch_size': 32,
    'learning_rate': 1e-3,

    'model_name': "four_room"
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "play"], default="train")
    parser.add_argument("--checkpoint", type=str, default="models/four_room.pt")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=10)

    args = parser.parse_args()

    env = FourRooms()

    if args.mode == "train":
        env = FourRooms()
        run_training(env, train_parameters, num_trials=args.trials, name="four_room")

    elif args.mode == "play":
        env = FourRooms(render_mode="human")
        agent = DQNAgent.from_checkpoint(args.checkpoint, env=env)
        agent.play(num_episodes=args.episodes)




