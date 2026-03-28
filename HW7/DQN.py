import os

import numpy as np
import torch
from torch import nn
import tqdm
from qnets import MLPQNet


class DQNAgent(object):
    def __init__(self, env, params):
        
        self.env = env
        self.params = params
        self.step_num = 0

        self.epsilon_scheduler = params["epsilon_scheduler"]

        self.action_space_list = params["action_space"]
        self.action_space = params["action_dim"]

        self.observation_space = params["observation_dim"]

        self.gamma = params["gamma"]

        self.hidden_layer_dim = params["hidden_layer_dim"]
        self.hidden_layer_num = params["hidden_layer_num"]

        self.batch_size = params["batch_size"]

        self.target_update_freq = params["target_update_freq"]
        self.behavior_update_freq = params["behavior_update_freq"]

        self.save_freq = params["save_freq"]

        self.replay_buffer = ReplayBuffer(capacity=params["replay_buffer_size"])

        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

        self.behavior_model = self.build_model()
        self.target_model = self.build_model()

        self.target_model.load_state_dict(self.behavior_model.state_dict()) # initialize the target model with the same weights as the behavior model

        self.target_model.to(self.device)
        self.behavior_model.to(self.device) 
        
        # Save Initial Model with no training
        self.save_root = f"models/{self.params['model_name']}"
        name = f"{self.params['model_name']}_step0.pt"
        self.save(os.path.join(self.save_root, name))

        self.loss_fn = nn.MSELoss()
        self.behavior_optimizer = torch.optim.Adam(self.behavior_model.parameters(), lr=params["learning_rate"])

    def behavior_policy(self, state, mode="train"):
        epsilon = self.epsilon_scheduler.get_value(self.step_num)
        sample = np.random.rand()

        if sample < epsilon and mode == "train":
            # explore: choose a random action
            action_idx = np.random.choice(np.arange(self.action_space))
       
        else:            
            # exploit: choose the action with the highest Q value
            self.behavior_model.eval()
            with torch.no_grad():
                state_tensor = torch.FloatTensor(np.array([state])).to(self.device)  # (1, state_dim)
                q_values = self.behavior_model(state_tensor)
                action_idx = torch.argmax(q_values).item()

        return action_idx
    
    def build_model(self):

        model = MLPQNet(self.observation_space, self.hidden_layer_dim, self.action_space, num_hidden=self.hidden_layer_num)
        return model.apply(MLPQNet.customized_weights_init)
    
    def update_model(self):
        self.behavior_model.train()

        states, actions, rewards, next_states, terminateds = self.replay_buffer.sample(self.batch_size)

        # convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)                # (64, state_dim)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(self.device) # (64, 1)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)             # (64,)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)    # (64, state_dim)
        terminateds = torch.FloatTensor(np.array(terminateds)).to(self.device)     # (64,)      

        # compute current Q values
        current_q_values = self.behavior_model(states).gather(dim=1, index=actions).squeeze(1) # (64,)

        # compute target Q values
        with torch.no_grad():
            next_q_values, idx = self.target_model(next_states).max(dim=1) # (64,)
            target_q_values = rewards + (self.gamma * next_q_values * (1 - terminateds)) # (64,)

        # compute loss
        loss = self.loss_fn(current_q_values, target_q_values)

        # Backpropagation and optimization
        self.behavior_optimizer.zero_grad()
        loss.backward()
        self.behavior_optimizer.step()

        return loss

    def train(self):
        train_returns = []
        train_loss = []
        last_best_return = -np.inf

        episode_steps = 0       
        rewards = []
        state, _ = self.env.reset()

        pbar = tqdm.trange(self.params['total_training_time_step'])

        for step in pbar:
            self.step_num = step
            
            action_idx = self.behavior_policy(state, mode="train")
            action = self.action_space_list[action_idx]
            next_state, reward, terminated, _, _ = self.env.step(action)
            self.replay_buffer.push(state, action_idx, reward, next_state, terminated)
            rewards.append(reward)

            if terminated:
                # compute the return
                G = 0
                for r in reversed(rewards):
                    G = r + self.params['gamma'] * G

                if G > last_best_return:
                    last_best_return = G
                    name = f"{self.params['model_name']}_best.pt"
                    self.save(os.path.join(self.save_root, name))

                train_returns.append(G)
                total_episodes = len(train_returns)

                # print the information
                pbar.set_description(
                    f"Ep={total_episodes} | "
                    f"G={np.mean(train_returns[-10:]) if train_returns else 0:.2f} | "
                    f"Steps={episode_steps}")
                
                rewards = []
                episode_steps = 0

                state, _ = self.env.reset()

            else:
                state = next_state
                episode_steps += 1

            if self.step_num >= self.params["start_training_step"] and len(self.replay_buffer.buffer) >= self.batch_size:
                if step % self.params["behavior_update_freq"] == 0:
                    loss = self.update_model().item()
                    train_loss.append(loss)

                if step % self.params["target_update_freq"] == 0:
                    self.target_model.load_state_dict(self.behavior_model.state_dict())

                if step % self.save_freq == self.save_freq - 1:
                    name = f"{self.params['model_name']}_step{step}.pt"
                    self.save(os.path.join(self.save_root, name))
            
        return train_returns, train_loss
    
    # ---- save / load / play ----

    def save(self, path):
        torch.save({
            "model_state_dict": self.behavior_model.state_dict(),
            "params": self.params,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.behavior_model.load_state_dict(checkpoint["model_state_dict"])
        self.behavior_model.to(self.device)
        self.behavior_model.eval()

    @classmethod
    def from_checkpoint(cls, path, env, render=True):
        """Load a trained agent from a checkpoint file.
        
        If no env is provided, creates a new one from params['env_name'].
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        params = checkpoint["params"]

        agent = cls(env, params)
        agent.behavior_model.load_state_dict(checkpoint["model_state_dict"])
        agent.behavior_model.eval()
        return agent

    def play(self, num_episodes=10):
        self.behavior_model.eval()
        episode_returns = []

        for ep in range(num_episodes):
            state, _ = self.env.reset()
            terminated, truncated = False, False
            total_reward = 0

            while not (terminated or truncated):
                action_idx = self.behavior_policy(state, mode="eval")
                action = self.action_space_list[action_idx]
                state, reward, terminated, truncated, _ = self.env.step(action)
                total_reward += reward

            episode_returns.append(total_reward)
            print(f"Episode {ep + 1}: reward = {total_reward:.1f}")

        print(f"\nMean reward over {num_episodes} episodes: {np.mean(episode_returns):.1f}")
        return episode_returns
    
            
class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = np.empty(capacity, dtype=object)
        self.position = 0
        self.size = 0

    def push(self, state, action, reward, next_state, terminated):
        self.buffer[self.position] = (state, action, reward, next_state, terminated)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.choice(self.size, batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        return zip(*batch)

