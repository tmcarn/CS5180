import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random
import gymnasium as gym


def fixed_policy_rollout(state, policy, env):
    # given a state, a policy and an environment, this function simulates an episode and returns the total reward
    terminal = False
    episode = []
    
    while not terminal:
        action = policy(state)
        next_s, reward, terminal, _, _ = env.step(action)
        episode.append((state, action, next_s, reward))
        state = next_s
    
    return episode

def plot_value_function(value_function, policy_name):
    # plot the value function as a heatmap
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), subplot_kw={'projection': '3d'})

    # Create mesh grid
    player_sums = np.arange(12, 22)
    dealer_cards = np.arange(1, 11)
    X, Y = np.meshgrid(dealer_cards, player_sums)

    for i in range(2):
        ax = axes[i]
        
        data = value_function[12:22, 1:11, i] # Crop to Non-zero State Space
        
        # Plot surface
        surf = ax.plot_surface(X, Y, data, cmap='RdYlGn', alpha=0.9)
        
        ax.set_xlabel('Dealer Showing')
        ax.set_ylabel('Player Sum')
        ax.set_zlabel('Value')
        ax.set_title(f'Usable Ace: {bool(i)}')
        fig.colorbar(surf, ax=ax, shrink=0.5)

    plt.tight_layout()
    plt.suptitle(f'Value Function for Blackjack ({policy_name})', fontsize=16)
    plt.savefig(f'plots/blackjack_value_function_{policy_name.lower().replace(" ", "_")}.png')
    plt.show()


'''
First-Visit Monte Carlo Policy Evaluation for Blackjack (Fixed Policy)
'''

def simple_policy(state):
    # a simple policy that sticks if the player score is >= 20 and hits otherwise
    if state[0] >= 20:
        return 0 # stick
    return 1 # hit

env = gym.make("Blackjack-v1")
state_space = env.observation_space
num_episodes = 500_000
gamma = 1.0

value_function = np.zeros((state_space[0].n, state_space[1].n, state_space[2].n))
num_occurrences = np.zeros((state_space[0].n, state_space[1].n, state_space[2].n))

for episode_num in range(num_episodes):
    init_state = env.reset()[0] # get a randomized initial state
    episode = fixed_policy_rollout(init_state, simple_policy, env)
    total_reward = 0

    for i in range(len(episode)-1, -1, -1): # iterate backwards through the episode
        state, action, next_s, reward = episode[i]
        total_reward = (gamma * total_reward) + reward
        num_occurrences[state] += 1 # increment the count of occurrences for the state
        value_function[state] += (total_reward - value_function[state]) / num_occurrences[state] # update the value function for the state

plot_value_function(value_function, "Fixed Policy") # plot the value function derived from the fixed policy


'''
First-Visit Monte Carlo Exploring Starts for Blackjack (Discover Optimal Policy)
'''

def plot_policy(policy):
    # plot the policy as a heatmap
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    # Create mesh grid
    player_sums = np.arange(12, 22)
    dealer_cards = np.arange(1, 11)
    X, Y = np.meshgrid(dealer_cards, player_sums)

    for i in range(2):
        ax = axes[i]
        
        data = policy[12:22, 1:11, i] # Crop to Non-zero State Space

        hit_color = "#247044"  # Green
        stick_color = "#f6a205"    # Red
        cmap = ListedColormap([stick_color, hit_color])
        
        # Plot heatmap
        heatmap = ax.imshow(data, cmap=cmap, origin='lower') 
        
        ax.set_xlabel('Dealer Showing')
        ax.set_ylabel('Player Sum')
        ax.set_title(f'Usable Ace: {bool(i)}')

        # Tickesr for better readability
        ax.set_xticks(range(10))
        ax.set_xticklabels(range(1, 11))
        ax.set_yticks(range(10))
        ax.set_yticklabels(range(12, 22))
        
        # Gridlines for better visibility
        ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=2)

    plt.tight_layout()
    plt.suptitle('Optimal Policy for Blackjack (Green=Hit, Orange=Stick)', fontsize=16)
    plt.savefig('plots/blackjack_optimal_policy.png')
    plt.show()

def dynamic_policy_rollout(state, init_action, policy, env):
    # given a state, a policy and an environment, this function simulates an episode and returns the total reward
    terminal = False
    episode = []

    # Take the initial action for exploring starts
    if init_action is not None:
        next_s, reward, terminal, _, _ = env.step(init_action)
        episode.append((state, init_action, next_s, reward))
        state = next_s

    while not terminal:
        action = policy[state]
        next_s, reward, terminal, _, _ = env.step(action)
        episode.append((state, action, next_s, reward))
        state = next_s
    
    return episode

env = gym.make("Blackjack-v1")
state_space = env.observation_space
action_space = env.action_space
num_episodes = 500_000
gamma = 1.0

policy = np.zeros((state_space[0].n, state_space[1].n, state_space[2].n), dtype=int) # initial policy is to stick for every state
action_value = np.zeros((state_space[0].n, state_space[1].n, state_space[2].n, action_space.n)) # action-value function initialized to zero
num_occurrences = np.zeros((state_space[0].n, state_space[1].n, state_space[2].n, action_space.n)) # count of occurrences for each state-action pair

for episode_num in range(num_episodes):
    init_state = env.reset()[0] # get a randomized initial state
    init_action = np.random.randint(0, action_space.n) # select a random action for the initial state (exploring starts)
    episode = dynamic_policy_rollout(init_state, init_action, policy, env)
    total_reward = 0

    for i in range(len(episode)-1, -1, -1): # iterate backwards through the episode
        state, action, next_s, reward = episode[i]
        total_reward = (gamma * total_reward) + reward
        state_action_pair = state + (int(action),)
        num_occurrences[state_action_pair] += 1 # increment the count of occurrences for the state
        action_value[state_action_pair] += (total_reward - action_value[state_action_pair]) / num_occurrences[state_action_pair] # update the action-value function for the state-action pair
        optimal_action = np.argmax(action_value[state]) # get the optimal action for the state based on the action-value function
        policy[state] = optimal_action # update the policy for the state to be the optimal action

plot_value_function(action_value.max(axis=3), "Optimal Policy") # plot the value function derived from the optimal policy
plot_policy(policy) # plot the optimal policy

