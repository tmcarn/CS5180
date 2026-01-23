import numpy as np
from matplotlib import pyplot as plt

"""
Here is the implementation of the 10-armed Bandit problem/testbed.
Must call the reset function whenever you want to generate a new 10-armed Bandit problem

"""
class Bandit(object):
    def __init__(self, k=10):
        # Number of the actions
        self.k = k

        # Numpy array to store the true action value the k arms/actions
        self.q_star = np.empty(self.k)

    def reset(self):
        # Reset the true action values to generate a new k-armed bandit problem
        # Value for each arm is randomly sampled from a normal distribution 
        # with mean = 0, variance = 1.0. 
        self.q_star = np.random.normal(loc=0, scale=1, size=self.k)
        
    def best_action(self):
        """Return the indices of all best actions/arms in a list variable
        """
        return np.where(self.q_star == self.q_star.max())[0].tolist()  

    def step(self, act):
        """
        Args:
            act (int): index of the action
        """
        # Compute the reward for each action
        # The reward for each action at time step t is sampled from a Gaussian distribution
        # For the k-th arm, the mean = q_star[k] (true value) and variance = 1
        rewards = np.random.normal(loc=self.q_star, scale=np.ones(self.k), size=self.k)
        return rewards[act]
    

def MultiArmBandit(k: int, num_samples: int):
    """
    Structure:
        1. Create multi-armed bandit env
        2. Pull each arm `num_samples` times and record the rewards
        3. Plot the rewards (e.g. violinplot, stripplot)

    Args:
        k (int): Number of arms in bandit environment
        num_samples (int): number of samples to take for each arm
    """

    env = Bandit(k=k)
    env.reset()

    reward_hist = np.zeros((num_samples, k))
    for sample in range(num_samples):
        for i in range(k):
            reward = env.step(i)
            reward_hist[sample, i] = reward

    return reward_hist

def plot_reward_dist(reward_data):
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.violinplot(reward_data)
    plt.axhline(0, linestyle="--")
    ax.set_xticks(np.arange(0, 11, 1))
    ax.set_title("Reward Distribution for Each Arm of Multi-Arm Bandit")
    ax.set_xlabel("Action")
    ax.set_ylabel("Reward Distribution")
    plt.savefig("plots/RewardDistribution.png")

plot_reward_dist(MultiArmBandit(10, 10_000))