import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from bandit import Bandit

class Agent():
    def __init__(self, k: int, init: int, alpha: float=None) -> None:
        """Default agent

        Args:
            k (int): number of arms
            init (init): initial value of Q-values
        """
        # Number of the arms. For example, k = 10 for 10-armed Bandit problem
        self.k = k

        # Initial Q value
        self.init = init

        # Q-values for each arm
        self.Q = None
        # Number of times each arm was pulled
        self.N = None
        # Current total number of steps
        self.t = None

        self.alpha = alpha

    def reset(self) -> None:
        """Initialize or reset Q-values and counts

        This method should be called after __init__() at least once
        """
        self.Q = self.init * np.ones(self.k, dtype=np.float32)
        self.N = np.zeros(self.k, dtype=int)
        self.t = 0

    def update(self, action: int, reward: float) -> None:
        """Update Q-values and N after observing reward.

        Args:
            action (int): index of pulled arm
            reward (float): reward obtained for pulling arm
        """
        # increase the time step
        self.t += 1
        
        # update the self.N
        self.N[action] += 1
        
        # update self.Q with the incremental update
        if self.alpha:
            self.Q[action] += self.alpha * (reward - self.Q[action])

        else:
            self.Q[action] += (1.0 / self.N[action]) * (reward - self.Q[action])


    @staticmethod
    def argmax(arr) -> int:
        """Argmax that breaks ties randomly

        Takes in a list of values and returns the index of the item with the highest value, breaking ties randomly.

        Note: np.argmax returns the first index that matches the maximum, so we define this method to use in EpsilonGreedy and UCB agents.
        Args:
            arr: sequence of values
        """
        arr = np.asarray(arr)
        max_val = arr.max()
        max_indices = np.flatnonzero(arr == max_val)
        return np.random.choice(max_indices)

class EpsilonGreedyAgent(Agent):
    def __init__(self, k, init, epsilon: float) -> None:
        """Epsilon greedy bandit agent

        Args:
            k (int): number of arms
            init (init): initial value of Q-values
            epsilon (float): random action probability
        """
        super().__init__(k, init)

        # Epsilon value
        self.epsilon = epsilon


    def choose_action(self) -> int:
        """Choose which arm to pull

        With probability 1 - epsilon, choose the best action (break ties arbitrarily, use argmax() from above).
        
        With probability epsilon, choose a random action.
        """
        # E - Greedy
        sample = np.random.random()

        if sample < self.epsilon:
            # Choose Random Action
            action = np.random.choice(np.arange(self.k))
        
        else:
            action = self.argmax(self.Q)
        
        return action
    

class UCBAgent(Agent):
    def __init__(self, k, init, c, alpha=None):
        super().__init__(k, init, alpha)
        self.c = c
    
    def choose_action(self) -> int:
        t = np.sum(self.N) # Total steps taken 
        action_update = self.Q + (self.c * np.sqrt(np.log(t+1) / (self.N+1)))
        action = self.argmax(action_update)
        return action
        


""" Here is the function to run the epsilon greedy agent. Please complete the missing part under "CODE HERE"
"""
# run epsilon greedy 
def run_epsilon_greedy_agent(run_num: int, time_step: int, epsilon: float=0.0, init: float=0.0):
    """
    Args:
        run_num (int): number of runs
        time_step (int): number of time steps per run
        epsilon (float): epsilon for the agent
        init (float): initial value for the Q. (i.e., Q1)
    """
    # create the 10-armed Bandit problem
    k = 10
    env = Bandit(k)
    env.reset()

    # create the agent with proper initial value and epsilon
    agent = EpsilonGreedyAgent(k=k, init=init, epsilon=epsilon)
    agent.reset()

    # create a numpy array to store rewards with shape (run_num, time_step)
    # For example, results_rewards[r, t] stores the reward for step t in the r-th running trail
    results_rewards = np.empty((run_num, time_step))
    
    # create a numpy array to store optimal action proportion with shape (run_num, time_step)
    # For example, results_action[r, t] stores 1 if the selected action at step t in the r-th runing trail is optimal
    # and 0 otherwise.
    results_action = np.empty((run_num, time_step))
    
    # create a numpy array to save upper_bound (only for plotting rewards; it should be 1 for plotting action optimality proportion)
    # For example, upper_bound[r] stores the true action value for the r-th running trail.
    upper_bound = np.empty(run_num)
    

    # loop for trails starts
    for r in tqdm(range(run_num), desc="run number", position=0):
        
        # reset the environment to create a new 10-armed bandit problem.
        env.reset()

        # reset the agent
        agent.reset()
        
        # compute the upper bound for each running trial and update upper_bound[r]
        best_action = env.best_action()[0]
        upper_bound[r] = env.q_star[best_action] #* time_step # If you took best action at every timestep
        
        # loop for each trail a fixed number of steps
        for t in tqdm(range(time_step), desc="time step", position=1, leave=False):
            
            # get the best action to execute at step t 
            action = agent.choose_action()
            
            # interact with the environment to receive rewards
            reward = env.step(action)
                        
            # update the agent based on the observed reward
            agent.update(action, reward)
                     
            """DO NOT CHANGE BELOW"""
            # save the reward
            results_rewards[r, t] = reward
            # check and save whether the action is optimal
            if action in env.best_action():
                results_action[r, t] = 1
            else:
                results_action[r, t] = 0
            
    return results_rewards, results_action, upper_bound

# run epsilon greedy 
def run_ucb_agent(run_num: int, time_step: int, init: float=0.0, c: float= 1.0):
    """
    Args:
        run_num (int): number of runs
        time_step (int): number of time steps per run
        epsilon (float): epsilon for the agent
        init (float): initial value for the Q. (i.e., Q1)
    """
    # create the 10-armed Bandit problem
    k = 10
    env = Bandit(k)
    env.reset()

    # create the agent with proper initial value and epsilon
    agent = UCBAgent(k=k, init=init, c=c)
    agent.reset()

    # create a numpy array to store rewards with shape (run_num, time_step)
    # For example, results_rewards[r, t] stores the reward for step t in the r-th running trail
    results_rewards = np.empty((run_num, time_step))
    
    # create a numpy array to store optimal action proportion with shape (run_num, time_step)
    # For example, results_action[r, t] stores 1 if the selected action at step t in the r-th runing trail is optimal
    # and 0 otherwise.
    results_action = np.empty((run_num, time_step))
    
    # create a numpy array to save upper_bound (only for plotting rewards; it should be 1 for plotting action optimality proportion)
    # For example, upper_bound[r] stores the true action value for the r-th running trail.
    upper_bound = np.empty(run_num)
    

    # loop for trails starts
    for r in tqdm(range(run_num), desc="run number", position=0):
        
        # reset the environment to create a new 10-armed bandit problem.
        env.reset()

        # reset the agent
        agent.reset()
        
        # compute the upper bound for each running trial and update upper_bound[r]
        best_action = env.best_action()[0]
        upper_bound[r] = env.q_star[best_action] #* time_step # If you took best action at every timestep
        
        # loop for each trail a fixed number of steps
        for t in tqdm(range(time_step), desc="time step", position=1, leave=False):
            
            # get the best action to execute at step t 
            action = agent.choose_action()
            
            # interact with the environment to receive rewards
            reward = env.step(action)
                        
            # update the agent based on the observed reward
            agent.update(action, reward)
                     
            """DO NOT CHANGE BELOW"""
            # save the reward
            results_rewards[r, t] = reward
            # check and save whether the action is optimal
            if action in env.best_action():
                results_action[r, t] = 1
            else:
                results_action[r, t] = 0
            
    return results_rewards, results_action, upper_bound


def plot_curves(arr_list, legend_list, color_list, upper_bound, ylabel, title):
    """
    Args:
        arr_list (list): list of results arrays to plot
        legend_list (list): list of legends corresponding to each result array
        color_list (list): list of color corresponding to each result array
        upper_bound (numpy array): array contains the best possible rewards for 2000 runs. the shape should be (2000,)
        ylabel (string): label of the Y axis
        
        Note that, make sure the elements in the arr_list, legend_list and color_list are associated with each other correctly. 
        Do not forget to change the ylabel for different plots.
        
        To plot the upper bound for % Optimal action figure, set upper_bound = np.ones(num_step), where num_step is the number of steps.
    """
    # set the figure type
    plt.clf()
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # PLEASE NOTE: Change the labels for different plots
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Steps")
    ax.set_ylim(-0.1, upper_bound.mean() + 0.1)
    
    # ploth results
    h_list = []
    for arr, legend, color in zip(arr_list, legend_list, color_list):
        # compute the standard error
        arr_err = arr.std(axis=0) / np.sqrt(arr.shape[0])
        # plot the mean
        h, = ax.plot(range(arr.shape[1]), arr.mean(axis=0), color=color, label=legend)
        # plot the confidence band
        arr_err = 1.96 * arr_err
        ax.fill_between(range(arr.shape[1]), arr.mean(axis=0) - arr_err, arr.mean(axis=0) + arr_err, alpha=0.3, color=color)
        # save the plot handle
        h_list.append(h) 
    
    # plot the upper bound
    h = plt.axhline(y=upper_bound.mean(), color='k', linestyle='--', label="upper bound")
    h_list.append(h)
    
    # plot legends
    ax.legend(handles=h_list)
    ax.set_title(title)
    plt.savefig(f"plots/{title}.png")



'''
Running E-greedy
'''

# always set the random seed for results reproduction
np.random.seed(1234)
    
# set the running parameters (Use 2000 runs and 1000 steps for final report)
run_num = 2000
time_step = 1000

reward_list = []
action_list = []

epsilons = [0.0, 0.01, 0.1]

for epsilon in epsilons:
    result_rewards, result_action, upper_bound = run_epsilon_greedy_agent(run_num, time_step, init=0.0, epsilon=epsilon)
    reward_list.append(result_rewards)
    action_list.append(result_action)

plot_curves(reward_list,
        ["ε = 0","ε = 0.01", "ε = 0.10"],
        ["tab:blue", "tab:orange", "tab:green"],
        upper_bound, # should be 100%
        "Reward",
        "Bandit Rewards with ε-Greedy")

plot_curves(action_list,
        ["ε = 0","ε = 0.01", "ε = 0.10"],
        ["tab:blue", "tab:orange", "tab:green"],
        np.full((run_num, time_step), 1), # should be 100%
        "% Optimal Action",
        "Bandit Optimal Action with ε-Greedy")

'''
Running 5 Experiments
'''
# always set the random seed for results reproduction
np.random.seed(1234)
    
# set the running parameters (Use 2000 runs and 1000 steps for final report)
run_num = 2000
time_step = 1000

reward_list = []
action_list = []

inits = [0, 5, 0, 5]
epsilons = [0, 0, 0.1, 0.1]

# Run E-Greedy
for i in range(4):
    result_rewards, result_action, upper_bound = run_epsilon_greedy_agent(run_num, time_step, init=inits[i], epsilon=epsilons[i])
    reward_list.append(result_rewards)
    action_list.append(result_action)

# Run UCB
result_rewards, result_action, upper_bound = run_ucb_agent(run_num, time_step, c=2)
reward_list.append(result_rewards)
action_list.append(result_action)

labels = []
for i in range(4):
    labels.append(f"ε = {epsilons[i]}, $q_0$ = {inits[i]}")
labels.append("UCB, c=2")

plot_curves(reward_list,
        labels,
        ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"],
        upper_bound, # should be 100%
        "Reward",
        "Bandit Rewards with ε-Greedy and UCB")

plot_curves(action_list,
        labels,
        ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"],
        np.full((run_num, time_step), 1), # should be 100%
        "% Optimal Action",
        "Bandit Optimal Action with ε-Greedy and UCB")


   