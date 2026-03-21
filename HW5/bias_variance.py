from windy_gridworld import WindyGridWorld, SARSA, ESoftAgent, plot_curves
import numpy as np
from matplotlib import pyplot as plt

class TD0:
    def __init__(self, env, policy, alpha, epsilon, gamma, timeout):
        self.env = env
        self.policy = policy # fixed plolicy used to evaluate (Q-Values from SARSA)
        
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.timeout = timeout

        # define the Value Function we will be estimating

    def e_soft_policy(self, state, policy):
        num_actions = len(self.env.action_space)
        dist = np.ones(num_actions) * self.epsilon / num_actions
        greedy_action = np.argmax(policy[tuple(state)])
        dist[greedy_action] += 1 - self.epsilon
        action_idx = np.random.choice(num_actions, p=dist)
        return action_idx
        

    def estimate(self, n, record_state=None, v = None):
        V = np.zeros(self.env.state_space.shape)
        targets = []
        for _ in range(n):
            state, terminated = self.env.reset()
            t = 0

            while not terminated and t < self.timeout:
                action_idx = self.e_soft_policy(state, self.policy) # select the action with the highest Q-value for the current state
                action = list(self.env.action_space)[action_idx]
                
                next_state, reward, terminated = self.env.step(action)

                if record_state is None:
                    # update the value function using TD(0) update rule
                    V[tuple(state)] += self.alpha * (reward + self.gamma * V[tuple(next_state)] - V[tuple(state)])

                elif tuple(state) == tuple(record_state):
                    # record from existing value function if record_state is specified
                    targets.append(reward + self.gamma * v[tuple(next_state)])
                
                state = next_state
                t += 1

        return V if record_state is None else targets
    

def plot_histograms(td_targets, mc_targets, number_samples):
    """
    Plot histograms of TD(0) and MC learning targets for each N.
    
    Args:
        td_targets: dict mapping N -> list of TD target values at state S
        mc_targets: dict mapping N -> list of MC target values at state S
        number_samples: list of N values, e.g. [1, 10, 50]
        true_value: optional true value of V(S) to plot as vertical line
    """
    fig, axes = plt.subplots(len(number_samples), 2, figsize=(10, 3 * len(number_samples)))

    for i, n in enumerate(number_samples):
        # TD(0) histogram
        axes[i, 0].hist(td_targets[i], bins=30, color="steelblue", edgecolor="black", alpha=0.7)
        axes[i, 0].set_title(f"TD(0) Targets (N={n})")
        axes[i, 0].set_xlabel("Target Value")
        axes[i, 0].set_ylabel("Frequency")

        # MC histogram
        axes[i, 1].hist(mc_targets[i], bins=30, color="coral", edgecolor="black", alpha=0.7)
        axes[i, 1].set_title(f"MC Targets (N={n})")
        axes[i, 1].set_xlabel("Target Value")
        axes[i, 1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()


"""
Initial Training to get base policy used for evaluation
"""
# Knight's move actions and no move actions enabled
enable_king_move_actions = False
enable_no_move_actions = False

run_num = 1
timeout = 8000

# create the environment
env = WindyGridWorld(enable_king_move=enable_king_move_actions, enable_no_move=enable_no_move_actions)

# parameters
epsilon = 0.1
alpha = 0.5
gamma = 1.0

# create the expected SARSA
for _ in range(run_num):
    # run for each trial
    controller_sarsa = SARSA(env, alpha, epsilon, gamma, timeout)
    episodes = controller_sarsa.run()

Q = controller_sarsa.Q

"""
Policy Evaluation
"""

number_samples = [1, 10, 50]

TD0_agent = TD0(env, Q, alpha, epsilon, gamma, timeout)
MC_agent = ESoftAgent(env, Q, epsilon, gamma, alpha, timeout)

start_state = env.start_state

td_start_targets = []
mc_start_targets = []

for n in number_samples:
    tdv = TD0_agent.estimate(n)
    mcv = MC_agent.estimate(n)

    td_start_targets.append(TD0_agent.estimate(100, record_state=start_state, v=tdv))
    mc_start_targets.append(MC_agent.estimate(100, record_state=start_state, v=mcv))

plot_histograms(td_start_targets, mc_start_targets, number_samples)