import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

class FourRooms(object):
    def __init__(self):
        # define the four room as a 2-D array for easy state space reference and visualization
        # 0 represents an empty cell; 1 represents a wall cell
        self.four_room_space = np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                         [1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1],
                                         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]])

        # find the positions for all empty cells
        # note that: the origin for a 2-D numpy array is located at top-left while the origin for the FourRooms is at
        # the bottom-left. The following codes performs the re-projection.
        empty_cells = np.where(self.four_room_space == 0.0)
        self.state_space = [[col, 10 - row] for row, col in zip(empty_cells[0], empty_cells[1])]

        # define the action space
        self.action_space = {'LEFT': np.array([-1, 0]),
                             'RIGHT': np.array([1, 0]),
                             'DOWN': np.array([0, -1]),
                             'UP': np.array([0, 1])}
        
        self.action_space_list = list(self.action_space.keys())

        # define the start state
        self.start_state = [0, 0]

        # define the goal state
        self.goal_state = [10, 10]

        # maximal time steps
        self.max_time_steps = 459

        # track the time step
        self.t = 0

    def reset(self):
        """
        Reset the agent's state to the start state [0, 0]
        Return both the start state and reward
        """
        # reset the agent state to be [0, 0]
        state = self.start_state
        # reset the reward to be 0
        reward = 0
        # reset the termination flag
        done = False
        # reset the time step tracker
        self.t = 0
        return state, reward, done

    def step(self, state, act):
        """
        Args:
            state: a list variable containing x, y integer coordinates. (i.e., [1, 1]).
            act: a string variable (i.e., "UP"). All feasible values are ["UP", "DOWN", "LEFT", "RIGHT"].
        Output args:
            next_state: a list variable containing x, y integer coordinates (i.e., [1, 1])
            reward: an integer. it can be either 0 or 1.
        """
        # Increase the time step
        self.t += 1

        # With probability 0.8, the agent takes the correct direction.
        # With probability 0.2, the agent takes one of the two perpendicular actions.
        # For example, if the correct action is "LEFT", then
        #     - With probability 0.8, the agent takes action "LEFT";
        #     - With probability 0.1, the agent takes action "UP";
        #     - With probability 0.1, the agent takes action "DOWN".
        if np.random.uniform() < 0.2:
            if act == "LEFT" or act == "RIGHT":
                act = np.random.choice(["UP", "DOWN"], 1)[0]
            else:
                act = np.random.choice(["RIGHT", "LEFT"], 1)[0]

        # Compute the next state
        next_state = self.take_action(state, act)

        # Compute the reward
        reward = 1.0 if next_state == [10, 10] else 0.0

        # Check the termination
        # If the agent reaches the goal, reward = 1, done = True
        # If the time steps reaches the maximal number, reward = 0, done = True.
        if next_state == [10, 10] or self.t == self.max_time_steps:
            done = True
        else:
            done = False

        return next_state, reward, done

    def take_action(self, state, act):
        """
        Input args:
            state (list): a list variable containing x, y integer coordinates. (i.e., [1, 1]).
            act (string): a string variable (i.e., "UP"). All feasible values are ["UP", "DOWN", "LEFT", "RIGHT"].
        Output args:
            next_state (list): a list variable containing x, y integer coordinates (i.e., [1, 1])
        """
        state = np.array(state)
        next_state = state + self.action_space[act]
        return next_state.tolist() if next_state.tolist() in self.state_space else state.tolist()
    
    def e_soft_policy(self, state, qvals, epsilon):
        num_actions = len(self.action_space)

        dist = np.ones(num_actions) * epsilon / num_actions
        greedy_action = self.argmax(qvals[state[0], state[1]])
        dist[greedy_action] += 1 - epsilon

        action_idx = np.random.choice(num_actions, p=dist)

        return action_idx
        
        
    def rollout(self, qvals, epsilon):
        """
        Input args:
            policy: a numpy array of shape (state_space_size, ) where each element is an integer in [0, 3] representing the action index.
            max_episode_length: an integer representing the maximal length of an episode.
        Output args:
            episode: a list of tuples. Each tuple is of the form (state, action, reward).
        """
        # reset the environment to get the start state
        state, _, _ = self.reset()
        episode = []
        first_visit = {}

        for step in range(self.max_time_steps):
            action_idx = self.e_soft_policy(state, qvals, epsilon) # get the action index from the policy
            action = self.action_space_list[action_idx] # convert the action index to action name
            next_state, reward, done = self.step(state, action) # take the action and observe the next state and reward

            if (state[0], state[1]) not in first_visit:
                first_visit[(state[0], state[1])] = step
            
            episode.append((state, action_idx, reward)) # store the transition in the episode
            state = next_state # update the current state

            if done:
                break   
        return episode, first_visit
    
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
    
def plot_rewards(final_returns, epsilon):
    # plot mean reward
    mean_reward = np.mean(final_returns, axis=0)
    plt.plot(np.arange(final_returns.shape[1]), mean_reward, linestyle="-", linewidth=2, label=f"$\epsilon$ = {epsilon}")


env = FourRooms()

# Initialize random policy for all states
state_space_size = len(env.state_space)

epsilon = [0.1, 0.01, 0]
gamma = 0.99
num_trials = 10
num_episodes = int(1e4)

final_returns = np.zeros((num_trials, num_episodes))

# Initialize Figure
plt.figure(figsize=(8, 6))

for epsilon in [0.1, 0.01, 0]:
    for trial in range(num_trials):
        q_values = np.zeros((11, 11, 4)) # Initialize Q-values for all state-action pairs to zero
        occurences = np.zeros((11, 11, 4))

        for ep_num in tqdm(range(num_episodes)): # Run for a certain number of episodes
            episode, first_visits = env.rollout(q_values, epsilon)
            total_discounted_reward = 0

            for i in range(1, len(episode)+1): 
                state, action_idx, reward = episode[-i] # Loop through backwards
                total_discounted_reward = reward + gamma * total_discounted_reward

                if first_visits[(state[0], state[1])] == len(episode) - i: # Check if this is the first visit to the state
                    occurences[state[0], state[1], action_idx] += 1
                    q_values[state[0], state[1], action_idx] += (total_discounted_reward - q_values[state[0], state[1], action_idx]) / occurences[state[0], state[1], action_idx]

            
            final_returns[trial, ep_num] = total_discounted_reward

    plot_rewards(final_returns, epsilon)

# Plot Theoretically Optimal Return
optimal_return = 1.0 * (gamma ** 20)
plt.axhline(y=optimal_return, color='r', linestyle='--', label="Optimal Return")

# plot the figure
plt.ylabel("Average reward")
plt.xlabel("Episode")
plt.legend()
plt.title("First-Visit Monte Carlo Control with $\epsilon$-Soft Policy")
plt.savefig("plots/four_rooms_mc_control.png")
plt.show()





