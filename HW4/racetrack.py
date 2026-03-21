import numpy as np
import matplotlib.pyplot as plt
import random

from tqdm import tqdm


# Define the Racetrack domain 1
racetrack_v1_arr = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1]])

racetrack_v2_arr = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1]])



class Racetrack(object):
    def __init__(self, version):
        # Load the pre-defined the domain having the following representation
        #   - 1: track cell
        #   - 0: empty cell
        #   - 2: empty cell on the start line
        #   - 3: empty cell on the finish line
        self.version = version

        if self.version == "v1":
            self.domain_arr = racetrack_v1_arr.copy()
        else:
            self.domain_arr = racetrack_v2_arr.copy()

        # domain size
        self.height, self.width = self.domain_arr.shape

        # State space consists of:
        # Agent location
        self.empty_cell_locs = self.render_cell_locations(val=0.0)
        self.track_cell_locs = self.render_cell_locations(val=1.0)
        self.start_cell_locs = self.render_cell_locations(val=2.0)
        self.finish_cell_locs = self.render_cell_locations(val=3.0)

        # Action space
        self.action_space = [[-1, -1], [-1, 0], [-1, 1],
                             [0, -1], [0, 0], [0, 1],
                             [1, -1], [1, 0], [1, 1]]

        # construct the state space
        self.state_space = []
        for loc in self.start_cell_locs + self.empty_cell_locs + self.finish_cell_locs:
            for i in range(5):
                for j in range(5):
                    self.state_space.append(loc + [i, j])

        # track the agent's location
        self.state = None
        self.action = None
        self.t = None

    def reset(self):
        # randomly select one cell from the start line
        start_loc = random.sample(self.start_cell_locs, 1)[0]
        # reset the velocity to be zero for both x and y directions
        start_vel = [0, 0]
        # the state is a combination of location and velocity
        # for example: [loc_x, loc_y, vel_x, vel_y]
        state = start_loc + start_vel
        # reward
        reward = None
        # done
        done = False
        # track agent's location
        self.state = tuple(state)
        self.t = 0
        return state, reward, done

    def step(self, state, action):
        """
        Args:
            state (list): a list variable consists of agent's location + agent's current velocity. e.g., [x, y, v_x, v_y]
            action (list): a list variable consists of agent's velocity increments. e.g., [increments_v_x, increments_v_y]
        """
        # reward is -1 for every time step until the agent passes the finish line
        reward = -1
        self.t += 1
        
        # with the probability = 0.1, set action = [0, 0]
        if np.random.random() < 0.1:
            action = [0, 0]

        # update the velocity components
        # note that, both velocity is discrete and constraint within [0, 4]
        next_vel_x = np.clip(state[2] + action[0], a_min=0, a_max=4)
        next_vel_y = np.clip(state[3] + action[1], a_min=0, a_max=4)
        next_state_vel = [next_vel_x, next_vel_y]

        # only the cells on the start line can have both 0 velocities
        if next_state_vel == [0, 0]:
            if state not in self.start_cell_locs:
                # non-zero for velocities
                if state[2] == 0 and state[3] != 0:
                    next_state_vel = [0, 1]
                if state[2] != 0 and state[3] == 0:
                    next_state_vel = [1, 0]
                if state[2] != 0 and state[3] != 0:
                    non_zero_idx = random.sample([0, 1], 1)[0]
                    next_state_vel[non_zero_idx] = 1

        # update the next state location based on the updated velocities
        next_state_loc = [np.clip(state[0] + next_state_vel[0], a_min=0, a_max=self.width-1),
                          np.clip(state[1] + next_state_vel[1], a_min=0, a_max=self.height-1)]

        # check whether the agent hits the track
        if next_state_loc in self.track_cell_locs:
            # move back to the start line
            next_state_loc = random.sample(self.start_cell_locs, 1)[0]
            # reduce velocity to be 0s
            next_state_vel = [0, 0]
            # episode continue
            done = False # 
            # next state
            next_state = next_state_loc + next_state_vel
            return next_state, reward, done

        # check whether the agent pass the finish line
        if next_state_loc in self.finish_cell_locs:
            next_state = next_state_loc + next_state_vel
            done = True
            return next_state, 0, done

        # otherwise combine the next state
        next_state = next_state_loc + next_state_vel
        # termination
        done = False

        # track the agent's state
        self.state = tuple(next_state)
        self.action = action
        return next_state, reward, done

    def render_cell_locations(self, val):
        row_loc_indices, col_loc_indices = np.where(self.domain_arr == val)
        cell_locations = [[c, (self.height-1) - r] for r, c in zip(row_loc_indices, col_loc_indices)]
        return cell_locations

    def render(self):
        plt.clf()
        plt.title(f"s = {self.state}, a = {self.action}")
        plot_arr = self.domain_arr.copy()
        plot_arr[(self.height - 1) - self.state[1], self.state[0]] = 4
        plt.imshow(plot_arr)
        plt.show(block=False)
        plt.pause(0.01)

class ESoftAgent(object):  
    def __init__(self, racetrack_env: Racetrack, epsilon=0.1):
        self.racetrack_env = racetrack_env
        self.action_space = self.racetrack_env.action_space
        self.epsilon = epsilon
        self.max_time_steps = 500

    def e_soft_policy(self, state, qvals):
        num_actions = len(self.action_space)
        dist = np.ones(num_actions) * self.epsilon / num_actions
        greedy_action = np.argmax(qvals[tuple(state)])
        dist[greedy_action] += 1 - self.epsilon
        action_idx = np.random.choice(num_actions, p=dist)

        return action_idx
    
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
        
        
    def rollout(self, qvals, use_epsilon=True):
        """
        Input args:
            policy: a numpy array of shape (state_space_size, ) where each element is an integer in [0, 3] representing the action index.
            max_episode_length: an integer representing the maximal length of an episode.
        Output args:
            episode: a list of tuples. Each tuple is of the form (state, action, reward).
        """
        # reset the environment to get the start state
        state, _, _ = self.racetrack_env.reset()
        episode = []
        first_visit = {}

        for step in range(self.max_time_steps):

            if use_epsilon:
                action_idx = self.e_soft_policy(state, qvals) # get the action index from the policy
            else:
                action_idx = np.argmax(qvals[tuple(state)]) # get the greedy action index from the Q-values

            action = self.action_space[action_idx] # convert the action index to action name
            next_state, reward, done = self.racetrack_env.step(state, action) # take the action and observe the next state and reward

            if tuple(state) not in first_visit:
                first_visit[tuple(state)] = step
            
            episode.append((state, action_idx, reward)) # store the transition in the episode
            state = next_state # update the current state

            if done:
                break   
        return episode, first_visit
    
def plot_rewards(final_returns):
    # plot mean reward
    mean_reward = np.mean(final_returns, axis=0)
    std_reward = np.std(final_returns, axis=0)
    plt.fill_between(np.arange(final_returns.shape[1]), mean_reward - std_reward, mean_reward + std_reward, alpha=0.2, label="+- 1 std")
    plt.plot(np.arange(final_returns.shape[1]), mean_reward, linestyle="-", linewidth=2)

def plot_trajectory(env, episodes, offpolicy=False):
    plt.figure(figsize=(8, 6))
    plt.title("Trajectory of the agent")
    plot_arr = env.domain_arr.copy()

    color_val = 4
    for episode in episodes:
        for state, _, _ in episode:
            plot_arr[(env.height - 1) - state[1], state[0]] = color_val
        color_val += 1

    plt.imshow(plot_arr)
    if offpolicy:
        plt.savefig(f"plots/racetrack_trajectory_offpolicy_track{env.version}.png")
    else:
        plt.savefig(f"plots/racetrack_trajectory_track{env.version}.png")
    plt.show()

    
"""E-Soft Monte Carlo Control Training For Racetrack #1"""
env = Racetrack(version="v1")
agent = ESoftAgent(racetrack_env=env)


# Initialize random policy for all states
state_space_shape = (env.width, env.height, 5, 5) # (loc_x, loc_y, vel_x, vel_y)
action_space_shape = (len(agent.action_space),) # 9 actions
state_action_space_shape = state_space_shape + action_space_shape # (loc_x, loc_y, vel_x, vel_y, d_vel_x, d_vel_y)

gamma = 1.0
epsilon_values = [0.1]
num_trials = 10
num_episodes = 2000

final_returns = np.zeros((num_trials, num_episodes))

# Initialize Figure
plt.figure(figsize=(8, 6))

for epsilon in epsilon_values:
    for trial in range(num_trials):
        q_values = np.zeros(state_action_space_shape) # Initialize Q-values for all state-action pairs to zero
        occurences = np.zeros(state_action_space_shape) # Initialize occurrence counts for all state-action pairs to zero
        
        for ep_num in tqdm(range(num_episodes)): # Run for a certain number of episodes
            episode, first_visits = agent.rollout(q_values)
            total_discounted_reward = 0

            for i in range(1, len(episode)+1): 
                state, action_idx, reward = episode[-i] # Loop through backwards
                total_discounted_reward = reward + gamma * total_discounted_reward

                if first_visits[tuple(state)] == len(episode) - i: # Check if this is the first visit to the state
                    occurences[tuple(state)+(action_idx,)] += 1
                    q_values[tuple(state)+(action_idx,)] += (total_discounted_reward - q_values[tuple(state)+(action_idx,)]) / occurences[tuple(state)+(action_idx,)]

            final_returns[trial, ep_num] = total_discounted_reward

    plot_rewards(final_returns)

# plot the figure
plt.ylabel("Average reward")
plt.xlabel("Episode")
plt.legend()
plt.title("First-Visit Monte Carlo Control with $\epsilon$-Soft Policy (Track #1)")
plt.savefig("plots/racetrack_esoft_track1.png")
plt.show()

# Final Rollout
episodes = []
for i in range(5):
    episode, _ = agent.rollout(q_values, use_epsilon=False)
    episodes.append(episode)
plot_trajectory(env, episodes)


"""E-Soft Monte Carlo Control Training For Racetrack #2"""
env = Racetrack(version="v2")
agent = ESoftAgent(racetrack_env=env)


# Initialize random policy for all states
state_space_shape = (env.width, env.height, 5, 5) # (loc_x, loc_y, vel_x, vel_y)
action_space_shape = (len(agent.action_space),) # 9 actions
state_action_space_shape = state_space_shape + action_space_shape # (loc_x, loc_y, vel_x, vel_y, d_vel_x, d_vel_y)

gamma = 1.0
epsilon_values = [0.1]
num_trials = 10
num_episodes = 2000

final_returns = np.zeros((num_trials, num_episodes))

# Initialize Figure
plt.figure(figsize=(8, 6))

for epsilon in epsilon_values:
    for trial in range(num_trials):
        q_values = np.zeros(state_action_space_shape) # Initialize Q-values for all state-action pairs to zero
        occurences = np.zeros(state_action_space_shape) # Initialize occurrence counts for all state-action pairs to zero
        
        for ep_num in tqdm(range(num_episodes)): # Run for a certain number of episodes
            episode, first_visits = agent.rollout(q_values)
            total_discounted_reward = 0

            for i in range(1, len(episode)+1): 
                state, action_idx, reward = episode[-i] # Loop through backwards
                total_discounted_reward = reward + gamma * total_discounted_reward

                if first_visits[tuple(state)] == len(episode) - i: # Check if this is the first visit to the state
                    occurences[tuple(state)+(action_idx,)] += 1
                    q_values[tuple(state)+(action_idx,)] += (total_discounted_reward - q_values[tuple(state)+(action_idx,)]) / occurences[tuple(state)+(action_idx,)]

            final_returns[trial, ep_num] = total_discounted_reward

    plot_rewards(final_returns)

# plot the figure
plt.ylabel("Average reward")
plt.xlabel("Episode")
plt.legend()
plt.title("First-Visit Monte Carlo Control with $\epsilon$-Soft Policy (Track #2)")
plt.savefig("plots/racetrack_esoft_track2.png")
plt.show()

# Final Rollout
episodes = []
for i in range(5):
    episode, _ = agent.rollout(q_values, use_epsilon=False)
    episodes.append(episode)
plot_trajectory(env, episodes)

'''
Off-Policy Monte Carlo Control with Epsilon-Soft Policy for Racetrack #1
'''
env = Racetrack(version="v1")
agent = ESoftAgent(racetrack_env=env)

# Initialize random policy for all states
state_space_shape = (env.width, env.height, 5, 5) # (loc_x, loc_y, vel_x, vel_y)
action_space_shape = (len(agent.action_space),) # 9 actions
state_action_space_shape = state_space_shape + action_space_shape # (loc_x, loc_y, vel_x, vel_y, d_vel_x, d_vel_y)

gamma = 1.0
epsilon_values = [0.1]
num_trials = 10
num_episodes = 2000

final_returns = np.zeros((num_trials, num_episodes))
val_final_returns = np.zeros((num_trials, num_episodes))

# Initialize Figure
plt.figure(figsize=(8, 6))

for epsilon in epsilon_values:
    for trial in range(num_trials):
        # q_values = np.zeros(state_action_space_shape) # Initialize Q-values for all state-action pairs to zero
        q_values = np.ones(state_action_space_shape) * -10 # Initialize Q-values for all state-action pairs to random values to encourage exploration
        cumulative_weights = np.zeros(state_action_space_shape) # Initialize cumulative weights for all state-action pairs to zero
        target_policy = np.zeros(state_space_shape, dtype=int) # Initialize target policy for all states to action index 0
        # occurences = np.zeros(state_action_space_shape) # Initialize occurrence counts for all state-action pairs to zero
        
        for ep_num in tqdm(range(num_episodes)): # Run for a certain number of episodes
            episode, first_visits = agent.rollout(q_values)
            total_discounted_reward = 0

            weight = 1.0

            for i in range(1, len(episode)+1): 
                state, action_idx, reward = episode[-i] # Loop through backwards
                total_discounted_reward = reward + gamma * total_discounted_reward

                cumulative_weights[tuple(state)+(action_idx,)] += weight
                q_values[tuple(state)+(action_idx,)] += (weight / cumulative_weights[tuple(state)+(action_idx,)]) * (total_discounted_reward - q_values[tuple(state)+(action_idx,)])
                target_policy[tuple(state)] = np.argmax(q_values[tuple(state)]) # Consistent ArgMax
                if target_policy[tuple(state)] != action_idx: 
                    break # Exits the loop if the action taken is not the action under the target policy
                weight /= (1 - epsilon + (epsilon / len(agent.action_space))) # Update the weight based on the probability of taking the action under the behavior policy 
            
            final_returns[trial, ep_num] = total_discounted_reward

            # Validation: Rollout the episode using the target policy and calculate the total discounted reward to track the learning progress
            val_episode, _ = agent.rollout(q_values, use_epsilon=False)
            val_total_discounted_reward = 0
            for _, _, val_reward in reversed(val_episode):
                val_total_discounted_reward = val_reward + gamma * val_total_discounted_reward

            val_final_returns[trial, ep_num] = val_total_discounted_reward

    plot_rewards(final_returns)
    plot_rewards(val_final_returns)

print("Final Average Reward:", np.mean(final_returns))
# plot the figure
plt.ylabel("Average reward")
plt.xlabel("Episode")
plt.title("Off-Policy Monte Carlo Control with $\epsilon$-Soft Policy (Track #1)")
plt.savefig("plots/racetrack_offpolicy_track1.png")
plt.show()

# Final Rollout
episodes = []
for i in range(5):
    episode, _ = agent.rollout(q_values, use_epsilon=False)
    episodes.append(episode)
plot_trajectory(env, episodes, offpolicy=True)


'''
Off-Policy Monte Carlo Control with Epsilon-Soft Policy for Racetrack #2
'''
env = Racetrack(version="v2")
agent = ESoftAgent(racetrack_env=env)

# Initialize random policy for all states
state_space_shape = (env.width, env.height, 5, 5) # (loc_x, loc_y, vel_x, vel_y)
action_space_shape = (len(agent.action_space),) # 9 actions
state_action_space_shape = state_space_shape + action_space_shape # (loc_x, loc_y, vel_x, vel_y, d_vel_x, d_vel_y)

gamma = 1.0
epsilon_values = [0.1]
num_trials = 10
num_episodes = 2000

final_returns = np.zeros((num_trials, num_episodes))
val_final_returns = np.zeros((num_trials, num_episodes))

# Initialize Figure
plt.figure(figsize=(8, 6))

for epsilon in epsilon_values:
    for trial in range(num_trials):
        # q_values = np.zeros(state_action_space_shape) # Initialize Q-values for all state-action pairs to zero
        q_values = np.ones(state_action_space_shape) * -10 # Initialize Q-values for all state-action pairs to random values to encourage exploration
        cumulative_weights = np.zeros(state_action_space_shape) # Initialize cumulative weights for all state-action pairs to zero
        target_policy = np.zeros(state_space_shape, dtype=int) # Initialize target policy for all states to action index 0
        # occurences = np.zeros(state_action_space_shape) # Initialize occurrence counts for all state-action pairs to zero
        
        for ep_num in tqdm(range(num_episodes)): # Run for a certain number of episodes
            episode, first_visits = agent.rollout(q_values)
            total_discounted_reward = 0

            weight = 1.0

            for i in range(1, len(episode)+1): 
                state, action_idx, reward = episode[-i] # Loop through backwards
                total_discounted_reward = reward + gamma * total_discounted_reward

                cumulative_weights[tuple(state)+(action_idx,)] += weight
                q_values[tuple(state)+(action_idx,)] += (weight / cumulative_weights[tuple(state)+(action_idx,)]) * (total_discounted_reward - q_values[tuple(state)+(action_idx,)])
                target_policy[tuple(state)] = np.argmax(q_values[tuple(state)]) # Consistent ArgMax
                if target_policy[tuple(state)] != action_idx: 
                    break # Exits the loop if the action taken is not the action under the target policy
                weight /= (1 - epsilon + (epsilon / len(agent.action_space))) # Update the weight based on the probability of taking the action under the behavior policy 
            
            final_returns[trial, ep_num] = total_discounted_reward

            # Validation: Rollout the episode using the target policy and calculate the total discounted reward to track the learning progress
            val_episode, _ = agent.rollout(q_values, use_epsilon=False)
            val_total_discounted_reward = 0
            for _, _, val_reward in reversed(val_episode):
                val_total_discounted_reward = val_reward + gamma * val_total_discounted_reward

            val_final_returns[trial, ep_num] = val_total_discounted_reward

    plot_rewards(final_returns)
    plot_rewards(val_final_returns)

print("Final Average Reward:", np.mean(final_returns))
# plot the figure
plt.ylabel("Average reward")
plt.xlabel("Episode")
plt.title("Off-Policy Monte Carlo Control with $\epsilon$-Soft Policy (Track #2)")
plt.savefig("plots/racetrack_offpolicy_track2.png")
plt.show()

# Final Rollout
episodes = []
for i in range(5):
    episode, _ = agent.rollout(q_values, use_epsilon=False)
    episodes.append(episode)
plot_trajectory(env, episodes, offpolicy=True)