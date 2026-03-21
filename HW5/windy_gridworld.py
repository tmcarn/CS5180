from matplotlib.pylab import gamma
import numpy as np
import matplotlib.pyplot as plt

class WindyGridWorld(object):
    def __init__(self, enable_king_move=False, enable_no_move=False, random_wind=False):
        # define the state space
        self.state_space = np.zeros((7, 10))

        # define the start state
        self.start_state = [3, 0]

        # define the goal state
        self.goal_state = [3, 7]

        self.random_wind = random_wind

        # define the wind
        self.wind = np.array([0, 0, 0, 1, 1, 1, 2, 2, 1, 0], dtype=int)

        # define the action space
        self.action_space = {
            "up": np.array([-1, 0]),
            "down": np.array([1, 0]),
            "left": np.array([0, -1]),
            "right": np.array([0, 1])
        }
        
        # Enable King's moves
        if enable_king_move:
            self.action_space["up_left"] = np.array([-1, -1])
            self.action_space["up_right"] = np.array([-1, 1])
            self.action_space["down_left"] = np.array([1, -1])
            self.action_space["down_right"] = np.array([1, 1])
            
            if enable_no_move:
                self.action_space["no_move"] = np.array([0, 0])
            
                
        # track the current state, time step, and action
        self.state = None
        self.t = None
        self.act = None

    def reset(self):
        # reset the agent to the start state
        self.state = self.start_state
        # reset the time step tracker
        self.t = 0
        # reset the action tracker
        self.act = None
        # reset the terminal flag
        terminated = False
        return self.state, terminated

    def step(self, act):

        if self.random_wind:
            wind = self.wind + np.random.choice([-1, 0, 1], size=self.wind.shape)
        else:
            wind = self.wind

        next_state = self.state + self.action_space[act] - np.array([wind[self.state[1]], 0])
        # clip the next state to be within the grid boundaries
        next_state[0] = np.clip(next_state[0], 0, self.state_space.shape[0] - 1)
        next_state[1] = np.clip(next_state[1], 0, self.state_space.shape[1] - 1)
        
        if (next_state == self.goal_state).all():
            reward = 0.0
            terminated = True
        
        else:
            reward = -1.0
            terminated = False
        
        self.state = next_state
        self.t += 1
        self.act = act
        
        return self.state, reward, terminated

    def render(self):
        # plot the agent and the goal
        # agent = 1
        # goal = 2
        plot_arr = self.grid.copy()
        plot_arr[self.state[0], self.state[1]] = 1.0
        plot_arr[self.goal_state[0], self.goal_state[1]] = 2.0
        plt.clf()
        fig, arr = plt.subplots(1, 1)
        arr.set_title(f"state={self.state}, act={self.act}")
        arr.imshow(plot_arr)
        plt.show(block=False)
        plt.pause(1)
        plt.close(fig)

class SARSA(object):
    def __init__(self, env: WindyGridWorld, alpha, epsilon, gamma, timeout):
        # define the parameters
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

        # environment
        self.env = env

        # define the Q value table
        state_action_shape = self.env.state_space.shape + (len(self.env.action_space),)
        self.Q = np.random.random(state_action_shape)

        # define the timeout
        self.timeout = timeout

    def behavior_policy(self, state):
        sample = np.random.rand()
        if sample < self.epsilon:
            # explore: choose a random action
            action = np.random.choice(list(self.env.action_space.keys()))
       
        else:            
            # exploit: choose the action with the highest Q value
            action_index = np.argmax(self.Q[tuple(state)])
            action = list(self.env.action_space.keys())[action_index]
        
        return action

    def update(self, s, a, r, s_prime, a_prime):
        a_index = list(self.env.action_space.keys()).index(a)
        a_prime_index = list(self.env.action_space.keys()).index(a_prime)
        self.Q[tuple(s) + (a_index,)] += self.alpha * (r + self.gamma * self.Q[tuple(s_prime) + (a_prime_index,)] - self.Q[tuple(s) + (a_index,)])

    def rollout(self):
        rolling = True
        state, terminated = self.env.reset()
        action = self.behavior_policy(state)
        while rolling:
            next_state, reward, terminated = self.env.step(action)
            next_action = self.behavior_policy(next_state)
            self.update(state, action, reward, next_state, next_action)
            state = next_state
            action = next_action
            if (self.env.t > self.timeout) or terminated: # Episode timeout or goal reached
                rolling = False

        return self.env.t
        

    def run(self):
        max_time_steps = 8000
        time_steps = 0
        episodes_hist = np.zeros(max_time_steps)
        episodes = 0

        while time_steps < max_time_steps:
            # time_steps_start = time_steps_end = time_steps
            time_steps_end = time_steps + self.rollout()
            episodes += 1
            episodes_hist[time_steps:time_steps_end] = episodes

            time_steps = time_steps_end
        
        return episodes_hist
    
class ExpectedSARSA(object):
    def __init__(self, env: WindyGridWorld, alpha, epsilon, gamma, timeout):
        # define the parameters
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

        # environment
        self.env = env

        # define the Q value table
        state_action_shape = self.env.state_space.shape + (len(self.env.action_space),)
        self.Q = np.random.random(state_action_shape)

        # define the timeout
        self.timeout = timeout

    def behavior_policy(self, state):
        sample = np.random.rand()
        if sample < self.epsilon:
            # explore: choose a random action
            action = np.random.choice(list(self.env.action_space.keys()))
       
        else:            
            # exploit: choose the action with the highest Q value
            action_index = np.argmax(self.Q[tuple(state)])
            action = list(self.env.action_space.keys())[action_index]
        
        return action

    def update(self, s, a, r, s_prime, a_prime):
        a_index = list(self.env.action_space.keys()).index(a)
        a_prime_index = list(self.env.action_space.keys()).index(a_prime)
        
        optimal_action_index = np.argmax(self.Q[tuple(s_prime)])
        
        expectation = 0
        for i in range(len(self.env.action_space)):
            if i == optimal_action_index:
                expectation += (1 - (self.epsilon / len(self.env.action_space))) * self.Q[tuple(s_prime) + (i,)]
            else:
                expectation += (self.epsilon / len(self.env.action_space)) * self.Q[tuple(s_prime) + (i,)]

        self.Q[tuple(s) + (a_index,)] += self.alpha * (r + self.gamma * expectation - self.Q[tuple(s) + (a_index,)])

    def rollout(self):
        rolling = True
        state, terminated = self.env.reset()
        action = self.behavior_policy(state)
        while rolling:
            next_state, reward, terminated = self.env.step(action)
            next_action = self.behavior_policy(next_state)
            self.update(state, action, reward, next_state, next_action)
            state = next_state
            action = next_action
            if (self.env.t > self.timeout) or terminated: # Episode timeout or goal reached
                rolling = False

        return self.env.t
        

    def run(self):
        max_time_steps = 8000
        time_steps = 0
        episodes_hist = np.zeros(max_time_steps)
        episodes = 0

        while time_steps < max_time_steps:
            # time_steps_start = time_steps_end = time_steps
            time_steps_end = time_steps + self.rollout()
            episodes += 1
            episodes_hist[time_steps:time_steps_end] = episodes

            time_steps = time_steps_end
        
        return episodes_hist
    
class QLearning(object):
    def __init__(self, env: WindyGridWorld, alpha, epsilon, gamma, timeout):
        # define the parameters
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

        # environment
        self.env = env

        # define the Q value table
        state_action_shape = self.env.state_space.shape + (len(self.env.action_space),)
        self.Q = np.random.random(state_action_shape)

        # define the timeout
        self.timeout = timeout

    def behavior_policy(self, state):
        sample = np.random.rand()
        if sample < self.epsilon:
            # explore: choose a random action
            action = np.random.choice(list(self.env.action_space.keys()))
       
        else:            
            # exploit: choose the action with the highest Q value
            action_index = np.argmax(self.Q[tuple(state)])
            action = list(self.env.action_space.keys())[action_index]
        
        return action

    def update(self, s, a, r, s_prime):
        a_index = list(self.env.action_space.keys()).index(a)
        self.Q[tuple(s) + (a_index,)] += self.alpha * (r + self.gamma * np.max(self.Q[tuple(s_prime)]) - self.Q[tuple(s) + (a_index,)])

    def rollout(self):
        rolling = True
        state, terminated = self.env.reset()
        action = self.behavior_policy(state)
        while rolling:
            next_state, reward, terminated = self.env.step(action)
            self.update(state, action, reward, next_state)
            state = next_state
            action = self.behavior_policy(state)
            if (self.env.t > self.timeout) or terminated: # Episode timeout or goal reached
                rolling = False

        return self.env.t
        

    def run(self):
        max_time_steps = 8000
        time_steps = 0
        episodes_hist = np.zeros(max_time_steps)
        episodes = 0

        while time_steps < max_time_steps:
            # time_steps_start = time_steps_end = time_steps
            time_steps_end = time_steps + self.rollout()
            episodes += 1
            episodes_hist[time_steps:time_steps_end] = episodes

            time_steps = time_steps_end
        
        return episodes_hist

class ESoftAgent(object):  
    def __init__(self, env, policy=None, epsilon=0.1, gamma=1, alpha=0.5, timeout=8000):
        self.env = env
        self.action_space = self.env.action_space
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.timeout = timeout

        self.policy = policy # fixed plolicy used to evaluate (Q-Values from SARSA)

        # define the Q value table
        state_action_shape = self.env.state_space.shape + (len(self.env.action_space),)
        self.Q = np.random.random(state_action_shape)
        self.occurences = np.zeros(state_action_shape) # Initialize occurrence counts for all state-action pairs to zero


    def e_soft_policy(self, state, policy):
        num_actions = len(self.action_space)
        dist = np.ones(num_actions) * self.epsilon / num_actions
        greedy_action = np.argmax(policy[tuple(state)])
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
        
        
    def rollout(self, use_epsilon=True, evaluation=False):
        """
        Input args:
            policy: a numpy array of shape (state_space_size, ) where each element is an integer in [0, 3] representing the action index.
            max_episode_length: an integer representing the maximal length of an episode.
        Output args:
            episode: a list of tuples. Each tuple is of the form (state, action, reward).
        """
        # reset the environment to get the start state
        state, terminated = self.env.reset()
        episode = []
        first_visit = {}

        for step in range(self.timeout):

            if use_epsilon:
                if evaluation:
                    action_idx = self.e_soft_policy(state, self.policy) # get the action index from the pretrained policy
                else:
                    action_idx = self.e_soft_policy(state, self.Q) # get the action index from the current policy
            
            else:
                action_idx = np.argmax(self.Q[tuple(state)]) # get the greedy action index from the Q-values
            
            action = list(self.action_space)[action_idx] # convert the action index to action name
            next_state, reward, terminated = self.env.step(action) # take the action and observe the next state and reward

            if tuple(state) not in first_visit:
                first_visit[tuple(state)] = step
            
            episode.append((state, action_idx, reward)) # store the transition in the episode
            state = next_state # update the current state

            if terminated:
                break

        return episode, first_visit, step
    
    def run(self):
        time_steps = 0
        episodes_hist = np.zeros(self.timeout)
        episodes = 0

        while time_steps < self.timeout:
            episode, first_visits, step = self.rollout()

            total_discounted_reward = 0
            for i in range(1, len(episode)+1): 
                state, action_idx, reward = episode[-i] # Loop through backwards
                total_discounted_reward = reward + self.gamma * total_discounted_reward

                if first_visits[tuple(state)] == len(episode) - i: # Check if this is the first visit to the state
                    self.occurences[tuple(state)+(action_idx,)] += 1
                    self.Q[tuple(state)+(action_idx,)] += (total_discounted_reward - self.Q[tuple(state)+(action_idx,)]) / self.occurences[tuple(state)+(action_idx,)]

            episodes += 1
            episodes_hist[time_steps:time_steps+step] = episodes
            time_steps += step

        return episodes_hist
    
    def estimate(self, n, record_state=None, v=None):

        V = np.zeros(self.env.state_space.shape)
        targets = []

        for _ in range(n):
            episode, first_visits, step = self.rollout(use_epsilon=True, evaluation=True) # Generate an episode using the epsilon-soft policy
            total_discounted_reward = 0
            for i in range(1, len(episode)+1): 
                state, _, reward = episode[-i] # Loop through backwards
                total_discounted_reward = reward + self.gamma * total_discounted_reward

                if first_visits[tuple(state)] == len(episode) - i: # Check if this is the first visit to the state
                    if record_state is None:
                        V[tuple(state)] += self.alpha * (total_discounted_reward - V[tuple(state)])
                    
                    elif tuple(state) == tuple(record_state):
                        targets.append(total_discounted_reward)

        return V if record_state is None else targets

def plot_curves(arr_list, legend_list, color_list, ylabel):
    """
    Args:
        arr_list (list): list of results arrays to plot
        legend_list (list): list of legends corresponding to each result array
        color_list (list): list of color corresponding to each result array
        ylabel (string): label of the Y axis

    Make sure the elements in the arr_list, legend_list, and color_list are associated with each other correctly.
    Do not forget to change the ylabel for different plots.
    """
    # Clear the current figure
    plt.clf()
    fig, ax = plt.subplots(figsize=(12, 8))

    # Set labels
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time Steps")

    # Plot results
    h_list = []
    for arr, legend, color in zip(arr_list, legend_list, color_list):
        # Compute the mean and standard error while ignoring NaN values
        mean_arr = np.nanmean(arr, axis=0)
        arr_err = np.nanstd(arr, axis=0) / np.sqrt(np.sum(~np.isnan(arr), axis=0))
        
        # Plot the mean
        h, = ax.plot(range(len(mean_arr)), mean_arr, color=color, label=legend)
        
        # Plot the confidence band
        arr_err = 1.96 * arr_err  # 95% confidence interval
        ax.fill_between(range(len(mean_arr)),
                        mean_arr - arr_err,
                        mean_arr + arr_err,
                        alpha=0.3, color=color)
        # Save the plot handle
        h_list.append(h)

    # Set the title (adjust as needed)
    ax.set_title("Windy Gridworld Results")
    ax.legend(handles=h_list)
    plt.show()

def run_on_policy_td_control(run_num, timeout):

    enable_king_move_actions = False
    enable_no_move_actions = False
    
    # create the environment
    env = WindyGridWorld(enable_king_move=enable_king_move_actions, enable_no_move=enable_no_move_actions)

    # parameters
    epsilon = 0.1
    alpha = 0.5
    gamma = 1.0



    # create the expected SARSA
    expected_sarsa_results_list = []
    for _ in range(run_num):
        # run for each trial
        controller_expected_sarsa = ExpectedSARSA(env, alpha, epsilon, gamma, timeout)
        episodes = controller_expected_sarsa.run()
        # append the results
        expected_sarsa_results_list.append(episodes[0:8000])
        
    # create the SARSA
    sarsa_results_list = []
    for _ in range(run_num):
        # run for each trial
        controller_sarsa = SARSA(env, alpha, epsilon, gamma, timeout)
        episodes = controller_sarsa.run()
        # append the results
        sarsa_results_list.append(episodes[0:8000])

    # create the Q learning
    q_learning_results_list = []
    for _ in range(run_num):
        # run for each trial
        controller_q_learning = QLearning(env, alpha, epsilon, gamma, timeout)
        episodes = controller_q_learning.run()
        # append the results
        q_learning_results_list.append(episodes[0:8000])

    # create the Q learning
    mc_learning_results_list = []
    for _ in range(run_num):
        # run for each trial
        controller_mc_learning = ESoftAgent(env, epsilon, gamma, timeout)
        episodes = controller_mc_learning.run()
        # append the results
        mc_learning_results_list.append(episodes[0:8000])

    sarsa_array = np.array(sarsa_results_list)
    expected_sarsa_array = np.array(expected_sarsa_results_list)
    q_learning_array = np.array(q_learning_results_list)
    mc_learning_array = np.array(mc_learning_results_list)

    plot_curves(
        [sarsa_array, q_learning_array, expected_sarsa_array, mc_learning_array],
        ['SARSA', 'Q-learning', 'Expected SARSA', 'MC Learning'],
        ['r', 'g', 'b', 'm'],
        "Episodes"
    )
    

if __name__ == "__main__":
    # set randomness
    np.random.seed(1234)

    # trial number
    trial_num = 10
    # maximal time steps
    max_time_steps = 8000
    
    # run SARSAs, Q Learning, and MC
    run_on_policy_td_control(trial_num, max_time_steps)