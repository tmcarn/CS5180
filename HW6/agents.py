import numpy as np
from collections.abc import Callable

from four_rooms_env import FourRooms

class FuncApproxSARSA(object):
    def __init__(self, env: FourRooms, feature_fn: Callable, n_features: int, alpha: float, epsilon: float, gamma: float, timeout: int):
        # define the parameters
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

        # environment
        self.env = env

        # define the feature function        
        self.feature_fn = feature_fn

        # define the weight vector for linear function approximation
        # self.w = np.random.random(n_features)
        self.w = np.zeros(n_features)

        # define the timeout
        self.timeout = timeout

    def q_function(self, state, action):
        return np.dot(self.w, self.feature_fn(state, action))

    def behavior_policy(self, state):
        sample = np.random.rand()
        if sample < self.epsilon:
            # explore: choose a random action
            action = np.random.choice(list(self.env.action_space.keys()))
       
        else:            
            # exploit: choose the action with the highest Q value
            best_q_value = -np.inf
            for action in self.env.action_space.keys():
                q_value = self.q_function(state, action)
                if q_value > best_q_value:
                    best_q_value = q_value
                    best_action = action
            
            action = best_action
        
        return action

    def update(self, s, a, r, s_prime, a_prime):
        # a_index = list(self.env.action_space.keys()).index(a)
        # a_prime_index = list(self.env.action_space.keys()).index(a_prime)
        self.w += self.alpha * (r + self.gamma * self.q_function(s_prime, a_prime) - self.q_function(s, a)) * self.feature_fn(s, a) # Gradient descent update for linear function approximation
    
    def rollout(self):
        rolling = True
        state, reward, terminated = self.env.reset()
        action = self.behavior_policy(state)
        while rolling:
            next_state, reward, terminated = self.env.step(state, action)
            next_action = self.behavior_policy(next_state)
            self.update(state, action, reward, next_state, next_action)
            state = next_state
            action = next_action
            if (self.env.t > self.timeout) or terminated: # Episode timeout or goal reached
                rolling = False

        return self.env.t
        

    def run(self):
        time_steps = 0
        episodes_hist = np.zeros(self.timeout)
        episodes = 0

        while time_steps < self.timeout:
            # time_steps_start = time_steps_end = time_steps
            time_steps_end = time_steps + self.rollout()
            episodes += 1
            episodes_hist[time_steps:time_steps_end] = episodes

            time_steps = time_steps_end
        
        return episodes_hist
       
class QLearning(object):
    def __init__(self, env, alpha, epsilon, gamma, timeout):
        # define the parameters
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

        # environment
        self.env = env

        # define the Q value table
        state_action_shape = self.env.state_space_shape + (len(self.env.action_space),)
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
        state, reward, terminated = self.env.reset()
        action = self.behavior_policy(state)
        while rolling:
            next_state, reward, terminated = self.env.step(state, action)
            self.update(state, action, reward, next_state)
            state = next_state
            action = self.behavior_policy(state)
            if (self.env.t > self.timeout) or terminated: # Episode timeout or goal reached
                rolling = False

        return self.env.t
        

    def run(self):
        time_steps = 0
        episodes_hist = np.zeros(self.timeout)
        episodes = 0

        while time_steps < self.timeout:
            # time_steps_start = time_steps_end = time_steps
            time_steps_end = time_steps + self.rollout()
            episodes += 1
            episodes_hist[time_steps:time_steps_end] = episodes

            time_steps = time_steps_end
        
        return episodes_hist

class FuncApproxSARSA(object):
    def __init__(self, env: FourRooms, feature_fn: Callable, n_features: int, alpha: float, epsilon: float, gamma: float, timeout: int):
        # define the parameters
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

        # environment
        self.env = env

        # define the feature function        
        self.feature_fn = feature_fn

        # define the weight vector for linear function approximation
        self.w = np.random.random(n_features)
        # self.w = np.zeros(n_features)

        # define the timeout
        self.timeout = timeout

    def q_function(self, state, action):
        return np.dot(self.w, self.feature_fn(state, action))

    def behavior_policy(self, state):
        sample = np.random.rand()
        if sample < self.epsilon:
            # explore: choose a random action
            action = np.random.choice(list(self.env.action_space.keys()))
       
        else:            
            # exploit: choose the action with the highest Q value
            best_q_value = -np.inf
            for action in self.env.action_space.keys():
                q_value = self.q_function(state, action)
                if q_value > best_q_value:
                    best_q_value = q_value
                    best_action = action
            action = best_action
        
        return action

    def update(self, s, a, r, s_prime, a_prime):
        # a_index = list(self.env.action_space.keys()).index(a)
        # a_prime_index = list(self.env.action_space.keys()).index(a_prime)
        self.w += self.alpha * (r + self.gamma * self.q_function(s_prime, a_prime) - self.q_function(s, a)) * self.feature_fn(s, a) # Gradient descent update for linear function approximation
    
    def rollout(self):
        rolling = True
        state, reward, terminated = self.env.reset()
        action = self.behavior_policy(state)
        while rolling:
            next_state, reward, terminated = self.env.step(state, action)
            next_action = self.behavior_policy(next_state)
            self.update(state, action, reward, next_state, next_action)
            state = next_state
            action = next_action
            if (self.env.t > self.timeout) or terminated: # Episode timeout or goal reached
                rolling = False

        return self.env.t
        

    def run(self):
        time_steps = 0
        episodes_hist = np.zeros(self.timeout)
        episodes = 0

        while time_steps < self.timeout:
            # time_steps_start = time_steps_end = time_steps
            time_steps_end = time_steps + self.rollout()
            episodes += 1
            episodes_hist[time_steps:time_steps_end] = episodes

            time_steps = time_steps_end
        
        return episodes_hist
     