import numpy as np

# The GridWorld domain in Example 3.5
class GridWorld(object):
    def __init__(self):
        # define the state space
        self.state_space = [
            [0, 0], [0, 1], [0, 2], [0, 3], [0, 4],
            [1, 0], [1, 1], [1, 2], [1, 3], [1, 4],
            [2, 0], [2, 1], [2, 2], [2, 3], [2, 4],
            [3, 0], [3, 1], [3, 2], [3, 3], [3, 4],
            [4, 0], [4, 1], [4, 2], [4, 3], [4, 4]
        ]

        # define special states
        self.A, self.B = np.array([0, 1]), np.array([0, 3])
        self.A_prime, self.B_prime = np.array([4, 1]), np.array([2, 3])

        self.action_order = ["north", "south", "west", "east"]

        # define the action space
        self.action_space = {
            "north": [-1, 0],
            "south": [1, 0],
            "west": [0, -1],
            "east": [0, 1]
        }

    def reset(self):
        pass

    def step(self, s, a) -> (list, float):
        """
        Args:
            s (list): a list contains the position of the current state
            a (str): name of the action
        """
        # convert the state to numpy array
        s_arr = np.array(s)
        # convert the action to numpy array
        a_arr = np.array(self.action_space[a])

        # compute the next state and reward using the dynamics function
        next_s, r = self.dynamics_func(s_arr, a_arr)

        # return the next state and the reward
        return next_s, r

    def dynamics_func(self, s_arr, a_arr) -> (list, float):
        """
        Args:
            s_arr (numpy.array): numpy array contains the position of the current state
            a_arr (numpy.array): numpy array contains the change of the current state
        """
        # check for special states A and B
        # From state A, all four actions yield a reward of +10 and take the agent to A_prime
        if np.array_equal(s_arr, self.A):
            return self.A_prime.tolist(), 10.0

        # From state B, all actions yield a reward of +5 and take the agent to B prime.
        if np.array_equal(s_arr, self.B):
            return self.B_prime.tolist(), 5.0

        # check for normal states
        # compute the next state position and reward
        next_s = s_arr + a_arr
        if next_s.tolist() not in self.state_space:
            # Actions that would take the agent off the grid leave its location unchanged, but also result in a reward
            # of -1
            return s_arr.tolist(), -1.0
        else:
            # Other actions result in a reward of 0
            return next_s.tolist(), 0.0
        
'''
Helper functions for displaying the optimal state value and optimal policy
'''
        
# Function to print the optimal state value
def print_optimal_state_value(s_v):
    """
    Args:
        s_v (numpy.array): a 2-D numpy array contains the optimal state values with size 5 x 5
    """
    print("=============================")
    print("==  Optimal State Value    ==")
    print("=============================")  
    print(s_v.round(decimals=1))
    print("=============================")

# Function to print the optimal policy 
def print_optimal_policy(s_v, env, ga):
    """
    Args:
        s_v (numpy.array): a 2-D numpy array contains the optimal state value with size 5 x 5
        env (env): the grid-world environment
        ga (float): gamma 
    """
    print("=============================")
    print("==     Optimal Policy      ==")
    print("=============================")
    action_names = list(env.action_space.keys())
    for i in range(5):
        for j in range(5):
            q_v = []
            for a in env.action_space.keys():
                next_s, r = env.step([i, j], a)
                q_v.append(r + ga * s_v[next_s[0], next_s[1]])
            q_v = np.array(q_v)

            actions = np.where(q_v == q_v.max())[0]
            actions = [action_names[a] for a in actions]

            print(f"{[i, j]} = {actions}")
        print("------------------------------")


'''
Value Iteration Algorithm
'''
def run_value_iteration(env, threshold, gamma):
    """
    Args: 
        env: the grid-world environment, we use it to compute:
            - the next state: s'
            - the transition probability: p(s'|s,a)
            - the reward : r
        threshold: threshold determining the estimation threshold
        gamma: the discounted factor
        
        Note: we use the vanilla implementation, where we maintain two separate numpy arrays to store the
              state value and the updated state value. 
    """
    # initialize the state value to be 0
    state_value = np.zeros((5, 5))
    
    # iteration counter
    iter_counter = 0

    # loop forever
    while True:
        # Logic: assuming the value iteration should be terminated for the current iteration
        # unless there exists one state whose value estimation error > threshold. i.e. (abs(new_v - old_v) > threshold)
        is_terminal = True

        # save the new state value
        new_state_value = np.zeros_like(state_value)

        # loop all states 
        # each state is the position of the agent in the grid. e.g., [i, j]
        # where i, j in [0, 4]
        for i in range(5):
            for j in range(5):
                # obtain the current state value estimation
                old_v = state_value[i, j]

                # Loop over all actions to find the optimal action that yields the maximum state value estimation.
                new_v = float("-inf")
                for action in env.action_space.keys():
                    next_s, r = env.dynamics_func(np.array([i, j]), np.array(env.action_space[action]))
                    v_s_prime = r + (gamma * state_value[next_s[0], next_s[1]])

                    # update the new_v if it is greater than the current new_v
                    if v_s_prime > new_v:
                        new_v = v_s_prime
                
                # check the termination
                if abs(new_v - old_v) > threshold:
                    is_terminal = False
                
                """DO NOT CHANGE BELOW"""
                # store the updated value in the new_state_value
                new_state_value[i, j] = new_v

        # update the current state value with the updated values
        state_value = new_state_value.copy()

        iter_counter += 1
        
        # terminate the loop 
        if is_terminal:
            break

    print(f"Value Iteration converged after {iter_counter} iterations.")

    return state_value


# run value iteration

# create the envrionment
my_grid = GridWorld()
my_grid.reset()

# threshold determining the accuracy of the estimation
threshold = 1e-3

# discounted factor
gamma = 0.8

# run the value iteration
state_value = run_value_iteration(my_grid, threshold, gamma)

# print the optimal state value
print_optimal_state_value(state_value)

# print the optimal policy
print_optimal_policy(state_value, my_grid, gamma)



'''
Policy Iteration Algorithm
'''


def policy_evaluation(env, policy, threshold, gamma):
    """
    Args:
        env: the grid-world environment, we use it to compute:
            - the next state: s'
            - the transition probability: p(s'|s,a)
            - the reward : r
        policy (numpy.array): a 2-D numpy array stores the action to take at each location.
        threshold (float): threshold determining the estimation threshold
        gamma (float): the discounted factor
        
        Note: we use the vanilla implementation, where we maintain two separate numpy arrays to store the
              state value and the updated state value. 
    """
    # initialize the state values
    state_value = np.zeros((5, 5))

    # start evaluate the current policy
    while True:
        # set terminal flag similar to the value iteration flag.
        is_terminal = True

        # new state value
        new_state_value = np.zeros_like(state_value)

        # loop all states
        for i in range(5):
            for j in range(5):
                # store the old state value
                old_v = state_value[i, j]
                
                # update the state value using the equation 4.5
                # Hints: how many next state are there given a deterministic policy and 
                # a deterministic environment.   
                action = env.action_order[policy[i, j]]
                next_s, r = env.dynamics_func(np.array([i, j]), np.array(env.action_space[action]))
                new_v = r + (gamma * state_value[next_s[0], next_s[1]])

                # check termination
                if abs(new_v - old_v) > threshold:
                    is_terminal = False

                # store the updated state value for state [i, j]
                new_state_value[i, j] = new_v

        # update state value
        state_value = new_state_value.copy()

        # check termination
        if is_terminal:
            break

    return state_value


def policy_improvement(env, policy, state_value, gamma):
    # set the policy improvement flag
    policy_stable = True
    
    # loop all states
    for i in range(5):
        for j in range(5):
            # store the old action
            old_action = policy[i, j]


            # compute a new greedy action based on the latest state value
            new_action = None
            new_q_v = float("-inf")

            for action_idx, action in enumerate(env.action_space.keys()):
                next_s, r = env.dynamics_func(np.array([i, j]), np.array(env.action_space[action]))
                q_v = r + (gamma * state_value[next_s[0], next_s[1]])

                # update the new_action if it is greater than the current q_v
                if q_v > new_q_v:
                    new_q_v = q_v
                    new_action = action_idx
            
            # check if the policy is stable
            if old_action != new_action:
                policy_stable = False

            # update the policy with the new greedy policy
            policy[i, j] = new_action

    return policy.astype(int), policy_stable

"""Running policy iteration"""

my_grid = GridWorld()
my_grid.reset()

# threshold and gamma
threshold, gamma = 1e-3, 0.8

# initialize a random policy
policy = np.random.randint(low=0, high=4, size=(5, 5))

# run policy iteration
while True:
    # policy evaluation
    state_value = policy_evaluation(my_grid, policy, threshold, gamma)

    # policy improvement
    policy, policy_stable = policy_improvement(my_grid, policy, state_value, gamma)

    # check if policy is stable
    if policy_stable:
        break

# print the optimal state value
print_optimal_state_value(state_value)

# print the optimal policy
print_optimal_policy(state_value, my_grid, gamma)