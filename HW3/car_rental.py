import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt

class JackCarRental(object):
    def __init__(self, max_car_num=20, modified_reward = False):
        # Define the state space (the state is the number of cars at each location at the end of the day)
        self.max_num = max_car_num
        # Define the table for the state value
        self.state_value = np.zeros((self.max_num + 1, self.max_num + 1))

        self.modified_reward = modified_reward

        # Define the action space (the number of the cars that Jack plans to move from lot1 to lot2)
        # For example, if a >= 0, moving |a| cars from lot1 to lot2; Otherwise, moving |a| cars
        # from lot2 to lot1.
        self.action_space = np.linspace(start=-5, stop=5, num=11, dtype=int)

        # Define a truncated version of the Poisson distribution
        # Note that, Poisson distribution is a discrete distribution for infinite possible values.
        # In this problem, we only want to consider the values with probability bigger than a threshold.
        # We ignore other values because their probabilities of happening are too small to be considered.
        self.truncated_threshold = 1e-6
        
        self.request_distribution_lot1 = TruncatedPoissonDistribution(3, self.truncated_threshold)
        self.return_distribution_lot1 = TruncatedPoissonDistribution(3, self.truncated_threshold)
        
        self.request_distribution_lot2 = TruncatedPoissonDistribution(4, self.truncated_threshold)
        self.return_distribution_lot2 = TruncatedPoissonDistribution(2, self.truncated_threshold)

        # Pre-compute the transition function p(s'|s, a) and r(s, a) and store them as numpy arrays
        self.p_lot1, self.r_lot1 = self.open_to_close(self.request_distribution_lot1, self.return_distribution_lot1)
        self.p_lot2, self.r_lot2 = self.open_to_close(self.request_distribution_lot2, self.return_distribution_lot2)

    def reset(self):
        pass

    def step(self, s, a):
        """ Step function: it returns the next state given the current state and the action
        Args:
            s (list of two integers): a list variable contains two integers that represent the number of the cars
                                      at lot1 and lot2 by the end of the day, respectively.
            a (int): an integer variable represents the number of cars that Jack wants to move from lot1 to lot2.
                     Note, if a >= 0, Jack moves |a| cars from lot1 to lot2; Otherwise, he moves the cars conversely.
        Returns:
            next_s (list of two integers): a list variable contains two integers that represent the number of the cars
                                           at lot1 and lot2 by the end of the day, respectively.
            reward (float): a float variable represents the rewards by taking action a at state s. Note that: it
                            aggregates the cost of moving cars and gain of renting out cars.
            prob (float) between (0, 1): a float variable represents the probability of transit to next_s from s by
                                         taking a. In other words, prob = p(next_s|s, a)
        """
        """Business ends"""
        # Note the state is the number of cars at the two lots by the end of the day
        car_num_lot1, car_num_lot2 = s

        """Car moving overnight"""
        # We define the action as moving cars from lot1 to lot2:
        # if a >= 0: moving cars from lot1 to lot2; Otherwise moving car from lot2 to lot1
        # update the cars in both slots (overnight)
        move_car_num = self.move_car(s, a)
        car_num_lot1_after_move = car_num_lot1 - move_car_num
        car_num_lot2_after_move = car_num_lot2 + move_car_num

        """Business starts"""
        # compute the requests for lot1 and lot2
        request_car_lot1_num = self.request_distribution_lot1.sample()
        request_car_lot2_num = self.request_distribution_lot2.sample()

        # compute the return for lot1 and lot2
        return_car_lot1_num = self.return_distribution_lot1.sample()
        return_car_lot2_num = self.return_distribution_lot2.sample()

        """Business ends"""
        # compute the number of cars at each lot
        car_num_lot1_new = self.update_car_num(car_num_lot1_after_move, request_car_lot1_num, return_car_lot1_num)
        car_num_lot2_new = self.update_car_num(car_num_lot2_after_move, request_car_lot2_num, return_car_lot2_num)

        # compute the p(s'|s, a)
        prob = self.p_lot1[car_num_lot1_after_move][car_num_lot1_new] * \
               self.p_lot2[car_num_lot2_after_move][car_num_lot2_new]

        # compute the reward
        if self.modified_reward:
            reward = self.compute_reward_modified(move_car_num, car_num_lot1_after_move, car_num_lot2_after_move)
        else:
            reward = self.compute_reward(move_car_num, car_num_lot1_after_move, car_num_lot2_after_move)

        # compute the next state
        next_s = [car_num_lot1_new, car_num_lot2_new]

        # the return is the next_state, reward (cost + expected incomes from lot1 and lot2), p(s'|s, a)
        return next_s, reward, prob

    def compute_reward(self, moved_cars, car_num_lot1, car_num_lot2):
        """ Compute the total reward between two consecutive days, it contains the following stages:
                - When business ends, Jack moves |moved_cars| from lot1 to lot2. The reward is negative and equals to
                  -2 * abs(moved_cars). Note, moved_cars is the actual movable cars.
                - After car moving, when business starts the next day, requests come independently in lot1 and lot2.
                  Therefore, the reward is positive. We aggregate the reward from lot1 and lot2 given the number of cars
                  at lot1 and lot2 after car moving.
            Args:
                moved_cars (int): number of cars that are actually moved from lot1 to lot2 by Jack.
                                  Note: if moved_cars >=0, Jack moves car from lot1 to lot2; Otherwise, he moves
                                  from lot2 to lot1.
                car_num_lot1 (int): number of cars at lot1 after the car moving
                car_num_lot2 (int): number of cars at lot2 after the car moving
        """
        # compute the cost + expected reward at lot1 + expected reward at lot2
        return -2 * abs(moved_cars) + self.r_lot1[car_num_lot1] + self.r_lot2[car_num_lot2]

    def compute_reward_modified(self, moved_cars, car_num_lot1, car_num_lot2):
        """ Besides the cost of moving cars between lot1 and lot2, the reward function is adjusted based on the
            following modifications:
                - One car can be moved from lot1 to lot2 for free.
                - If num_car_after_move > 10, additional $4 are charged at each time lot regardless of how many cars
                - $2 per car moving fee
                - $10 per car renting income
        """
        reward = self.r_lot1[car_num_lot1] + self.r_lot2[car_num_lot2]

        num_cars_after_move_lot1 = car_num_lot1 - moved_cars
        num_cars_after_move_lot2 = car_num_lot2 + moved_cars
        
        # If num_car_after_move > 10, additional $4 are charged at each time lot regardless of how many cars
        if num_cars_after_move_lot1 > 10:
            reward -= 4
        if num_cars_after_move_lot2 > 10:
            reward -= 4

        # Penalty for moving cars
        reward -= 2 * abs(moved_cars)

        # One car can be moved from lot1 to lot2 for free
        if moved_cars > 0: # Positive indicates moving from lot1 to lot2
            reward += 2 # refund $2 for one car moving from lot1 to lot2

        return reward

    def open_to_close(self, request_distribution, return_distribution):
        """ Considering one lot (lot1 or lot2), the possible number of cars when business opens is [0, 25]
            25 = 20 (maximal number when business closes) + (5 cars moved by Jack).
            When business ends, the possible number of cars is [0, 20].

            We consider another form of transition probability p(s'|s, a) and reward function r(s, a).

            Since the dynamics for the overnight car moving is deterministic (i.e., by default, Jack moves certain
            amount of cars between lot1 and lot2 once), the stochasticity comes from the request/return from the
            customers. Therefore, we can re-write the transition function as:
                p(lot_end|lot_end_yesterday, a) = p(lot_end|lot_open) * p(lot_open|lot_end_yesterday,a) = p(lot_end|lot_open),
                where y is the number of cars at one lot after Jack moves the cars p(lot_open|lot_end_yesterday,a) = 1.

            The stochasticity mainly comes from the second stage we call it which is modeled in this open_to_close function.
        Args:
            request_distribution (TruncatedPoissonDistribution): a truncated Possion distribution under a threshold for requesting
            return_distribution (TruncatedPoissonDistribution): a truncated Possion distribution under a threshold for returning

        Returns:
            p_arr (np.array): it stores p(car_num_end|car_num_after_move) for all possible combinations. The shape is
                              26 x 21.
            r_arr (np.array): it stores the expected reward for each car_num_after_move in [0, 25]. It also equals to
                              r(s, a) since the moving action is deterministic.
        """
        # Numpy array to store the factorized transition function
        # state_open take integer values from [0, 25]
        # state_end take integer values from [0, 20]
        # p(s_end|s_end_yesterday, a) = p(s_end|s_open) since the moving action is deterministic.
        p_arr = np.zeros((26, 21))
        # Numpy array to store the reward r(s, a) = r(s_open) because the moving the deterministic
        r_arr = np.zeros(26)

        for request_num, request_prob in request_distribution:
            # the positive rewards only come from the request
            # and the initial number of cars will matter
            # since the reward contains stochasticity, we compute the expectation
            # of the reward for each possible state at one parking lot
            for n in range(26):
                r_arr[n] += request_prob * 10 * min(n, request_num)

            # compute the transition function
            for return_num, return_prob in return_distribution:
                for n in range(26):
                    # compute the new n by the end of the day
                    new_n = self.update_car_num(n, request_num, return_num)
                    # update the probability incrementally
                    p_arr[n][new_n] += request_prob * return_prob

        return p_arr, r_arr

    @staticmethod
    def update_car_num(cars_num, request_cars_num, return_cars_num):
        """ Update the number of cars in lot1/lot2 after requesting and returning
        Args:
            cars_num (int): Number of the cars at lot1/lot2 when the business starts
            request_cars_num (int): Number of the cars requested by the customers
            return_cars_num (int): Number of the cars returned by the customers
        """
        return min(max(cars_num - request_cars_num, 0) + return_cars_num, 20)

    @staticmethod
    def move_car(state, action):
        """ Compute the actual number of cars to be moved from lot1 to lot2
        Args:
            state (list): a list variable contains the number of cars (int) at lot1 and lot2.
            action (int): an integer between [-5, 5]
        """
        # Note: action is the number of cars Jack wants to move from lot1 to lot2. But the actual number should follow
        # the condition [-1 * car_num_lot2, car_num_lot1]
        return int(np.clip(action, a_min=-state[1], a_max=state[0]))
    
    
class TruncatedPoissonDistribution(object):
    def __init__(self, mean, threshold):
        # Check the validation of the mean and threshold
        assert isinstance(mean, int), mean > 0
        assert 0 < threshold < 1.0

        # Store the mean of the Poisson distribution
        self.mean = mean
        # Store the threshold of p(k). Only k with p(k) > threshold would be considered
        self.truncated_threshold = threshold

        # create a list to store the selected discrete values and their probabilities
        self.truncated_values, self.truncated_prob = self.truncate_poisson()

    def truncate_poisson(self):
        """ Create the truncated Poisson distribution with a finite set of ks and a normalized probabilities.
        """
        # Create original Poisson distribution
        distribution = poisson(self.mean)

        # Find the maximal k to be considered
        max_k = 0
        while distribution.pmf(max_k) > self.truncated_threshold:
            max_k += 1

        # Create the truncated value list
        value_list = list(np.linspace(start=0, stop=max_k, num=max_k+1, dtype=int))

        # Create the probability
        prob_list = [distribution.pmf(k) for k in value_list]

        # Normalize the probability for the truncated values
        prob_list = (prob_list / np.sum(prob_list)).tolist()

        return value_list, prob_list

    def sample(self):
        """ Sample a k using the new truncated values and probabilities.
        """
        return np.random.choice(a=self.truncated_values, p=self.truncated_prob)

    def __iter__(self):
        """ Iterate all ks and its corresponding probabilities
        """
        return zip(self.truncated_values, self.truncated_prob)
    
def compute_expected_return(state, action, env, gamma, state_value):
    """ Function is used to compute the expected return given s and a. 
        It returns the value = sum_s' p(s'|s, a)[r(s, a) + gamma * V(s')]
    Args:
        state (list): state (i.e, s)
        action (int): action (i.e, a)
        env: the jack's car rental environment
        gamma (float): discount factor
        state_value (numpy.array): current state value
    """
    # compute the number of car to move
    car_num_to_move = env.move_car(state, action)

    # compute the state after moving
    state_after_move = [state[0] - car_num_to_move, state[1] + car_num_to_move]

    # compute the expectation using the model of the env
    # new_v(s) <--- \sum_{s'} p(s'|s, a)[r(s, a) + \gamma * old_v(s')]
    new_v = 0
    # The space of |s'| = 21 x 21. Because each lot will have the number of cars within [0, 20].
    for n_1 in range(21):
        for n_2 in range(21):
            # compute the transition probability p(s' | s, a) for one possible s'
            # p(s'|s, a) = p([n1, n2] | s, a) = p([n1, n2] | s_after_move) * p(s_after_move|s, a)
            #            = p([n1, n2] | s_after_move) since p(s_after_move|s,a) = 1 given a is deterministic.
            #            = p(n1 | s_after_move) * p(n2 | s_after_move) since lot1 and lot2 evolves independently.
            #            = p(n1 | s_after_move_lot1) * p(n2 | s_after_move_lot2)
            # prob = p(s'|s, a)
            prob = env.p_lot1[state_after_move[0]][n_1] * env.p_lot2[state_after_move[1]][n_2]

            # compute the reward = cost + expected rewards for lot1 and lot2 given the state after moving the car.
            # reward = r(s, a)
            if env.modified_reward:
                reward = env.compute_reward_modified(moved_cars=car_num_to_move,
                                                     car_num_lot1=state_after_move[0],
                                                     car_num_lot2=state_after_move[1])
            else:   
                reward = env.compute_reward(moved_cars=car_num_to_move,
                                        car_num_lot1=state_after_move[0],
                                        car_num_lot2=state_after_move[1])

            """ incrementally compute the new_v for the state-action pair (i.e., the expected return of (s, a))"""
            new_v += prob * (reward + (gamma * state_value[n_1][n_2]))

    return new_v

def policy_evaluation(state_value, policy, env, threshold, gamma):
    # iteration counter
    iter_counter = 0

    # check termination
    while True:
        # iteration counter
        iter_counter += 1

        # assume the current iteration should be terminated
        is_terminal = True

        # create a numpy array to store the new state value
        new_state_value = np.zeros_like(state_value)

        # loop all valid states
        for i in range(21):
            for j in range(21):
                """ Complete the update part of the policy evaluation"""
                """ Please use the compute_expected_return function above to update the new state value"""

                # store the old V value
                old_v = state_value[i, j]

                # get the action from the policy
                action = policy[i, j]

                # compute the expected return
                new_v = compute_expected_return([i, j], action, env, gamma, state_value)

                # Update terminal condition if value has not converged yet
                if abs(new_v - old_v) >= threshold:
                    is_terminal = False
                
                # store the updated state value
                new_state_value[i, j] = new_v

        # update the state value table
        state_value = new_state_value.copy()

        # check the termination
        if is_terminal:
            break

    return state_value

def policy_improvement(state_value, policy, env, gamma):
    # assume the policy is already stable
    is_stable = True
    # loop the state space
    for i in range(21):
        for j in range(21):
            """ CODE HERE """
            """ Complete the update part of the policy improvement"""
            """ Still please use the compute_expected_return function above to compute the new state value"""
            # state

            # obtain the old action
            action = policy[i, j]

            # compute a new greedy action
            new_a = 0
            max_v = float('-inf')
            for a in range(-5, 6):
                new_v = compute_expected_return([i, j], a, env, gamma, state_value)
                if new_v > max_v:
                    max_v = new_v
                    new_a = a

            # check if the policy is stable at state [i, j]
            if new_a != action:
                is_stable = False
                
            # update the policy
            policy[i, j] = new_a

    return policy, is_stable

def run_policy_iteration(env, threshold, gamma):
    # initialize the policy
    policy = np.zeros((21, 21), dtype=int)

    # initialize the state value
    state_value = np.zeros((21, 21))

    # run policy iteration
    policy_iter_counter = 0
    
    # save the policies in the iteration
    results_list = []

    # start policy iteration
    while True:
        # print info
        print(f"======================================")
        print(f"==   Policy iteration = {policy_iter_counter}")
        print(f"======================================")
        
        # policy evaluation
        print(f"Iter {policy_iter_counter}: Policy evaluation starts.")
        state_value = policy_evaluation(state_value, policy, env, threshold, gamma)
        print(f"Iter {policy_iter_counter}: Policy evaluation ends.")

        # policy improvement
        print(f"Iter {policy_iter_counter}: Policy improvement starts.")
        policy, is_stable = policy_improvement(state_value, policy, env, gamma)
        print(f"Iter {policy_iter_counter}: Policy improvement ends.")
        
        # save to the list
        results_list.append({"state_value": state_value.copy(),
                             "policy": policy.copy(),
                             "title": f"Iteration = {policy_iter_counter}"})

        if is_stable:
            break
        else:
            policy_iter_counter += 1
                
    print("======================")
    print("Policy iteration ends.")
            
    return results_list
    
def plot_policy(policy, title, file_suffix=""):
    policy = np.flip(policy, axis=0)

    car_num_lot1 = list(np.linspace(start=20, stop=0, num=21, dtype=int))
    car_num_lot2 = list(np.linspace(start=0, stop=20, num=21, dtype=int))

    fig, ax = plt.subplots()
    im = ax.imshow(policy, cmap="RdGy")

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(car_num_lot2)), labels=car_num_lot2)
    ax.set_yticks(np.arange(len(car_num_lot1)), labels=car_num_lot1)

    ax.set_xlabel("Number of Cars at Lot 2")
    ax.set_ylabel("Number of Cars at Lot 1")

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(car_num_lot1)):
        for j in range(len(car_num_lot2)):
            text = ax.text(j, i, policy[i, j],
                           ha="center", va="center", color="w")

    ax.set_title(title)
    fig.tight_layout()
    plt.savefig(f"plots/{title+file_suffix}.png")
    plt.show()

#Plot the value function
def plot_optimal_values(state_value, title, fname="state_values.png"):
    x = np.linspace(0, 20, 21)
    y = np.linspace(0, 20, 21)

    X, Y = np.meshgrid(x, y)
    Z = state_value
    
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.set_title(title)

    ax.set_xlabel('Number of Cars at Lot 1')
    ax.set_ylabel('Number of Cars at Lot 2')
    ax.set_zlabel('State Value')

    plt.savefig(f"plots/{fname}")
    plt.show()

'''
Policy Iteration with Default reward function
'''
# run the policy iteration
env = JackCarRental()
env.reset()

# set the threshold for policy evaluation
threshold = 1e-3

# set the gamma
gamma = 0.9

# run the policy iteration
results_list = run_policy_iteration(env, threshold, gamma) # Default reward function

# plot the final policy
for res in results_list:
    policy = res['policy']
    title = f"Policy Iteration Result: {res['title']}"
    plot_policy(policy, title)
    
state_value = results_list[0]['state_value']
title = f"State Value Function ({results_list[0]['title']})"
plot_optimal_values(state_value, title)

'''
Policy Iteration with Modified reward function
'''
# run the policy iteration
env = JackCarRental(modified_reward=True)
env.reset()

# set the threshold for policy evaluation
threshold = 1e-3

# set the gamma
gamma = 0.9

# run the policy iteration
results_list = run_policy_iteration(env, threshold, gamma) # Modified reward function

# plot the final policy
for res in results_list:
    policy = res['policy']
    title = res['title']
    plot_policy(policy, title, file_suffix="_modified_reward")
    
state_value = results_list[0]['state_value']
title = f"State Value Function ({results_list[0]['title']}): Modified Reward"
plot_optimal_values(state_value, title, fname="state_values_modified_reward.png")