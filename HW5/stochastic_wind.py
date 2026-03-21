from windy_gridworld import WindyGridWorld, QLearning, ExpectedSARSA, plot_curves
import numpy as np

# Knight's move actions and no move actions enabled
enable_king_move_actions = True
enable_no_move_actions = True

stochastic_wind = False

run_num = 10
timeout = 8000

# create the environment
env = WindyGridWorld(enable_king_move=enable_king_move_actions, enable_no_move=enable_no_move_actions, random_wind=stochastic_wind)

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

# create the Q learning
q_learning_results_list = []
for _ in range(run_num):
    # run for each trial
    controller_q_learning = QLearning(env, alpha, epsilon, gamma, timeout)
    episodes = controller_q_learning.run()
    # append the results
    q_learning_results_list.append(episodes[0:8000])


# Knight's move actions and no move actions enabled
enable_king_move_actions = True
enable_no_move_actions = True
stochastic_wind = True

run_num = 10
timeout = 8000

# create the environment
env = WindyGridWorld(enable_king_move=enable_king_move_actions, enable_no_move=enable_no_move_actions, random_wind=stochastic_wind)

# parameters
epsilon = 0.1
alpha = 0.5
gamma = 1.0

# create the expected SARSA
rand_wind_expected_sarsa_results_list = []
for _ in range(run_num):
    # run for each trial
    controller_expected_sarsa = ExpectedSARSA(env, alpha, epsilon, gamma, timeout)
    episodes = controller_expected_sarsa.run()
    # append the results
    rand_wind_expected_sarsa_results_list.append(episodes[0:8000])

# create the Q learning
rand_wind_q_learning_results_list = []
for _ in range(run_num):
    # run for each trial
    controller_q_learning = QLearning(env, alpha, epsilon, gamma, timeout)
    episodes = controller_q_learning.run()
    # append the results
    rand_wind_q_learning_results_list.append(episodes[0:8000])




expected_sarsa_array = np.array(expected_sarsa_results_list)
q_learning_array = np.array(q_learning_results_list)
rand_wind_expected_sarsa_array = np.array(rand_wind_expected_sarsa_results_list)
rand_wind_q_learning_array = np.array(rand_wind_q_learning_results_list)

plot_curves(
    [expected_sarsa_array, q_learning_array,rand_wind_expected_sarsa_array, rand_wind_q_learning_array],
    ['Expected SARSA', 'Q-Learning', 'Random Wind Expected SARSA', 'Random Wind Q-Learning'],
    ['r', 'g', 'b', 'm'],
    "Episodes"
)
