from turtle import distance

from four_rooms_env import FourRooms
from agents import FuncApproxSARSA, QLearning
import numpy as np
import matplotlib.pyplot as plt

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
    ax.set_title("Four Rooms Environment with Function Approximation")
    ax.legend(handles=h_list)
    plt.show()

def run_experiment(env, feature_fn, n_features, run_num):
    trial_history = []
    for i in range(run_num):
        agent = FuncApproxSARSA(env, feature_fn, n_features,
                                alpha=0.01, epsilon=0.1, gamma=0.9, timeout=50_000)
        episode_history = agent.run()
        trial_history.append(episode_history)
    return np.array(trial_history)

def run_gt_experiment(env, run_num):
    trial_history = []
    for i in range(run_num):
        agent = QLearning(env, alpha=0.01, epsilon=0.1, gamma=0.9, timeout=50_000)
        episode_history = agent.run()
        trial_history.append(episode_history)
    return np.array(trial_history)

'''
ONE HOT ENCODING FEATURE FUNCTION
'''
def one_hot_encoding_feature_fn(state, action):
    feature_vector = np.zeros(env.state_space_shape[0] * env.state_space_shape[1] * len(env.action_space))
    state_index = state[0] * env.state_space_shape[1] + state[1]
    action_index = list(env.action_space.keys()).index(action)
    feature_vector[state_index * len(env.action_space) + action_index] = 1
    return feature_vector

env = FourRooms()
n_features = env.state_space_shape[0] * env.state_space_shape[1] * len(env.action_space)
one_hot_trials = run_experiment(env, one_hot_encoding_feature_fn, n_features, run_num=10)

'''
TILE CODING FEATURE FUNCTION
'''
def tile_coding_feature_fn(state, action, tile_width):
    num_tiles_x = (env.state_space_shape[0] // tile_width) + 1
    num_tiles_y = (env.state_space_shape[1] // tile_width) + 1
    num_actions = len(env.action_space)
    
    feature_vector = np.zeros(num_tiles_x * num_tiles_y * num_actions)
    
    tile_x = state[0] // tile_width
    tile_y = state[1] // tile_width
    action_index = list(env.action_space.keys()).index(action)
    
    tile_index = (tile_x * num_tiles_y + tile_y) * num_actions + action_index
    feature_vector[tile_index] = 1
    
    return feature_vector

tile_width = 2
num_tiles_x = (env.state_space_shape[0] // tile_width) + 1
num_tiles_y = (env.state_space_shape[1] // tile_width) + 1
num_actions = len(env.action_space)

n_features = num_tiles_x * num_tiles_y * num_actions
tile_feature_fn = lambda state, action: tile_coding_feature_fn(state, action, tile_width=tile_width)
tile_trials = run_experiment(env, tile_feature_fn, n_features, run_num=10)


''' FOUR TILES '''
tile_width = 6

num_tiles_x = (env.state_space_shape[0] // tile_width) + 1
num_tiles_y = (env.state_space_shape[1] // tile_width) + 1
num_actions = len(env.action_space)

n_features = num_tiles_x * num_tiles_y * num_actions
tile_feature_fn = lambda state, action: tile_coding_feature_fn(state, action, tile_width=tile_width)
four_tile_trials = run_experiment(env, tile_feature_fn, n_features, run_num=10)

'''
GROUND TRUTH: Tabular Q-Learning
'''
gt_trials = run_gt_experiment(env, run_num=10)

plot_curves([one_hot_trials, tile_trials, four_tile_trials, gt_trials], [f"SARSA with One-Hot Encoding ({11 * 11} tiles)", f"SARSA with Tile Coding ({6*6} tiles)", f"SARSA with Four Tiles ({2*2} tiles)", "Ground Truth: Tabular Q-Learning"], ["blue", "red", "green", "orange"], "Total Episodes")

# ========================================== #
''' IMPLEMENTATION OF SEMI_GRADIENT SARSA AND Q-LEARNING '''
# ========================================== #


# ========================================== #
''' FEATURE FUNCTION EXPERIMENTATION '''
# ========================================== #

env = FourRooms()

# (x,y) state representation
def xy_feature_fn(state, action):
    n_actions = len(env.action_space)
    action_index = list(env.action_space.keys()).index(action)
    feature_vector = np.zeros((n_actions, 3))
    feature_vector[action_index, 0] = state[0]
    feature_vector[action_index, 1] = state[1]
    feature_vector[action_index, 2] = 1  # Bias term
    return feature_vector.flatten()

n_features = 3 * len(env.action_space)  
xy_trials = run_experiment(env, xy_feature_fn, n_features, run_num=10)

def norm_xy_feature_fn(state, action):
    n_actions = len(env.action_space)
    action_index = list(env.action_space.keys()).index(action)
    feature_vector = np.zeros((n_actions, 3))
    feature_vector[action_index, 0] = state[0] / env.state_space_shape[0]
    feature_vector[action_index, 1] = state[1] / env.state_space_shape[1]
    feature_vector[action_index, 2] = 1  # Bias term
    return feature_vector.flatten()

n_features = 3 * len(env.action_space)
norm_xy_trials = run_experiment(env, norm_xy_feature_fn, n_features, run_num=10)

def distance_to_goal_feature_fn(state, action):
    n_actions = len(env.action_space)
    action_index = list(env.action_space.keys()).index(action)
    feature_vector = np.zeros((n_actions, 4))
    
    distance = np.sqrt((state[0] - env.goal_state[0]) ** 2 + (state[1] - env.goal_state[1]) ** 2)
    max_distance = np.sqrt(env.state_space_shape[0]**2 + env.state_space_shape[1]**2)
    norm_distance = distance / max_distance

    feature_vector[action_index, 0] = state[0] / env.state_space_shape[0]
    feature_vector[action_index, 1] = state[1] / env.state_space_shape[1]
    feature_vector[action_index, 2] = norm_distance  # Normalized distance to goal
    feature_vector[action_index, 3] = 1  # Bias term

    return feature_vector.flatten()

n_features = 4 * len(env.action_space)
distance_trials = run_experiment(env, distance_to_goal_feature_fn, n_features, run_num=10)

plot_curves([xy_trials, norm_xy_trials, distance_trials], [f"SARSA with (x,y) state representation", f"SARSA with Normalized (x,y) state representation", f"SARSA with Distance to Goal feature"], ["blue", "red", "green"], "Episode Returns")