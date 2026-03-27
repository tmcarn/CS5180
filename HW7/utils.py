from matplotlib import pyplot as plt
import numpy as np
from DQN import DQNAgent
import os


def pad_to_same_length(list_of_arrays):
    """Pad ragged lists to the same length with NaN."""
    max_len = max(len(a) for a in list_of_arrays)
    padded = np.full((len(list_of_arrays), max_len), np.nan)
    for i, a in enumerate(list_of_arrays):
        padded[i, :len(a)] = a
    return padded


def plot_curves(arr_list, legend_list, color_list, ylabel, fig_title, path):
    """
    Args:
        arr_list (list): list of results arrays to plot
        legend_list (list): list of legends corresponding to each result array
        color_list (list): list of color corresponding to each result array
        ylabel (string): label of the Y axis

        Note that, make sure the elements in the arr_list, legend_list and
        color_list are associated with each other correctly.
        Do not forget to change the ylabel for different plots.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time Steps")

    h_list = []
    for arr, legend, color in zip(arr_list, legend_list, color_list):
        arr_mean = np.nanmean(arr, axis=0)
        arr_err = np.nanstd(arr, axis=0) / np.sqrt(np.sum(~np.isnan(arr), axis=0))
        arr_err *= 1.96

        h, = ax.plot(range(arr.shape[1]), arr_mean, color=color, label=legend)
        ax.fill_between(
            range(arr.shape[1]),
            arr_mean - arr_err,
            arr_mean + arr_err,
            alpha=0.3,
            color=color,
        )
        h_list.append(h)

    ax.set_title(f"{fig_title}")
    ax.legend(handles=h_list)
    plt.savefig(path)


def run_training(env, train_parameters, num_trials=10, name=""):
    all_returns = []
    all_loss = []

    for trial in range(num_trials):
        agent = DQNAgent(env=env, params=train_parameters)
        train_returns, train_loss = agent.train()
        all_returns.append(train_returns)
        all_loss.append(train_loss)

    all_returns = pad_to_same_length(all_returns)
    all_loss = pad_to_same_length(all_loss)

    root = "plots"
    returns_path = os.path.join(root, f"{name}_returns.png")
    loss_path = os.path.join(root, f"{name}_loss.png")

    plot_curves([all_returns], ["DQN"], ["blue"], "Episode Returns", "DQN on Four Rooms", returns_path)
    plot_curves([all_loss], ["DQN"], ["blue"], "Loss", "DQN on Four Rooms", loss_path)
