import numpy as np
import matplotlib.pyplot as plt

class RandomWalk:
    def __init__(self):
        """Initialize the 5-state Random Walk MRP."""
        self.states = ['A', 'B', 'C', 'D', 'E']
        self.index = 2 # Start at state C
        self.terminal_states = ['left_terminal', 'right_terminal']

    def reset(self):
        """Reset the environment to the start state (C)."""
        self.index = 2
        return self.states[self.index]
    
    def step(self, action):
        """Take a step in the environment.
        
        Args:
            action (int): -1 for left, +1 for right.
        
        Returns:
            next_state (str): The next state.
            reward (float): Reward received.
            done (bool): Whether the episode has ended.
        """
        self.index += action
        
        if self.index < 0:
            return 'left_terminal', 0.0, True
        
        elif self.index >= len(self.states):
            return 'right_terminal', 1.0, True
        
        else:
            return self.states[self.index], 0.0, False

class TDLearner:
    def __init__(self, alpha=0.1):
        """Initialize the TD(0) learning algorithm."""
        self.alpha = alpha
        self.value_function = {}  # Initialize value function
    
    def update(self, state, reward, next_state):
        """Perform a TD(0) update step.
        
        Args:
            state (str): Current state.
            reward (float): Reward received.
            next_state (str): Next state after transition.
        """
        if state not in self.value_function:
            self.value_function[state] = 0.0
        
        # TD(0) update rule
        self.value_function[state] += self.alpha * (reward + self.value_function.get(next_state, 0.0) - self.value_function[state])


def run_experiment(num_episodes=1000):
    """Run the Random Walk experiment with TD(0)."""
    env = RandomWalk()
    agent = TDLearner(alpha=0.1)
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = np.random.choice([-1, 1])  # Random left or right move
            next_state, reward, done = env.step(action)
            agent.update(state, reward, next_state)
            state = next_state
    
    return sorted(agent.value_function.values())  # Return learned values

def plot_results(estimated_values):
    """Plot estimated values against true values."""
    plt.figure(figsize=(10, 5))
    states = ['A', 'B', 'C', 'D', 'E']
    true_values = [1/6, 2/6, 3/6, 4/6, 5/6]  # True values for states A to E
    plt.plot(states, true_values, label='True Values', marker='o')
    plt.plot(states, estimated_values, label='Estimated Values', marker='o')
    plt.xlabel('States')
    plt.ylabel('Value')
    plt.title('TD(0) Estimated Values vs True Values')
    plt.legend()
    plt.grid()
    plt.savefig("plots/RandomWalk_TD0_Estimated_Values.png")  # Save the plot as a PNG file
    plt.show()  



if __name__ == "__main__":
    estimated_values = run_experiment()
    print("Estimated Values:", estimated_values)
    plot_results(estimated_values)