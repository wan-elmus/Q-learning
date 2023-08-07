
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map


sns.set_theme()

# Parameters

# Define the parameters using NamedTuple
class FrozenLakeParams(NamedTuple):
    total_episodes: int
    learning_rate: float
    gamma: float
    epsilon: float
    map_size: int
    seed: int
    is_slippery: bool
    n_runs: int
    action_size: int
    state_size: int
    proba_frozen: float
    savefig_folder: Path
    
# Create the FrozenLakeParams instance 
params = FrozenLakeParams(
    total_episodes=2000,
    learning_rate=0.8,
    gamma=0.95,
    epsilon=0.1,
    map_size=5,
    seed=123,
    is_slippery=False,
    n_runs=20,
    action_size=None,
    state_size=None,
    proba_frozen=0.9,
    savefig_folder=Path("./img/"),
) 

# Set the seed
rng = np.random.default_rng(params.seed)

# Create the figure folder if it doesn't exist
params.savefig_folder.mkdir(parents=True, exist_ok=True)

# Frozen Lake env

# Create the Frozen Lake environment
env = gym.make(
    "FrozenLake-v1",
    desc=None,  # The description of the map will be generated automatically
    map_name=None,
    is_slippery=params.is_slippery,
    render_mode="rgb_array",  
)

# Update the params object with the correct state_size and action_size
params = params._replace(action_size=env.action_space.n)
params = params._replace(state_size=env.observation_space.n)

env.close()  

# Creating the Q-Table

class QLearningTable:
    def __init__(self, gamma, learning_rate, state_size, action_size):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros((state_size, action_size))

    def reset_qtable(self):
        self.q_table = np.zeros((self.state_size, self.action_size))

    def update(self, state, action, reward, new_state):
        predict = self.q_table[state, action]
        target = reward + self.gamma * np.max(self.q_table[new_state, :])
        self.q_table[state, action] += self.learning_rate * (target - predict)

class EpsilonGreedyPolicy:
    def __init__(self, epsilon, rng):
        self.epsilon = epsilon
        self.rng = rng

    def choose_action(self, action_space, state, q_table):
        """Choose an action `a` in the current world state (s)."""
        # First, we randomize a number
        explor_exploit_tradeoff = self.rng.uniform(0, 1)

        # Exploration
        if explor_exploit_tradeoff < self.epsilon:
            action = action_space.sample()

        # Exploitation (taking the biggest Q-value for this state)
        else:
            # Break ties randomly
            # If all actions are the same for this state, we choose a random one
            # (otherwise `np.argmax()` would always take the first one)
            if np.all(q_table[state, :] == q_table[state, 0]):
                action = action_space.sample()
            else:
                action = np.argmax(q_table[state, :])
        return action


rng = np.random.default_rng()  # Create an instance of the random number generator
explorer = EpsilonGreedyPolicy(epsilon=params.epsilon, rng=rng)  # Pass the rng instance to the EpsilonGreedyPolicy class 

def run_env(params):
    env = gym.make("FrozenLake-v1", is_slippery=params.is_slippery)
    learner = QLearningTable(
        gamma=params.gamma,
        learning_rate=params.learning_rate,
        state_size=params.state_size,
        action_size=params.action_size,
    )
    explorer = EpsilonGreedyPolicy(params.epsilon, rng=np.random.default_rng(params.seed))

    rewards = np.zeros((params.total_episodes, params.n_runs))
    steps = np.zeros((params.total_episodes, params.n_runs))
    episodes = np.arange(params.total_episodes)
    qtables = np.zeros((params.n_runs, params.state_size, params.action_size))
    all_states = []
    all_actions = []

    for run in range(params.n_runs):  # Run several times to account for stochasticity
        learner.reset_qtable()  # Reset the Q-table between runs

        for episode in tqdm(episodes, desc=f"Run {run+1}/{params.n_runs} - Episodes", leave=False):
            state = env.reset()  # state is a tuple, e.g., (0,)
            state = state[0]  # Convert state to a scalar integer
            step = 0
            done = False
            total_rewards = 0
            
            
            

            while not done:
                action = explorer.choose_action(action_space=env.action_space, state=state, q_table=learner.q_table)

            # Log all states and actions
            all_states.append(state)  # Append the scalar state value
            all_actions.append(action)

            # Take the action (a) and observe the outcome state(s') and reward (r)
            new_state, reward, done, _, = env.step(action)  # Store only the required values

            learner.update(state, action, reward, new_state[0])  # Convert new_state to scalar integer

            total_rewards += reward
            step += 1

            # Our new state is state
            state = new_state[0]  # Convert new_state to scalar integer


            # Log all rewards and steps
            rewards[episode, run] = total_rewards
            steps[episode, run] = step

        qtables[run, :, :] = learner.q_table

    env.close()  
    return rewards, steps, episodes, qtables, all_states, all_actions

# plot average rewards and steps for different hyperparameter settings
def plot_hyperparameter_analysis(results, hyperparameter, hyperparameter_values, ylabel):
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, value in enumerate(hyperparameter_values):
        avg_results = np.mean(results[i], axis=1)
        sns.lineplot(data=pd.Series(avg_results), ax=ax, label=f"{hyperparameter} = {value}")
    ax.set_xlabel("Episodes")
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.show()

# Test different hyperparameter settings
learning_rates = [0.2, 0.5, 0.8]
gammas = [0.8, 0.9, 0.95]
epsilons = [0.1, 0.2, 0.3]

all_rewards = []
all_steps = []

for lr in learning_rates:
    for gamma in gammas:
        for epsilon in epsilons:
            # Update the params instance with the new hyperparameters
            params = params._replace(learning_rate=lr, gamma=gamma, epsilon=epsilon)

            rewards, steps, _, _, _, _ = run_env(params)
            all_rewards.append(rewards)
            all_steps.append(steps)


# Plot the impact of learning_rate on average rewards and steps
plot_hyperparameter_analysis(all_rewards, "Learning Rate", learning_rates, "Average Rewards")
plot_hyperparameter_analysis(all_steps, "Learning Rate", learning_rates, "Average Steps")

# Plot the impact of gamma on average rewards and steps
plot_hyperparameter_analysis(all_rewards, "Gamma", gammas, "Average Rewards")
plot_hyperparameter_analysis(all_steps, "Gamma", gammas, "Average Steps")

# Plot the impact of epsilon on average rewards and steps
plot_hyperparameter_analysis(all_rewards, "Epsilon", epsilons, "Average Rewards")
plot_hyperparameter_analysis(all_steps, "Epsilon", epsilons, "Average Steps")

# Visualization

def postprocess(episodes, params, rewards, steps, map_size):
    """Convert the results of the simulation in dataframes."""
    res = pd.DataFrame(
        data={
            "Episodes": np.tile(episodes, reps=params.n_runs),
            "Rewards": rewards.flatten(),
            "Steps": steps.flatten(),
        }
    )
    res["cum_rewards"] = rewards.cumsum(axis=0).flatten(order="F")
    res["map_size"] = np.repeat(f"{map_size}x{map_size}", res.shape[0])

    st = pd.DataFrame(data={"Episodes": episodes, "Steps": steps.mean(axis=1)})
    st["map_size"] = np.repeat(f"{map_size}x{map_size}", st.shape[0])
    return res, st

def qtable_directions_map(qtable, map_size):
    """Get the best learned action & map it to arrows."""
    qtable_val_max = qtable.max(axis=1).reshape(map_size, map_size)
    qtable_best_action = np.argmax(qtable, axis=1).reshape(map_size, map_size)
    directions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    qtable_directions = np.empty(qtable_best_action.flatten().shape, dtype=str)
    eps = np.finfo(float).eps  # Minimum float number on the machine
    for idx, val in enumerate(qtable_best_action.flatten()):
        if qtable_val_max.flatten()[idx] > eps:
            # Assign an arrow only if a minimal Q-value has been learned as best action
            # otherwise since 0 is a direction, it also gets mapped on the tiles where
            # it didn't actually learn anything
            qtable_directions[idx] = directions[val]
    qtable_directions = qtable_directions.reshape(map_size, map_size)
    return qtable_val_max, qtable_directions

def plot_q_values_map(qtable, env, map_size):
    """Plot the last frame of the simulation and the policy learned."""
    qtable_val_max, qtable_directions = qtable_directions_map(qtable, map_size)

    # Plot the last frame
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    ax[0].imshow(env.render())
    ax[0].axis("off")
    ax[0].set_title("Last frame")

    # Plot the policy
    sns.heatmap(
        qtable_val_max,
        annot=qtable_directions,
        fmt="",
        ax=ax[1],
        cmap=sns.color_palette("Blues", as_cmap=True),
        linewidths=0.7,
        linecolor="black",
        xticklabels=[],
        yticklabels=[],
        annot_kws={"fontsize": "xx-large"},
    ).set(title="Learned Q-values\nArrows represent best action")
    for _, spine in ax[1].spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("black")
    img_title = f"frozenlake_q_values_{map_size}x{map_size}.png"
    fig.savefig(params.savefig_folder / img_title, bbox_inches="tight")
    plt.show()
    
def plot_states_actions_distribution(states, actions, map_size):
    """Plot the distributions of states and actions."""
    labels = {"LEFT": 0, "DOWN": 1, "RIGHT": 2, "UP": 3}

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.histplot(data=states, ax=ax[0], kde=True)
    ax[0].set_title("States")
    sns.histplot(data=actions, ax=ax[1])
    ax[1].set_xticks(list(labels.values()), labels=labels.keys())
    ax[1].set_title("Actions")
    fig.tight_layout()
    img_title = f"frozenlake_states_actions_distrib_{map_size}x{map_size}.png"
    fig.savefig(params.savefig_folder / img_title, bbox_inches="tight")
    plt.show()
    
map_sizes = [5, 8, 11, 14]
res_all = pd.DataFrame()
st_all = pd.DataFrame()

for map_size in map_sizes:
    env = gym.make(
        "FrozenLake-v1",
        is_slippery=params.is_slippery,
        render_mode="rgb_array",
        desc=generate_random_map(
            size=map_size, p=params.proba_frozen, seed=params.seed
        ),
    )

    params = params._replace(action_size=env.action_space.n)
    params = params._replace(state_size=env.observation_space.n)
    env.action_space.seed(
        params.seed
    )  # Set the seed to get reproducible results when sampling the action space
    learner = QLearningTable(
        learning_rate=params.learning_rate,
        gamma=params.gamma,
        state_size=params.state_size,
        action_size=params.action_size,
    )
    explorer = EpsilonGreedyPolicy(
        epsilon=params.epsilon,
    )

    print(f"Map size: {map_size}x{map_size}")
    rewards, steps, episodes, qtables, all_states, all_actions = run_env()

    # Save the results in dataframes
    res, st = postprocess(episodes, params, rewards, steps, map_size)
    res_all = pd.concat([res_all, res])
    st_all = pd.concat([st_all, st])
    qtable = qtables.mean(axis=0)  # Average the Q-table between runs

    plot_states_actions_distribution(
        states=all_states, actions=all_actions, map_size=map_size
    )  # Sanity check
    plot_q_values_map(qtable, env, map_size)

    env.close()  
    
def plot_steps_and_rewards(rewards_df, steps_df):
    """Plot the steps and rewards from dataframes."""
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.lineplot(
        data=rewards_df, x="Episodes", y="cum_rewards", hue="map_size", ax=ax[0]
    )
    ax[0].set(ylabel="Cumulated rewards")

    sns.lineplot(data=steps_df, x="Episodes", y="Steps", hue="map_size", ax=ax[1])
    ax[1].set(ylabel="Averaged steps number")

    for axi in ax:
        axi.legend(title="map size")
    fig.tight_layout()
    img_title = "frozenlake_steps_and_rewards.png"
    fig.savefig(params.savefig_folder / img_title, bbox_inches="tight")
    plt.show()


plot_steps_and_rewards(res_all, st_all)      