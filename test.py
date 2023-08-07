








# ... (Previous code remains the same)

def run_env(params, learning_rate, gamma, epsilon):
    env = gym.make("FrozenLake-v1", is_slippery=params.is_slippery)

    params = params._replace(action_size=env.action_space.n)
    params = params._replace(state_size=env.observation_space.n)
    env.action_space.seed(params.seed)

    learner = QLearningTable(
        learning_rate=learning_rate,
        gamma=gamma,
        state_size=params.state_size,
        action_size=params.action_size,
    )
    explorer = EpsilonGreedyPolicy(epsilon=epsilon)

    rewards = np.zeros((params.total_episodes, params.n_runs))
    steps = np.zeros((params.total_episodes, params.n_runs))
    episodes = np.arange(params.total_episodes)
    qtables = np.zeros((params.n_runs, params.state_size, params.action_size))
    all_states = []
    all_actions = []

    for run in range(params.n_runs):  
        learner.reset_qtable()  

        for episode in tqdm(episodes, desc=f"Run {run+1}/{params.n_runs} - Episodes", leave=False):
            state = env.reset()  
            step = 0
            done = False
            total_rewards = 0

            while not done:
                action = explorer.choose_action(action_space=env.action_space, state=state, q_table=learner.q_table)

                all_states.append(state)
                all_actions.append(action)

                new_state, reward, done, _ = env.step(action)

                learner.update(state, action, reward, new_state)

                total_rewards += reward
                step += 1
                state = new_state

            rewards[episode, run] = total_rewards
            steps[episode, run] = step

        qtables[run, :, :] = learner.q_table

    env.close()
    return rewards, steps, episodes, qtables, all_states, all_actions

# Function to plot average rewards and steps for different hyperparameter settings
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
            rewards, steps, _, _, _, _ = run_env(params, learning_rate=lr, gamma=gamma, epsilon=epsilon)
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
