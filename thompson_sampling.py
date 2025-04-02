import numpy as np

def thompson_sampling(num_arms, num_rounds, reward_function):
    """
    Implements the Thompson Sampling algorithm for the multi-armed bandit problem.

    Parameters:
        num_arms (int): Number of arms (actions) available.
        num_rounds (int): Number of rounds to play.
        reward_function (function): A function that takes an arm index and returns a reward (0 or 1).

    Returns:
        list: A list of selected arms for each round.
        list: A list of cumulative rewards over rounds.
    """
    # Initialize variables
    successes = np.zeros(num_arms)  # Number of successes for each arm
    failures = np.zeros(num_arms)  # Number of failures for each arm
    selected_arms = []
    cumulative_rewards = []

    total_reward = 0

    for _ in range(num_rounds):
        # Sample from Beta distribution for each arm
        beta_samples = [np.random.beta(successes[i] + 1, failures[i] + 1) for i in range(num_arms)]
        
        # Select the arm with the highest sampled value
        chosen_arm = np.argmax(beta_samples)
        selected_arms.append(chosen_arm)
        
        # Get reward for the chosen arm
        reward = reward_function(chosen_arm)
        total_reward += reward
        cumulative_rewards.append(total_reward)
        
        # Update successes and failures
        if reward == 1:
            successes[chosen_arm] += 1
        else:
            failures[chosen_arm] += 1

    return selected_arms, cumulative_rewards

