import numpy as np
from algorithms.mab_base import MabBase

class UCB(MabBase):
    """
    Upper Confidence Bound algorithm for multi-armed bandit problems.
    This class implements the UCB algorithm for selecting the best arm based on
    the average reward and the number of times each arm has been played.
    """
    def __init__(self, R, window_size=120):
        super().__init__(R, window_size)
        # self.weights = np.zeros(self.n_arms)
        self.reward = np.ones(self.n_samples - self.window_size)
        self.played_times = np.zeros(self.n_arms)

    def run(self):
        for t in range(self.window_size, self.n_samples):
            # Get current slice from previous data (window size)
            slice = self.R[:, t - self.window_size:t]

            # Compute the orthogonal portfolio
            eigenvectors, eigenvalues, portfolio_reward, sharpe_ratio = self.orthogonal_portfolio(slice)

            # get cutoff number
            cutoff = self.cutoff_function()

            # Compute the Upper Bound of expected reward
            sr_upper_bound = sharpe_ratio + np.sqrt((2 * np.log(self.played_times)) / (self.window_size * self.played_times))

            # Compute the optimal portfolio
            passive = np.argmax(sr_upper_bound[:cutoff])
            active = np.argmax(sr_upper_bound[cutoff:]) + cutoff

            self.played_times[passive] += 1
            self.played_times[active] += 1

            # Optimize the weights
            self.update_weight_reward(H=eigenvectors, A=eigenvalues, passive=passive, active=active)



    
