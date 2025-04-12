import numpy as np
from algorithms.mab_base import MabBase

class UCB(MabBase):
    """
    Upper Confidence Bound algorithm for multi-armed bandit problems.
    """
    def __init__(self, R, window_size=120):
        super().__init__(R, window_size)

    def run(self):
        for t in range(self.window_size, self.n_samples):
            # Get current slice from previous data (window size)
            slice = self.R[:, t - self.window_size:t]

            # Compute the orthogonal portfolio
            eigenvectors, eigenvalues, portfolio_reward, sharpe_ratio = self.orthogonal_portfolio(slice)

            # get cutoff
            cutoff = self.cutoff()

            # Compute the Upper Bound of expected reward
            sr_upper_bound = sharpe_ratio + np.sqrt((2 * np.log(t)) / (self.window_size * self.played_times))

            # Compute the optimal portfolio
            passive = np.argmax(sr_upper_bound[:cutoff])
            active = np.argmax(sr_upper_bound[cutoff:]) + cutoff

            # Update played_times, reward, and weight
            self.update(t=t, H=eigenvectors, A=eigenvalues, passive=passive, active=active)



    
