import numpy as np
from algorithms.mab_base import MabBase

class UCB(MabBase):
    """
    Upper Confidence Bound algorithm for multi-armed bandit problems.
    """
    def __init__(self, R, window_size=120, cutoff=5):
        super().__init__(R, window_size=window_size, cutoff=cutoff)

    def run(self):
        for t in range(self.window_size, self.n_samples):
            # Get current slice from previous data (window size)
            slice = self.R[:, t - self.window_size:t]

            # Compute the orthogonal portfolio
            eigenvectors, eigenvalues, portfolio_reward, sharpe_ratio = self.orthogonal_portfolio(slice)

            # Compute the Upper Bound of expected reward
            sr_upper_bound = sharpe_ratio + np.sqrt((2 * np.log(t)) / (self.window_size * self.played_times))

            # Compute the optimal portfolio
            passive = np.argmax(sr_upper_bound[:self.cutoff])
            active = np.argmax(sr_upper_bound[self.cutoff:]) + self.cutoff

            # Update played_times, reward, and weight
            self.update(t=t, H=eigenvectors, A=eigenvalues, passive=passive, active=active)

    # def cutoff(self, eigenvalues):
    #     print(f"eigenvalues size: {eigenvalues.shape}")
    #     print(f"eigenvalues: {eigenvalues}")
    #     # eigenvalues is sorted 2D array 48x48
    #     # find the significant drop in eigenvalues
    #     # differences is 1D array
    #     differences = np.diff(eigenvalues, axis=0)
    #     print(f"differences: {differences}")

    #     # differences is 2D array
    #     if len(differences.shape) > 1:
    #         differences = differences.flatten()
    #     print(f"differences size: {differences.shape}")
    #     print(f"differences: {differences}")

    #     # Find the index of the maximum drop
    #     drop_index = np.argmin(differences)
    #     cutoff = drop_index + 1 # +1 to account for the index shift due to np.diff
    #     print(f"cutoff: {cutoff}")
    #     return cutoff


    
