import numpy as np
from scipy.stats import norm, skew, kurtosis
from algorithms.mab_base import MabBase

class PWUCB(MabBase):
    """
    Upper Confidence Bound with Probabilistic Sharpe Ration algorithm for multi-armed bandit problems.
    Attributes:
        psr (dict): Dictionary to store the Probabilistic Sharpe Ratio for each time step.
        psr_set (list): List to store the Probabilistic Sharpe Ratio for the current time step.
    """
    def __init__(self, R, window_size, cutoff):
        super().__init__(R, window_size=window_size, cutoff=cutoff)
        self.psr = {}

    def run(self):
        for t in range(self.window_size, self.n_samples):
            # Get current slice from previous data (window size)
            slice = self.R[:, t - self.window_size:t]

            # Compute the orthogonal portfolio
            eigenvectors, eigenvalues, portfolio_reward, sharpe_ratio = self.orthogonal_portfolio(slice)

            self.psr_set = []
            for a in range(len(sharpe_ratio)):
                sr = sharpe_ratio[a]
                n = (self.window_size + self.played_times[a]) / self.window_size
                skewness = skew(portfolio_reward[a, :])
                kurto = kurtosis(portfolio_reward[a, :])
                nomin = (sr - np.mean(sharpe_ratio)) * np.sqrt(n)
                denom = np.sqrt(np.abs((1 + 0.5 * sr**2 - skewness*sr + ((kurto-3)/4)*sr**2))/(n-1))
                #psr = norm.cdf(nomin/denom)
                self.psr_set.append(nomin/denom)

            #self.psr_set = scl.fit_transform(np.array(self.psr_set).reshape(-1,1)).reshape(-1)
            self.psr_set = np.array([norm.cdf(a) for a in self.psr_set])
            self.psr[t] = self.psr_set
            
            # Compute the upper bound of expected reward
            sr_upper_bound = (sharpe_ratio + np.sqrt((2*np.log(t)) / (self.window_size + self.played_times))) * np.array(self.psr_set)
            
            # Select the optimal arm
            passive = np.argmax(sr_upper_bound[:self.cutoff])
            active = np.argmax(sr_upper_bound[self.cutoff:]) + self.cutoff
            
            # Update played_times, reward, and weight
            self.update(t=t, H=eigenvectors, A=eigenvalues, passive=passive, active=active)

