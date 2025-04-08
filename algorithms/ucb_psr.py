import numpy as np
from scipy.stats import norm, skew, kurtosis
from algorithms.mab_base import MabBase


class UCBPSR(MabBase):
    """
    Upper Confidence Bound with Probabilistic Sharpe Ration algorithm for multi-armed bandit problems.
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

            self.psr_set = []
            for a in range(len(sharpe_ratio)):
                sr = sharpe_ratio[a]
                n = (self.window_size + self.played_times[a]) / self.window_size
                skewness = skew(portfolio_reward[a, :])
                kurto = kurtosis(portfolio_reward[a, :])
                nomin = (sr-np.mean(sharpe_ratio))*np.sqrt(n)
                denom = np.sqrt(np.abs((1 + 0.5 * sr**2 - skewness*sr + ((kurto-3)/4)*sr**2))/(n-1))
                #psr = norm.cdf(nomin/denom)
                self.psr_set.append(nomin/denom)

            #self.psr_set = scl.fit_transform(np.array(self.psr_set).reshape(-1,1)).reshape(-1)
            self.psr_set = np.array([norm.cdf(a) for a in self.psr_set])
            self.psr[t] = self.psr_set
            
            # Compute the upper bound of expected reward
            sharpe_ratio_upper_bound = (sharpe_ratio + np.sqrt((2*np.log(t)) / (self.window_size + self.played_times))) * np.array(self.psr_set)
            
            passive = np.argmax(sharpe_ratio_upper_bound[:cutoff])
            active = np.argmax(sharpe_ratio_upper_bound[cutoff:]) + cutoff
            
            # Select the optimal arm
            #action1 = np.argmax(self.psr_set[:l])
            #action2 = np.argmax(self.psr_set[l:])+l

            self.played_times[passive] += 1
            self.played_times[active] += 1

            # Optimize the weights
            self.update_weight_reward(H=eigenvectors, A=eigenvalues, passive=passive, active=active)

