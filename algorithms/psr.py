import numpy as np
from scipy.stats import norm, skew, kurtosis
from algorithms.mab_base import MabBase

class PSR(MabBase):
    """
    Probalilistic Sharpe Ratio algorithm for multi-armed bandit problems.
    """
    def __init__(self, R, window_size=120):
        super().__init__(R, window_size)

        self.reward = np.zeros(self.n_samples - self.window_size)
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
            #psr_second_set = []
            for a in range(len(sharpe_ratio)):
                sr = sharpe_ratio[a]
                n = (self.window_size + self.played_times[a]) / self.window_size
                skewness = skew(portfolio_reward[a, :])
                kurto = kurtosis(portfolio_reward[a, :])
                nomin = (sr-np.mean(sharpe_ratio))*np.sqrt(n-1)
                denom = np.sqrt(np.abs(1-(skewness*sr) + ((kurto-1)/4)*(sr**2)))
                #psr = norm.cdf(nomin/denom)
                self.psr_set.append(nomin/denom)
                
            self.psr_set = np.array([norm.cdf(a) for a in self.psr_set])
            self.psr[t] = self.psr_set
            
            # Compute the upper bound of expected reward
            #sharpe_ratio_upper_bound = sharpe_ratio + np.sqrt((2*np.log(t))/(window_size+self.played_times))
            #sharpe_ratio_upper_bound = (sharpe_ratio + \
            #    np.sqrt((2*np.log(t))/(window_size+self.played_times)))*np.array(self.psr_set)
            
            #action1 = np.argmax(sharpe_ratio_upper_bound[:l])
            #action2 = np.argmax(sharpe_ratio_upper_bound[l:])+l
            
            # Select the optimal arm
            passive = np.argmax(self.psr_set[:cutoff])
            active = np.argmax(self.psr_set[cutoff:])+cutoff

            self.played_times[passive] += 1
            self.played_times[active] += 1

            # Optimize the weights
            self.update_weight_reward(H=eigenvectors, A=eigenvalues, passive=passive, active=active)
            