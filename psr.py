import numpy as np
from scipy.stats import norm, skew, kurtosis
from mab import MAB

class PSR(MAB):
    """
    Probalilistic Sharpe Ratio algorithm for multi-armed bandit problems.
    """
    def __init__(self, R, window_size=120):
        super().__init__(R, window_size)

        self.reward = np.zeros(self.n_samples - self.window_size)
        self.played_times = np.zeros(self.n_arms)

    def algorithm(self, Hnorm, Anorm, sharpe_ratio, cutoff):
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
        action1 = np.argmax(self.psr_set[:l])
        action2 = np.argmax(self.psr_set[l:])+l

        self.played_times[action1] += 1
        self.played_times[action2] += 1

        # Optimal weight
        Adiag = ANew.diagonal()
        theta = Adiag[action1] / (Adiag[action1] + Adiag[action2])
        self.weight = (1-theta)*H[:,action1] + theta*H[:,action2]
        
        self.reward[t-window_size] = self.weight.dot(self.R[:,t])
        