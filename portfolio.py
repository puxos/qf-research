import numpy as np
from scipy.stats import norm, kurtosis, skew
from sklearn.preprocessing import MinMaxScaler

class BanditPortfolio:
    def __init__(self, R):
        self.R = R
        self.n_arms, self.n_samples = R.shape
        
    def UCB(self, window_size):
        self.reward = np.ones(self.n_samples - window_size)
        self.played_times = np.ones(self.n_arms)
        
        for t in range(window_size, self.n_samples):
            sliceR = self.R[:, t-window_size:t]
            
            # Compute the covariance matrix
            covariance_matrix =  np.cov(sliceR)
            
            # Eigenvalue Decomposition
            A, H = np.linalg.eig(covariance_matrix)
            
            # All eigenvalues are positive
            assert(np.sum(A<0) == 0)
            
            # Sort the eigenvalues
            idx = np.argsort(-A)
            A = np.diag(A[idx])
            H = H[:,idx]
            l = np.argwhere(np.median(np.diag(A)) > np.diag(A))[0][0]
            
            # Normalized weight
            H /= np.sum(H, axis= 0)
            ANew =  H.T.dot(covariance_matrix).dot(H)
            
            # Compute the sharpe ratio
            portfolio_reward = H.T.dot(sliceR)

            sharpe_ratio = np.mean(portfolio_reward, axis=1) / np.sqrt(ANew.diagonal())
            #sharpe_ratio = MinMaxScaler().fit_transform(sharpe_ratio.reshape(-1,1)).reshape(-1)
            
            # Compute the upper bound of expected reward
            sharpe_ratio_upper_bound = sharpe_ratio + np.sqrt((2*np.log(t))/(window_size+self.played_times))
            
            # Select the optimal arm
            action1 = np.argmax(sharpe_ratio_upper_bound[:l])
            action2 = np.argmax(sharpe_ratio_upper_bound[l:])+l

            self.played_times[action1] += 1
            self.played_times[action2] += 1

            # Optimal weight
            Adiag = ANew.diagonal()
            theta = Adiag[action1] / (Adiag[action1] + Adiag[action2])
            self.weight = (1-theta)*H[:,action1] + theta*H[:,action2]
            
            self.reward[t-window_size] = self.weight.dot(self.R[:,t])
            
    def TS(self, window_size):
        self.reward = np.ones(self.n_samples - window_size)
        self.mv_reward = np.ones(self.n_samples - window_size)
        self.played_times = np.ones(self.n_arms)
        self.success = np.ones(4)
        self.fail = np.ones(4)

        for t in range(window_size, self.n_samples):

            sliceR = self.R[:, t-window_size:t]

            # Compute the covariance matrix
            covariance_matrix = np.cov(sliceR)  # histroical covariance

            # Eigenvalue Decomposition
            A, H = np.linalg.eig(covariance_matrix)  # equation 5

            # All eigenvalues are positive
            assert(np.sum(A < 0) == 0)

            # Sort the eigenvalues
            idx = np.argsort(-A)  # sort eigenvalues
            A = np.diag(A[idx])  #  eigenvalues as vector
            H = H[:, idx]  #  n(number of assets) orthonormal portfolios
            l = np.argwhere(np.median(np.diag(A)) > np.diag(A))[0][0]

            # Normalized weight
            H /= np.sum(H, axis=0)  # equation 7 normalized eigenvectors
            #  equation 8 normalized eigenvalues matrix
            ANew = H.T.dot(covariance_matrix).dot(H)

            # Compute the sharpe ratio
            #  Return of each port. in the slice window
            portfolio_reward = H.T.dot(sliceR)
            # Estimator of portfolio return, rolling average
            sharpe_ratio = np.mean(
                portfolio_reward, axis=1) / np.sqrt(ANew.diagonal())
            # Normalize the sharpe ratio
            #sharpe_ratio = MinMaxScaler().fit_transform(
            #    sharpe_ratio.reshape(-1, 1)).reshape(-1)

            # Compute the upper bound of expected reward
            sharpe_ratio_upper_bound = sharpe_ratio + \
                np.sqrt((2*np.log(t))/(window_size+self.played_times))

            # Select the optimal arm
            #  passive portfolios
            action1 = np.argmax(sharpe_ratio_upper_bound[:l])
            #  active portfolios
            action2 = np.argmax(sharpe_ratio_upper_bound[l:])+l

            #  update the times portfolio played
            self.played_times[action1] += 1
            self.played_times[action2] += 1  # update the second action

            # Optimal weight, Min var allocation between 2 chosen portfolios
            Adiag = ANew.diagonal()
            theta = Adiag[action1] / (Adiag[action1] + Adiag[action2])
            
            self.psr_set = []
            #psr_second_set = []
            for a in range(len(sharpe_ratio)):
                sr = sharpe_ratio[a]
                n = (window_size+self.played_times[a])/window_size
                skewness = skew(portfolio_reward[a, :])
                kurto = kurtosis(portfolio_reward[a, :])
                nomin = (sr-np.mean(sharpe_ratio))*np.sqrt(n)
                denom = np.sqrt(np.abs((1 + 0.5*sr**2 - skewness*sr + ((kurto-3)/4)*sr**2))/(n-1))
                self.psr_set.append(nomin/denom)
                
                
            self.psr_set = np.array([norm.cdf(a) for a in self.psr_set])
            
            sharpe_ratio_upper_bound_psr = (sharpe_ratio + \
                np.sqrt((2*np.log(t))/(window_size+self.played_times)))*np.array(self.psr_set)
                
            
            action_1_1 = np.argmax(sharpe_ratio_upper_bound_psr[:l])
            action_1_2 = np.argmax(sharpe_ratio_upper_bound_psr[l:])+l


            # Optimal weight
            theta_ = Adiag[action_1_1] / (Adiag[action_1_1] + Adiag[action_1_2])
                        
            
            
            self.weight_1 = (1-theta)*H[:, action1] + theta*H[:, action2]
            
            self.weight_2 = np.ones(self.n_arms)/self.n_arms
            
            self.weight_3 = (np.linalg.inv(covariance_matrix)@np.ones(self.n_arms).reshape(-1, 1))/(np.ones(
                self.n_arms).reshape(-1, 1).T@np.linalg.inv(covariance_matrix)@np.ones(self.n_arms).reshape(-1, 1))
            
            self.weight_4 = (1-theta_)*H[:,action_1_1] + theta_*H[:,action_1_2]
            
            
            final_actions = [self.weight_1, self.weight_2, self.weight_3, self.weight_4]
            draws = [np.random.beta(self.success[action], self.fail[action])
                     for action in range(4)]
            
            final_weight = final_actions[np.argmax(draws)]
            
            self.reward[t-window_size] = final_weight.T.dot(self.R[:, t])
            self.mv_reward[t-window_size] = self.weight_3.T.dot(self.R[:, t])
            
            rewards = [self.weight_1.T.dot(self.R[:, t]), self.weight_2.T.dot(self.R[:, t]), self.weight_3.T.dot(self.R[:, t]), self.weight_4.T.dot(self.R[:, t])]
            
            

            if np.max(rewards) == final_weight.T@self.R[:, t]:
                self.success[np.argmax(draws)] += 1
                other_idx = [a for a in range(4) if a != np.argmax(rewards)]
                self.fail[other_idx[0]] +=1
                self.fail[other_idx[1]] +=1
                self.fail[other_idx[2]] +=1
            else:
                other_idx = [a for a in range(4) if (a != np.argmax(rewards)) & (a != np.argmax(draws))]
                self.success[np.argmax(rewards)] +=1
                self.fail[other_idx[0]] +=1
                self.fail[other_idx[1]] +=1
                self.fail[np.argmax(draws)] += 1
                
    def UCBPSR(self, window_size):
        self.reward = np.ones(self.n_samples - window_size)
        self.played_times = np.ones(self.n_arms)
        self.psr = {}
        
        for t in range(window_size, self.n_samples):
            sliceR = self.R[:, t-window_size:t]
            
            # Compute the covariance matrix
            covariance_matrix =  np.cov(sliceR)
            
            # Eigenvalue Decomposition
            A, H = np.linalg.eig(covariance_matrix)
            
            # All eigenvalues are positive
            assert(np.sum(A<0) == 0)
            
            # Sort the eigenvalues
            idx = np.argsort(-A)
            A = np.diag(A[idx])
            H = H[:,idx]
            l = np.argwhere(np.median(np.diag(A)) > np.diag(A))[0][0]
            
            # Normalized weight
            H /= np.sum(H, axis= 0)
            ANew =  H.T.dot(covariance_matrix).dot(H)
            
            # Compute the sharpe ratio
            portfolio_reward = H.T.dot(sliceR)

            
            sharpe_ratio = (np.mean(portfolio_reward, axis=1) / np.sqrt(ANew.diagonal()))
            #sharpe_ratio = MinMaxScaler().fit_transform(sharpe_ratio.reshape(-1,1)).reshape(-1)
            #scl = MinMaxScaler()
            #sharpe_ratio = scl.fit_transform(sharpe_ratio.reshape(-1,1)).reshape(-1)
            
            
            
            self.psr_set = []
            for a in range(len(sharpe_ratio)):
                sr = sharpe_ratio[a]
                n = (window_size+self.played_times[a])/window_size
                skewness = skew(portfolio_reward[a, :])
                kurto = kurtosis(portfolio_reward[a, :])
                nomin = (sr-np.mean(sharpe_ratio))*np.sqrt(n)
                denom = np.sqrt(np.abs((1 + 0.5*sr**2 - skewness*sr + ((kurto-3)/4)*sr**2))/(n-1))
                #psr = norm.cdf(nomin/denom)
                self.psr_set.append(nomin/denom)
            #self.psr_set = scl.fit_transform(np.array(self.psr_set).reshape(-1,1)).reshape(-1)
            self.psr_set = np.array([norm.cdf(a) for a in self.psr_set])
            self.psr[t] = self.psr_set
            
            # Compute the upper bound of expected reward
            #sharpe_ratio_upper_bound = sharpe_ratio + np.sqrt((2*np.log(t))/(window_size+self.played_times))
            sharpe_ratio_upper_bound = (sharpe_ratio + \
                np.sqrt((2*np.log(t))/(window_size+self.played_times)))*np.array(self.psr_set)
            
            action1 = np.argmax(sharpe_ratio_upper_bound[:l])
            action2 = np.argmax(sharpe_ratio_upper_bound[l:])+l
            
            # Select the optimal arm
            #action1 = np.argmax(self.psr_set[:l])
            #action2 = np.argmax(self.psr_set[l:])+l

            self.played_times[action1] += 1
            self.played_times[action2] += 1

            # Optimal weight
            Adiag = ANew.diagonal()
            theta = Adiag[action1] / (Adiag[action1] + Adiag[action2])
            self.weight = (1-theta)*H[:,action1] + theta*H[:,action2]
            
            self.reward[t-window_size] = self.weight.dot(self.R[:,t])
            
    def PSR(self, window_size):
        self.reward = np.ones(self.n_samples - window_size)
        self.played_times = np.ones(self.n_arms)
        self.psr = {}
        
        for t in range(window_size, self.n_samples):
            sliceR = self.R[:, t-window_size:t]
            
            # Compute the covariance matrix
            covariance_matrix =  np.cov(sliceR)
            
            # Eigenvalue Decomposition
            A, H = np.linalg.eig(covariance_matrix)
            
            # All eigenvalues are positive
            assert(np.sum(A<0) == 0)
            
            # Sort the eigenvalues
            idx = np.argsort(-A)
            A = np.diag(A[idx])
            H = H[:,idx]
            l = np.argwhere(np.median(np.diag(A)) > np.diag(A))[0][0]
            
            # Normalized weight
            H /= np.sum(H, axis= 0)
            ANew =  H.T.dot(covariance_matrix).dot(H)
            
            # Compute the sharpe ratio
            portfolio_reward = H.T.dot(sliceR)

            
            sharpe_ratio = (np.mean(portfolio_reward, axis=1) / np.sqrt(ANew.diagonal()))
            scl = MinMaxScaler(feature_range=(-3,3))
            #sharpe_ratio = scl.fit_transform(sharpe_ratio.reshape(-1,1)).reshape(-1)
            
            self.psr_set = []
            #psr_second_set = []
            for a in range(len(sharpe_ratio)):
                sr = sharpe_ratio[a]
                n = (window_size+self.played_times[a])/window_size
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