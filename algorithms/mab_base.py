import numpy as np

class MabBase:
    """
    Base class for multi-armed bandit algorithms.
    This class provides a common interface for all multi-armed bandit algorithms.
    It should be subclassed to implement specific algorithms.
    Attributes:
        R (ndarray): The reward matrix of shape (n_arms, n_samples).
        n_arms (int): The number of arms.
        n_samples (int): The number of samples.
        window_size (int): The size of the sliding window for the algorithm.
        reward (numpy.ndarray): The reward array.
        played_times (numpy.ndarray): The number of times each arm has been played.
    """
    def __init__(self, R, window_size=120):
        self.R = R
        self.n_arms, self.n_samples = R.shape
        self.window_size = window_size
        self.reward = np.ones(self.n_samples - self.window_size)
        self.played_times = np.ones(self.n_arms)

    def run(self):
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def cutoff(self):
        return 24 # by default, the cutoff is 24

    def orthogonal_portfolio(self, data):
        """
        Compute the orthogonal portfolio based on the covariance matrix of the returns.
        Parameters:
            data (numpy.ndarray): The input data slice of shape (n_arms, window_size)
        
        Returns: tuple: A tuple containing:
            normalized_eigenvectors (numpy.ndarray): The normalized eigenvectors.
            normalized_eigenvalues (numpy.ndarray): The normalized eigenvalues.
            portfolio_reward (numpy.ndarray): The portfolio reward.
            sharpe_ratio (numpy.ndarray): The Sharpe ratio of each portfolio.
        """
        # Step 1: Center the data (skipping this step as the data is already centered)

        # Step 2: Compute the covariance matrix
        convariance_matrix = np.cov(data)

        # Step 3: Perform eigenvalue decomposition
        # eigenvalues(A), eigenvectors(H)
        eigenvalues, eigenvectors = np.linalg.eig(convariance_matrix)    # equation 5

        # Check if all eigenvalues are non-negative
        assert(np.all(eigenvalues >= 0))  
        
        # Step 4: Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices] # n(number of assets) orthogonal portfolios

        # Step 5: Normalize eigenvectors matrix (L1 normalization)
        # Normalize eigenvectors matrix (equation 7)
        eigenvectors /= np.sum(eigenvectors, axis=0)
        # Normalize eigenvalues matrix (equation 8)
        eigenvalues = eigenvectors.T.dot(convariance_matrix).dot(eigenvectors)

        # Step 6: Compute the Sharpe Ratio of each portfolio (equation 10)
        portfolio_reward = eigenvectors.T.dot(data)
        sharpe_ratio = np.mean(portfolio_reward, axis=1) / np.sqrt(eigenvalues.diagonal())

        return eigenvectors, eigenvalues, portfolio_reward, sharpe_ratio
    
    def update(self, t, H, A, passive, active):
        """
        Compute the optimal weights and rewards for the passive and active portfolios.
        Parameters:
            t (int): The current time step.
            H (numpy.ndarray): The eigenvectors matrix.
            A (numpy.ndarray): The eigenvalues matrix.
            passive (int): The index of the passive portfolio.
            active (int): The index of the active portfolio.
        """
        self.played_times[passive] += 1
        self.played_times[active] += 1

        Adiag = A.diagonal()
        theta = Adiag[passive] / (Adiag[active] + Adiag[passive])
        self.weight = (1 - theta) * H[:, passive] + theta * H[:, active]
        self.reward[t - self.window_size] = self.weight.dot(self.R[:, t])

    def get_cumulative_wealth(self):
        """
        Compute the cumulative wealth of the algorithm.
        Returns:
            numpy.ndarray: The cumulative wealth array.
        """
        return np.cumprod(self.reward)



    def test(self):
        for t in range(self.window_size, self.n_samples):
            # Get current slice from previous data (window size)
            R_slice = self.R[:, t - self.window_size:t]

            # Compute the covariance matrix of the slice returns
            convariance_matrix = np.cov(R_slice)

            # Eigenvalue decomposition
            # A is eigenvalues, H is eigenvectors
            A, H = np.linalg.eig(convariance_matrix)    # equation 5

            # Check if all eigenvalues are non-negative
            assert(np.all(A >= 0))  

            print(f"A: {A}")
            
            # Sort the eigenvalues
            index = np.argsort(-A)
            # print(f"index: {index}")
            A = np.diag(A[index])    # eigenvalues as vector
            H = H[:, index]   # n(number of assets) orthogonal portfolios
            cutoff = np.argwhere(np.median(np.diag(A)) > np.diag(A))[0][0]

            # Normalize eigenvectors matrix
            H /= np.sum(H, axis=0)  # equation 7
            
            # Normalize eigenvalues matrix 
            A_norm = H.T.dot(convariance_matrix).dot(H) # equation 8

            # Compute the Sharpe Ratio
            portfolio_reward = H.T.dot(R_slice)
            sharpe_ratio = np.mean(portfolio_reward, axis=1) / np.sqrt(A_norm.diagonal())

            # Compute the Upper Bound of expected reward
            sr_upper_bound = sharpe_ratio + np.sqrt((2 * np.log(t)) / (self.window_size+self.played_times))

            # Compute the optimal portfolio
            passive = np.argmax(sr_upper_bound[:cutoff])
            active = np.argmax(sr_upper_bound[cutoff:]) + cutoff

            print(f"passive: {passive}")
            print(f"active: {active}")

            self.played_times[passive] += 1
            self.played_times[active] += 1

            # # Optimize the weights
            Adiag = A_norm.diagonal()
            theta = Adiag[passive] / (Adiag[active] + Adiag[passive])

            self.weight = (1 - theta) * H[:, passive] + theta * H[:, active]
            self.reward[t - self.window_size] = self.weight.dot(self.R[:, t])

        print(f"reward: {self.reward}")