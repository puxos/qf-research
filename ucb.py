import numpy as np

def orthogonal_portfolio(sliceReturns, n, n_i):
    # Compute the covariance matrix of the slice returns
    # sliceReturns is a 2D array where each row represents a different arm
    # and each column represents a different round
    convariance_matrix = np.cov(sliceReturns)

    # Eigenvalue decomposition
    # The covariance matrix is symmetric, so we can use np.linalg.eigh
    # to compute the eigenvalues and eigenvectors
    # A is eigenvalues, H is eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(convariance_matrix)

    # Make sure the eigenvalues are real and positive
    assert(eigenvalues.sum(eigenvalues < 0) == 0)
    
    # Sort the eigenvalues
    index = np.argsort(-eigenvalues)
    A = np.diag(eigenvalues[index])
    H = eigenvectors[:, index]

    l = np.argwhere(np.median(np.diag(A)) > np.diag(A))[0][0]

    # Normalize weight
    H /= np.sum(H, axis=0)
    A_new = H.T.dot(convariance_matrix).dot(H)

    # Compute the Sharpe Ration
    portfolio_reward = H.T.dot(sliceReturns)
    sharpe_ratio = np.mean(portfolio_reward, axis=1) / np.sqrt(A_new.diagonal())

    # Compute the Upper Bound of expected reward
    sr_upper_bound = sharpe_ratio + np.sqrt(2 * np.log(n) / n_i) * np.sqrt(A_new.diagonal())

def compute_sharpe_ratio(sliceReturns, n, n_i):
    """
    Compute the Sharpe Ratio of the portfolio
    """
    # Compute the covariance matrix of the slice returns
    # sliceReturns is a 2D array where each row represents a different arm
    # and each column represents a different round
    convariance_matrix = np.cov(sliceReturns)

    # Eigenvalue decomposition
    # The covariance matrix is symmetric, so we can use np.linalg.eigh
    # to compute the eigenvalues and eigenvectors
    # A is eigenvalues, H is eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(convariance_matrix)

    # Make sure the eigenvalues are real and positive
    assert(eigenvalues.sum(eigenvalues < 0) == 0)
    
    # Sort the eigenvalues
    index = np.argsort(-eigenvalues)
    A = np.diag(eigenvalues[index])
    H = eigenvectors[:, index]

    l = np.argwhere(np.median(np.diag(A)) > np.diag(A))[0][0]

    # Normalize weight
    H /= np.sum(H, axis=0)

