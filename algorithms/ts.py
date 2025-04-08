import numpy as np
from algorithms.mab_base import MabBase

class ThompsonSampling(MabBase):
    """
    Thompson Sampling algorithm for multi-armed bandit problems.
    This class implements the Thompson Sampling algorithm for selecting
    the best arm based on the posterior distribution of the arms' rewards.
    """
    def __init__(self, R, window_size=120):
        super().__init__(R, window_size)
        self.mv_reward = np.ones(self.n_samples - self.window_size)
        self.success = np.ones(4)
        self.failure = np.ones(4)

    def run(self):
        pass