from dpart.dpart import dpart
from dpart.methods import ProbabilityTensor


class PrivBayes(dpart):
    def __init__(self, bounds: dict = None, n_parents: int = 2, n_bins: int = 20, epsilon: dict = None):
        self.n_bins = n_bins
        super().__init__(
            epsilon=epsilon, prediction_matrix="infer", n_parents=n_parents
        )

    def default_method(self, dtype):
        return ProbabilityTensor(n_parents=None, n_bins=self.n_bins)
