from dpart.dpart import dpart
from dpart.methods import ProbabilityTensor


class Histogram(dpart):
    def __init__(self, bounds: dict = None, n_bins: int = 20, epsilon: dict = None):
        self.n_bins = n_bins
        super().__init__(
            epsilon=epsilon,
        )

    def default_method(self):
        return ProbabilityTensor(n_parents=0, n_bins=self.n_bins)
