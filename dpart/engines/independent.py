from dpart.dpart import dpart
from dpart.methods import ProbabilityTensor


class Independent(dpart):
    def __init__(self, bounds: dict = None, n_bins: int = 20, epsilon: dict = None):
        self.n_bins = n_bins
        super().__init__(
            epsilon=epsilon,
            bounds=bounds,
            n_parents=0
        )

    def default_method(self, dtype):
        return ProbabilityTensor(n_bins=self.n_bins)
