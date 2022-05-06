from dpart.methods.probability_tensor import ProbabilityTensor


class HistogramSampler(ProbabilityTensor):
    def __init__(self, epsilon: float = None, n_bins=20):
        super().__init__(epsilon=epsilon, n_bins=n_bins, n_parents=0)
