import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import KBinsDiscretizer

from dpar.methods.base.sampler import Sampler


class ProbababilityTensor(Sampler):
    def __init__(self, epsilon: float = 1.0, n_bins=100):
        super().__init__(epsilon=epsilon)
        self.n_bins = n_bins  # np.linspace(0, 1, n_bins + 1)
        self.X_encoder = None
        self.y_encoder = None
        self.num_cols = None
        self.bin_y = None

    def preprocess(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        if self.label_encoder is None:
            self.num_cols = [
                col for col, dtype in X.dtypes.items() if dtype.kind in "Mmfui"
            ]
            # X Processor
            self.X_encoder = KBinsDiscretizer(encode="ordinal")
            self.X_encoder.fit(X)
            self.bin_y = y.dtype.kind in "fui"
            if self.bin_y:
                self.y_encoder = KBinsDiscretizer(encode="ordinal")
                self.y_encoder.fit(y)

        X = self.X_encoder.transform(X)
        y = self.y_encoder.transform(y)
        return X, y

    def fit(self, X: pd.DataFrame, y: pd.Series):
        pass

    def postprocess(self, y):
        if self.y_encoder is not None:
            y = self.y_encoder.inverse_transform(y)
        return y

    def sample(self, X):
        pass
