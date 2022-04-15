import numpy as np
import pandas as pd
from typing import Tuple
from diffprivlib.models import LinearRegression as DPLR

from dpar.methods.base import NumericalSampler
from dpar.methods.utils.sklearn_encoder import SklearnEncoder


class LinearRegression(NumericalSampler):
    def __init__(self, epsilon: float = 1.0, *args, **kwargs):
        super().__init__(epsilon=epsilon)
        self.X_encoder = None
        self.lr = DPLR(epsilon=(self.epsilon / 2), *args, **kwargs)
        self.sigma = None

    def preprocess(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        if self.label_encoder is None:
            # X Processor
            self.X_encoder = SklearnEncoder()
            self.X_encoder.fit(X)

        X = self.X_encoder.transform(X)
        return X, y

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.lr.fit(X, y)
        # compute sigma
        y_pred = self.lr.predict(X)
        residuals = y - y_pred
        normaliser = X.shape[0] - X.shape[1] - 1
        sensitivity = np.sqrt(1 / normaliser)  # assuming y is bounded between 0 and 1
        additive_noise = np.random.laplace(
            0, scale=sensitivity / (self.epsilon / 2)
        )  # laplace mechanism
        self.sigma = np.sqrt(np.sum(residuals ** 2) / normaliser) + additive_noise

    def postprocess(self, y: pd.Series) -> pd.Series:
        return y

    def sample(self, X: pd.DataFrame) -> pd.Series:
        y_pred = self.lr.predict(X) + np.random.normal(
            scale=self.sigma, size=X.shape[0]
        )
        # NOTE maybe clip to min/max bounds
        return y_pred
