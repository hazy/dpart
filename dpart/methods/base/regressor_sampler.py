import numpy as np
import pandas as pd

from dpart.methods.base import NumericalSampler
from dpart.methods.utils.sklearn_encoder import SklearnEncoder


class RegressorSampler(NumericalSampler):
    reg_class = None
    dp_reg_class = None

    def __init__(self, epsilon: float = None, one_hot: bool = False, *args, **kwargs):
        super().__init__(epsilon=epsilon)
        self.X_encoder = None
        self.sigma = None
        self.reg = None
        self.one_hot = one_hot
        self.args = args
        self.kwargs = kwargs

    def preprocess_X(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.X_encoder is None:
            # X Processor
            self.X_encoder = SklearnEncoder(
                mode=("one-hot" if self.one_hot else "ordinal")
            )
            self.X_encoder.fit(X)

        return pd.DataFrame(
            self.X_encoder.transform(X), columns=X.columns, index=X.index
        )

    def preprocess_y(self, y: pd.Series) -> pd.Series:
        return y

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if self.epsilon is not None:
            self.reg = self.dp_reg_class(
                epsilon=(self.epsilon / 2 if self.epsilon is not None else None),
                *self.args,
                **self.kwargs
            )
        else:
            self.reg = self.reg_class(
                *self.args,
                **self.kwargs
            )
        self.reg.fit(X, y)
        # compute sigma
        y_pred = self.reg.predict(X)
        residuals = y - y_pred
        normaliser = X.shape[0] - X.shape[1] - 1
        sensitivity = np.sqrt(1 / normaliser)  # assuming y is bounded between 0 and 1
        if self.epsilon is not None:
            additive_noise = np.random.laplace(
                0, scale=sensitivity / (self.epsilon / 2)
            )  # laplace mechanism
        else:
            additive_noise = 0

        self.sigma = np.abs(
            np.sqrt(np.sum(residuals ** 2) / normaliser) + additive_noise
        )

    def postprocess_y(self, y: pd.Series) -> pd.Series:
        return y

    def sample(self, X: pd.DataFrame) -> pd.Series:
        y_pred = self.reg.predict(X) + np.random.normal(
            scale=self.sigma, size=X.shape[0]
        )
        # NOTE maybe clip to min/max bounds
        return pd.Series(y_pred, index=X.index)
