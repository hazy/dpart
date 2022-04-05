import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import OrdinalEncoder
from diffprivlib.models import LogisticRegression as DPLR

from dpar.methods.base import CategorySampler
from dpar.methods.utils.sklearn_encoder import SklearnEncoder


class LogisticRegression(CategorySampler):
    def __init__(self, epsilon=1.0, *args, **kwargs):
        super().__init__(epsilon=epsilon)
        self.label_encoder = None
        self.X_encoder = None
        self.lr = DPLR(epsilon=self.epsilon, *args, **kwargs)

    def preprocess(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        if self.label_encoder is None:
            # X Processor
            self.X_encoder = SklearnEncoder()
            self.X_encoder.fit(X)
            # y Processor
            self.label_encoder = OrdinalEncoder()
            self.label_encoder.fit(y[:, np.newaxis])

        X = self.X_encoder.transform(X)
        y = self.label_encoder.transform(y[:, np.newaxis])[:, 0]

        return X, y

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.lr.fit(X, y)

    def postprocess(self, y: pd.Series) -> pd.Series:
        return self.label_encoder.inverse_transform(y)

    def sample(self, X: pd.DataFrame) -> pd.Series:
        y_proba = self.lr.predict_proba(X)

        uniform_noise = np.random.uniform(size=[X.shape[0], 1])
        y = np.sum(uniform_noise > np.cumsum(y_proba, axis=1), axis=1).astype(int)
        return y
