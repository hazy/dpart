import pandas as pd
from pathlib import Path
from typing import Tuple


class Sampler:
    category_support: bool = True
    numerical_support: bool = True

    def __init__(self, epsilon: float):
        self.set_epsilon(epsilon)

    def set_epsilon(self, epsilon: float):
        self.epsilon = epsilon

    def preprocess_X(self, X: pd.DataFrame) -> pd.DataFrame:
        return X

    def preprocess_y(self, y: pd.Series) -> pd.Series:
        return y

    def preprocess(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        return self.preprocess_X(X), self.preprocess_y(y)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        pass

    def postprocess_y(self, y: pd.Series) -> pd.Series:
        return y

    def sample(self, X: pd.DataFrame) -> pd.Series:
        pass

    def store(self, folder: Path):
        raise NotImplementedError

    def load(self, folder: Path):
        raise NotImplementedError
