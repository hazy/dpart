import pandas as pd
from pathlib import Path
from typing import Tuple


class Sampler:
    category_support: bool = True
    numerical_support: bool = True

    def __init__(self, epsilon: float):
        self.epsilon = epsilon

    def preprocess(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        return X, y

    def fit(self, X: pd.DataFrame, y: pd.Series):
        pass

    def postprocess(self, y: pd.Series) -> pd.Series:
        return y

    def sample(self, X: pd.DataFrame) -> pd.Series:
        pass

    def store(self, folder: Path):
        raise NotImplementedError

    def load(self, folder: Path):
        raise NotImplementedError
