import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder


class SklearnEncoder:
    accepted_modes = ["ordinal", "one-hot"]

    def __init__(self, mode="ordinal"):
        self.cat_cols = None
        self.mode = mode
        self.encoder = None
        assert (
            mode in self.accepted_modes
        ), f"mode '{mode}' not supported, only {self.accepted_modes}  are currently supported"

    def fit(self, X: pd.DataFrame):
        self.cat_cols = [
            col for col, col_data in X.items() if col_data.dtype.kind in "OSb"
        ]
        if self.mode == "ordinal":
            self.encoder = OrdinalEncoder()
        elif self.mode == "one-hot":
            self.encoder = OneHotEncoder()

        self.encoder.fit(X[self.cat_cols])

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        encoded_data = self.encoder.transform(X[self.cat_cols])
        return X.drop(self.cat_cols, axis=1).join(encoded_data, axis=1)

    def store(self, folder: Path):
        raise NotImplementedError

    def load(self, folder: Path):
        raise NotImplementedError
