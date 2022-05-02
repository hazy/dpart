import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler


class SklearnEncoder:
    accepted_modes = ["ordinal", "one-hot"]

    def __init__(self, mode="ordinal"):
        self.mode = mode
        self.encoders = None
        assert (
            mode in self.accepted_modes
        ), f"mode '{mode}' not supported, only {self.accepted_modes}  are currently supported"

    def fit(self, X: pd.DataFrame):
        self.encoders = {}
        for col, series in X.items():
            if series.dtype.kind in "OSb":
                if self.mode == "ordinal":
                    self.encoders[col] = OrdinalEncoder()
                else:
                    self.encoders[col] = OneHotEncoder()
            else:
                self.encoders[col] = MinMaxScaler()

            self.encoders[col].fit(X[[col]])

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                col: self.encoders[col].transform(series.to_frame()).squeeze()
                for col, series in X.items()
            },
            index=X.index,
        )

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        return self.transform(X)

    def store(self, folder: Path):
        raise NotImplementedError

    def load(self, folder: Path):
        raise NotImplementedError
