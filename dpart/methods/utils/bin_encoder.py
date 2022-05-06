import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder


class BinEncoder():
    def __init__(self, n_bins=20):
        self.n_bins = 20
        self.encoder = None
        self.dkind = None

    def fit(self, data: pd.Series):
        self.dkind = data.dtype.kind
        if self.dkind in "OSb":
            self.encoder = KBinsDiscretizer(bins=self.n_bins, encode="ordinal")
        else:
            self.encoder = OrdinalEncoder()

        self.encoder.fit(data.to_frame())

    def transform(self, data: pd.Series) -> pd.Series:
        return pd.Series(self.encoder.transform(data.to_frame()).squeeze(), index=data.index, name=data.name)

    def fit_transform(self, data: pd.Series) -> pd.Series:
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data: pd.Series) -> pd.Series:
        if self.dkind == "OSb":
            samples = self.encoder.inverse_transform(data.to_frame()).sequeeze()
        else:
            bin_edges = self.encoder.bin_edges_
            samples = np.random.uniform(bin_edges[data.to_numpy()], bin_edges[data.to_numpy() + 1])

        return pd.Series(samples, index=data.index, name=data.name)
