import numpy as np
import pandas as pd
from typing import Union
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder

from dpar.methods.utils.dp_utils import laplace_mechanism
from dpar.methods.utils.stats import normalise_proba, pchoice
from dpar.methods.base.sampler import Sampler


class ProbababilityTensor(Sampler):
    def __init__(self, epsilon: float = 1.0, n_bins=100):
        super().__init__(epsilon=epsilon)
        self.n_bins = n_bins  # np.linspace(0, 1, n_bins + 1)
        self.X_encoders = None
        self.y_encoder = None
        self.X_cols = None
        self.bin_y = None
        self.conditional_dist = None

    def get_encoder(self, dkind: str) -> Union[KBinsDiscretizer, OrdinalEncoder]:
        if dkind in "fui":
            return KBinsDiscretizer(encode="ordinal")
        else:
            return OrdinalEncoder()

    def preprocess_X(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.X_encoders is None:
            self.X_cols = list(X.columns)

            self.X_encoders = {}
            # X Processor
            for col, series in X.items():
                self.X_encoders[col] = self.get_encoder(dkind=series.dtype.kind)
                self.X_encoders[col].fit(X[col].to_frame())

        X_t = pd.DataFrame(
            {
                col: self.X_encoders[col].transform(series.to_frame())[:, 0]
                for col, series in X.items()
            },
            index=X.index,
        ).reindex(columns=self.X_cols)

        return X_t

    def preprocess_y(self, y: pd.Series) -> pd.Series:
        if self.y_encoder is None:
            self.y_encoder = self.get_encoder(dkind=y.dtype.kind)
            self.y_encoder.fit(y.to_frame())

        return pd.Series(
            self.y_encoder.transform(y.to_frame())[:, 0], index=y.index, name=y.name
        )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        joint_counts = np.zeros(shape=list(X.max(axis=0) + 1) + [y.max() + 1])
        joint_df = X.join(pd.Series(y, name=X.shape[1]))
        for node_tuple, grp in joint_df.groupby(joint_df.columns):
            joint_counts[node_tuple] += grp.shape[0]

        # add laplace noise
        if self.epsilon is not None:
            # NOTE we use sensitivity of `2` (not `2 / n_rows`) because we apply the noise to the counts (not the probabilities)
            sensitivity = 2
            # NOTE we use epsilon budget of `len(self.network)`` not (`len(self.network) - n_parents`) because we calculate the counts for each node
            # add noise
            joint_counts = laplace_mechanism(
                joint_counts, sensitivity=sensitivity, epsilon=self.epsilon
            )

        self.conditional_dist = normalise_proba(joint_counts, conditional=True)

    def postprocess_y(self, y: pd.Series) -> pd.Series:
        return pd.Series(
            self.y_encoder.inverse_transform(y.to_frame())[:, 0], y=y.index, name=y.name
        )

    def sample(self, X: pd.DataFrame) -> pd.Series:
        dists = self.conditional_dist[
            tuple([tuple(X[parent]) for parent in self.X_cols["all"]])
        ]

        y = pchoice(p=dists)
        return pd.Series(y, index=X.index)
