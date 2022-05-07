import numpy as np
import pandas as pd

from dpart.methods.utils.dp_utils import laplace_mechanism
from dpart.methods.utils.stats import normalise_proba, pchoice
from dpart.methods.utils.decorators import ignore_warning
from dpart.methods.utils.bin_encoder import BinEncoder
from dpart.methods.base.sampler import Sampler


class ProbabilityTensor(Sampler):
    def __init__(self, epsilon: float = None, n_bins=20, n_parents: int = None):
        super().__init__(epsilon=epsilon)
        self.n_bins = n_bins  # np.linspace(0, 1, n_bins + 1)
        self.X_encoders = None
        self.y_encoder = None
        self.X_cols = None
        self.bin_y = None
        self.conditional_dist = None
        self.n_parents = n_parents
        self.parents = None

    @ignore_warning
    def preprocess_X(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.X_encoders is None:
            self.X_cols = list(X.columns)

            self.X_encoders = {}
            # X Processor
            for col, series in X.items():
                self.X_encoders[col] = BinEncoder(n_bins=self.n_bins)
                self.X_encoders[col].fit(X[col])

        X_t = pd.DataFrame(
            {
                col: self.X_encoders[col].transform(series)
                for col, series in X.items()
            },
            index=X.index,
            dtype="int64",
        ).reindex(columns=self.X_cols)

        if self.n_parents is not None:
            self.parents = np.random.choice(
                list(X_t.columns), size=min(X_t.shape[1], self.n_parents), replace=False
            )
            X_t = X_t[self.parents]
        else:
            self.parents = X_t.columns
        return X_t

    @ignore_warning
    def preprocess_y(self, y: pd.Series) -> pd.Series:
        if self.y_encoder is None:
            self.y_encoder = BinEncoder(n_bins=self.n_bins)
            self.y_encoder.fit(y)

        y_t = self.y_encoder.transform(y)
        return y_t

    def fit(self, X: pd.DataFrame, y: pd.Series):
        joint_df = X.join(y)
        joint_counts = np.zeros(shape=list(joint_df.max(axis=0) + 1))

        for node_tuple, grp in joint_df.groupby(by=list(joint_df.columns)):
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

        self.conditional_dist = normalise_proba(
            joint_counts, conditional=True, _clip=True
        )

    def postprocess_y(self, y: pd.Series) -> pd.Series:
        return self.y_encoder.inverse_transform(y)

    def sample(self, X: pd.DataFrame) -> pd.Series:
        if len(self.parents) == 0:
            y = np.random.choice(self.conditional_dist.shape[-1], p=self.conditional_dist, size=X.shape[0])
        else:
            dists = self.conditional_dist[
                tuple([tuple(X[parent]) for parent in self.parents])
            ]

            y = pchoice(p=dists)
        return pd.Series(y, index=X.index)
