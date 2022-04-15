import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder

from dpar.methods.utils.dp_utils import laplace_mechanism
from dpar.methods.utils.stats import normalise_proba, pchoice
from dpar.methods.base.sampler import Sampler


class ProbababilityTensor(Sampler):
    def __init__(self, epsilon: float = 1.0, n_bins=100):
        super().__init__(epsilon=epsilon)
        self.n_bins = n_bins  # np.linspace(0, 1, n_bins + 1)
        self.X_encoder = None
        self.y_encoder = None
        self.X_cols = None
        self.bin_y = None
        self.conditional_dist = None

    def preprocess(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        if self.X_encoder is None:
            self.X_cols = {
                "all": list(X.columns),
                "num": [
                    col for col, dtype in X.dtypes.items() if dtype.kind in "Mmfui"
                ],
            }
            self.X_cols["cat"] = [col for col in X if col not in self.X_cols["num"]]
            self.X_encoder = {}
            # X Processor
            if self.num_cols:
                self.X_encoder["num"] = KBinsDiscretizer(encode="ordinal")
                self.X_encoder["num"].fit(X[self.num_cols])
            if self.cat_cols:
                self.y_encoder["cat"] = OrdinalEncoder()
                self.X_encoder["cat"].fit(X[self.cat_cols])

            self.bin_y = y.dtype.kind in "fui"
            if self.bin_y:
                self.y_encoder = KBinsDiscretizer(encode="ordinal")
            else:
                self.y_encoder = OrdinalEncoder()
            self.y_encoder.fit(y)
        X_t = pd.concat(
            [
                encoder.transform(X[self.X_cols[key]])
                for key, encoder in self.X_encoder.items()
            ],
            axis=1,
        ).reindex(columns=self.X_cols["all"])
        y_t = self.y_encoder.transform(y)
        return X_t, y_t

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

    def postprocess(self, y):
        if self.y_encoder is not None:
            y = self.y_encoder.inverse_transform(y)
        return y

    def sample(self, X):
        dists = self.conditional_dist[
            tuple([tuple(X[parent]) for parent in self.X_cols["all"]])
        ]

        y = pchoice(p=dists)
        return y
