import numpy as np
import pandas as pd

from dpart.utils.kahn import kahn_sort
from dpart.methods.utils.bin_encoder import BinEncoder
from dpart.utils.dependencies.selection import select_candidate


class DependencyManager():
    def __init__(self, epsilon: float = None, visit_order=None, prediction_matrix=None, n_bins=20, n_parents=2):
        self.encoders = None
        self.n_parents = n_parents
        self.n_bins = n_bins
        if prediction_matrix != "infer":
            epsilon = 0
        self.epsilon = epsilon
        self.visit_order = visit_order
        self.prediction_matrix = prediction_matrix

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.encoders is None:
            self.encoders = {}

        t_df = {}
        for col, series in df.items():
            if col not in self.encoders:
                self.encoders[col] = BinEncoder(n_bins=self.n_bins)
                self.encoders[col].fit(series)

            t_df[col] = self.encoders[col].transform(series)

        return pd.DataFrame(t_df)

    def infer_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        # Nodes
        root = np.random.choice(df.columns.tolist())
        visit_order = [root]
        prediction_matrix = {root: []}

        if self.epsilon is not None:
            eps = self.epsilon / (df.shape[1] - 1)
        else:
            eps = None
        # construct bayesian network
        for i in range(df.shape[1] - 1):
            selected_parents, selected_child = select_candidate(df=df, parents=visit_order, n_parents=self.n_parents, epsilon=eps)
            visit_order.append(selected_child)
            prediction_matrix[selected_child] = list(selected_parents)

        return visit_order, prediction_matrix

    def fit(self, df: pd.DataFrame):
        if self.prediction_matrix == "infer" and self.n_parents != 0:
            self.visit_order, self.prediction_matrix = self.infer_matrix(df)
        else:
            if self.prediction_matrix is not None:
                self.visit_order = list(kahn_sort(self.prediction_matrix))
            else:
                if self.visit_order is None:
                    self.visit_order = list(df.columns)

                self.prediction_matrix = {
                    col: self.visit_order[:idx]
                    for idx, col in enumerate(self.visit_order)
                }

                if self.n_parents == 0:
                    self.prediction_matrix = {
                        col: []
                        for idx, col in enumerate(self.visit_order)
                    }
                elif self.n_parents is not None:
                    self.prediction_matrix = {
                        col: deps[-self.n_parents:]
                        for col, deps in self.prediction_matrix.items()
                    }
