import numpy as np
import pandas as pd
from logging import getLogger
from dpar.methods import ProbababilityTensor

logger = getLogger("DPAR")


class DPAR:
    DEFAULT_METHOD = ProbababilityTensor

    def __init__(
        self,
        visit_order: list = None,
        methods: dict = None,
        bounds: dict = None,
        epsilon: float = 1.0,
    ):

        # Privact budget
        self._epsilon = epsilon

        # visit order
        self.visit_order = visit_order

        # method dict
        if methods is None:
            methods = {}
        self.methods = methods

        # bound dict
        if bounds is None:
            bounds = {}
        self.bounds = bounds
        self.dtypes = None
        self.root = None
        self.columns = None

    def root_column(self, df: pd.DataFrame) -> str:
        root_col = "__ROOT__"
        idx = 0
        while root_col in df.columns:
            root_col = f"__ROOT_{idx}__"
            idx += 1
        return root_col

    def normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, (lower, upper) in self.bounds.items():
            if upper == lower:
                df[col] = 0
            else:
                df[col] = (df[col] - lower) / (upper - lower)

        return df

    def fit(self, df: pd.DataFrame):
        # Capture dtypes
        self.dtypes = df.dtypes
        self.columns = df.columns
        # extract visit order
        if self.visit_order is None:
            logger.info("extract visit order")
            self.visit_order = list(df.columns)
            logger.debug(f"extracted visit order: {self.visit_order}")

        # extract_bounds
        for column in self.visit_order:
            if df[column].dtype.kind in "Mmfui":
                if column not in self.bounds:
                    logger.warning(f"Bounds not provided for column {column}")
                    self.bounds[column] = (df[column].min(), df[column].max())
                    logger.debug(
                        f"Extracted bounds for {column}: {self.bounds[column]}"
                    )

        # reorder and introduce initial columns
        self.root = self.root_column(df)
        t_df = self.normalise(df).reindex(columns=self.visit_order)
        t_df.insert(0, column=self.root, value=0)

        # build methods
        for idx, target in enumerate(self.visit_order):
            X = t_df[t_df.columns[: idx + 1]]
            y = t_df[target]

            if target not in self.methods:
                logger.warning(
                    f"target {target} has no specified method will use default {self.DEFAULT_METHOD.__name__}"
                )
                self.methods[target] = self.DEFAULT_METHOD(
                    epsilon=self._epsilon / len(self.visit_order)
                )

            logger.info(
                f"Fit target: {target} | sampler used: {self.methods[target].__class__.__name__}"
            )

            t_X, t_y = self.methods[target].preprocess(X=X, y=y)
            self.methods[target].fit(X=t_X, y=t_y)

    def denormalise(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in df.columns:
            if col in self.bounds:
                lower, upper = self.bounds[col]
                df[col] = (df[col] * (upper - lower)) + lower

            if self.dtypes[col].kind in "ui":
                df[col] = df[col].round().astype(self.dtypes[col])
            else:
                df[col] = df[col].astype(self.dtypes[col])
        return df

    def sample(self, n_records: int) -> pd.DataFrame:
        df = pd.DataFrame({self.root: 0}, index=np.arange(n_records))
        for target in self.visit_order:
            logger.info(f"Sample target {target}")
            logger.debug(f"Sample target {target} - preprocess feature matrix")
            t_X = self.methods[target].preprocess_X(df)
            logger.debug(f"Sample target {target} - Sample values")
            t_y = self.methods[target].sample(X=t_X)
            logger.debug(f"Sample target {target} - post process sampled values")
            y = self.methods[target].postprocess_y(y=t_y)
            logger.debug(f"Sample target {target} - Update feature matrix")
            df.insert(loc=df.shape[1], column=target, value=y)

        logger.info("denormalise sampled data")
        i_df = self.denormalise(df=df.drop(self.root, axis=1)).reindex(
            columns=self.columns
        )
        return i_df

    @property
    def epsilon(self):
        budgets = [method.epsilon for _, method in self.methods.items()]

        if pd.isnull(budgets).any():
            return None
        else:
            return sum(budgets)
