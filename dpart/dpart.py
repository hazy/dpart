import warnings
import numpy as np
import pandas as pd
from logging import getLogger
from typing import Union, Dict
from collections import defaultdict
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from diffprivlib.utils import PrivacyLeakWarning

from dpart.utils.dependencies import DependencyManager
from dpart.methods import ProbabilityTensor


logger = getLogger("dpart")
logger.setLevel("ERROR")


class dpart:
    default_numerical = ProbabilityTensor
    default_categorical = ProbabilityTensor

    def __init__(
        self,
        # methods
        methods: dict = None,
        # privacy settings
        epsilon: Union[dict, float] = None,
        bounds: dict = None,
        # dependencies
        dependency_manager=None,
        visit_order: list = None,
        prediction_matrix: dict = None,
        n_parents=2
    ):

        # Privact budget
        if epsilon is not None:
            if not isinstance(epsilon, dict):
                if prediction_matrix == "infer":
                    epsilon = {"dependency": epsilon / 2, "methods": epsilon / 2}
                else:
                    epsilon = {"dependency": 0, "methods": epsilon}
        else:
            epsilon = {
                "dependency": None,
                "methods": defaultdict(lambda: None)
            }
        self._epsilon = epsilon
        self.dep_manager = DependencyManager(
            epsilon=self._epsilon.get("dependency", None),
            visit_order=visit_order,
            prediction_matrix=prediction_matrix,
            n_parents=n_parents
        )

        # method dict
        if methods is None:
            methods = {}
        self.methods = methods
        self.encoders = None

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
        self.encoders = {}
        df = df.copy()
        for col, series in df.items():
            if series.dtype.kind in "OSb":
                t_dtype = "category"
                if col not in self.bounds:
                    if self._epsilon.get("methods", None) is not None:
                        warnings.warn(f"List of categories not sepecified for column '{col}'", PrivacyLeakWarning)
                    self.bounds[col] = {"categories": sorted(list(series.unique()), key=str)}
                self.encoders[col] = OrdinalEncoder(categories=[self.bounds[col]["categories"]])
            else:
                t_dtype = "float"
                if col not in self.bounds:
                    if self._epsilon.get("methods", None) is not None:
                        PrivacyLeakWarning(f"upper and lower bounds not specified for column '{col}'")
                    self.bounds[col] = {"min": series.min(), "max": series.max()}
                self.encoders[col] = MinMaxScaler(feature_range=[self.bounds[col]["min"], self.bounds[col]["max"]])
            df[col] = pd.Series(self.encoders[col].fit_transform(df[[col]]).squeeze(), name=col, index=df.index, dtype=t_dtype)

        return df

    def default_method(self, dtype):
        if dtype.kind in "OSb":
            return self.default_categorical()
        return self.default_numerical()

    def fit(self, df: pd.DataFrame):
        # dependency manager
        t_df = self.dep_manager.preprocess(df)
        self.dep_manager.fit(t_df)

        # Capture dtypes
        self.dtypes = df.dtypes
        self.columns = df.columns

        if not isinstance(self._epsilon["methods"], dict):
            total_budget = float(self._epsilon["methods"])
            self._epsilon["methods"] = defaultdict(lambda: total_budget / df.shape[1])

        # reorder and introduce initial columns
        self.root = self.root_column(df)
        t_df = self.normalise(df)
        t_df.insert(0, column=self.root, value=0)

        # build methods
        for idx, target in enumerate(self.dep_manager.visit_order):
            X_columns = [self.root] + self.dep_manager.prediction_matrix.get(target, [])
            X = t_df[X_columns]
            y = t_df[target]

            if target not in self.methods:
                def_method = self.default_method(self.dtypes[target])
                warnings.warn(
                    f"target {target} has no specified method will use default {def_method.__class__.__name__}"
                )
                self.methods[target] = def_method

            if self._epsilon["methods"][target] is not None:
                self.methods[target].set_epsilon(self._epsilon["methods"][target])

            logger.info(
                f"Fit target: {target} | sampler used: {self.methods[target].__class__.__name__}"
            )

            t_X, t_y = self.methods[target].preprocess(X=X, y=y)
            self.methods[target].fit(X=t_X, y=t_y)

    def denormalise(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in df.columns:
            df[col] = self.encoders[col].inverse_transform(df[[col]]).squeeze()

            if self.dtypes[col].kind in "ui":
                df[col] = df[col].round().astype(int).astype(self.dtypes[col])
            else:
                df[col] = df[col].astype(self.dtypes[col])
        return df

    def sample(self, n_records: int) -> pd.DataFrame:
        df = pd.DataFrame({self.root: 0}, index=np.arange(n_records))
        for target in self.dep_manager.visit_order:
            X_columns = [self.root] + self.dep_manager.prediction_matrix.get(target, [])
            logger.info(f"Sample target {target}")
            logger.debug(f"Sample target {target} - preprocess feature matrix")
            t_X = self.methods[target].preprocess_X(df[X_columns])
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
        budgets = [method.epsilon for _, method in self.methods.items()] + [self.dep_manager.epsilon]
        if pd.isnull(budgets).any():
            return None
        else:
            return sum(budgets)
