import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from logging import getLogger
from hazy_data import datasets
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
)

from dpar import DPAR
from dpar.methods import LogisticRegression, LinearRegression, RandomForestClassifier
from dpar.methods.utils.sklearn_encoder import SklearnEncoder


logger = getLogger("DPAR")
logger.setLevel("ERROR")


# Data Settings
DATASET = "adult"
LABEL = "income"
TEST_SIZE = 0.2

# Classifier Settings
CLF = DTC
SPLIT_SEED = 2021

EPSILONS = np.logspace(-3, 6, 10)
N_TRAIN = 5
N_GEN = 5
RESULTS_PATH = Path(__file__).parent / "results.csv"


def get_data(dataset, test_size=0.2, split_seed=None, label=None):
    df = datasets[dataset].df
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=split_seed, stratify=df[label]
    )
    return train_df, test_df


def evaluate(train, test):
    full_data = pd.concat([train, test], ignore_index=True)
    encoder = SklearnEncoder(mode="ordinal")
    t_data = encoder.fit_transform(full_data)

    t_train, t_test = (t_data.iloc[: train.shape[0]], t_data.iloc[train.shape[0] :])
    # fit model
    clf = CLF()
    clf.fit(t_train.drop(LABEL, axis=1), t_train[LABEL])
    X_test, y_target = t_test.drop(LABEL, axis=1), t_test[LABEL]
    y_pred, y_proba = clf.predict(X_test), clf.predict_proba(X_test)
    scores = {
        "f1": f1_score(y_true=y_target, y_pred=y_pred),
        "accuracy": accuracy_score(y_true=y_target, y_pred=y_pred),
        "recall": recall_score(y_true=y_target, y_pred=y_pred),
        "precision": precision_score(y_true=y_target, y_pred=y_pred),
        "auc": roc_auc_score(y_true=y_target, y_score=y_proba[:, -1]),
    }

    return scores


if __name__ == "__main__":
    results = []
    train_df, test_df = get_data(
        dataset=DATASET, test_size=TEST_SIZE, label=LABEL, split_seed=SPLIT_SEED
    )

    source_results = evaluate(train_df, test_df)
    source_results.update({"exp_idx": "source", "epsilon": None, "gen_idx": "source"})
    results.append(source_results)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for epsilon in tqdm(list(EPSILONS), desc="epsilon: ", leave=True):
            for exp_idx in tqdm(range(N_TRAIN), desc="train iteration: ", leave=False):
                dpar_model = DPAR(
                    methods={
                        col: RandomForestClassifier()
                        if series.dtype.kind in "OSb"
                        else LinearRegression()
                        for col, series in train_df.items()
                    },
                    epsilon=epsilon,
                )
                dpar_model.fit(train_df)

                for gen_idx in tqdm(range(N_GEN), desc="Gen iteration: ", leave=False):
                    exp_results = {
                        "exp_idx": exp_idx,
                        "epsilon": epsilon,
                        "gen_idx": gen_idx,
                    }
                    synth_df = dpar_model.sample(train_df.shape[0])
                    exp_scores = evaluate(synth_df, test_df)
                    exp_results.update(exp_scores)

                    results.append(exp_results)

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_PATH, index=False)
