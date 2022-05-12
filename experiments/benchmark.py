import pickle
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from logging import getLogger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
)

from dpart.methods.utils.sklearn_encoder import SklearnEncoder
from dpart.methods.utils.bin_encoder import BinEncoder
from dpart.engines import Independent, PrivBayes, DPsynthpop


logger = getLogger("dpart")
logger.setLevel("WARN")


engines = {"PrivBayes": PrivBayes,
           "dp-synthpop": DPsynthpop,
           "Independent": Independent
           }
# Data Settings
LABEL = "income"

# Classifier Settings
CLF = LogisticRegression

EPSILONS = np.logspace(-2, 3, 10)
N_TRAIN = 5
N_GEN = 5
RESULTS_PATH = Path(__file__).parent / "data" / "benchmark_results.csv"


def get_data():
    train_df = pd.read_pickle("data/adult/tiny_adult.pkl.gz")
    test_df = pd.read_pickle("data/adult/tiny_adult_test.pkl.gz")
    with Path("data/adult/tiny_adult_bounds.pkl").open("rb") as fr:
        bounds = pickle.load(fr)
    return train_df, test_df, bounds


def evaluate(train, test):
    full_data = pd.concat([train, test], ignore_index=True)
    encoder = SklearnEncoder(mode="ordinal")
    t_data = encoder.fit_transform(full_data)

    t_train, t_test = (t_data.iloc[: train.shape[0]], t_data.iloc[train.shape[0]:])
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


def low_baseline(train, test):
    majority_label = train[LABEL].value_counts().idxmax()
    y_pred = [1] * test.shape[0]
    y_target = (test[LABEL] == majority_label).astype(int)

    scores = {
        "f1": f1_score(y_true=y_target, y_pred=y_pred),
        "accuracy": accuracy_score(y_true=y_target, y_pred=y_pred),
        "recall": recall_score(y_true=y_target, y_pred=y_pred),
        "precision": precision_score(y_true=y_target, y_pred=y_pred),
    }

    return scores


def similarity(real, synth, n_bins=100):
    full_df = pd.concat([real, synth], axis=0)
    scores = []
    for col, series in full_df.items():
        encoder = BinEncoder(n_bins=n_bins)
        t_series = encoder.fit_transform(series)
        t_real, t_synth = t_series.iloc[:real.shape[0]], t_series.iloc[real.shape[0]:]

        score = pd.DataFrame({"real": t_real.value_counts(normalize=True), "synth": t_synth.value_counts(normalize=True)}).fillna(0).min(axis=1).sum()
        scores.append(score)
    return np.mean(scores)


if __name__ == "__main__":
    results = []
    train_df, test_df, bounds = get_data(
    )

    low_results = low_baseline(train_df, test_df)
    low_results.update({"exp_idx": "low_baseline", "epsilon": None, "gen_idx": "source", "engine": None})

    source_results = evaluate(train_df, test_df)
    source_results.update({"exp_idx": "high_baseline", "epsilon": None, "gen_idx": "source", "engine": None})

    results += [source_results, low_results]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ebar = tqdm(list(engines.items()), desc="engine: ", leave=True)
        for engine_name, engine in ebar:
            ebar.set_description(f"engine : {engine_name}")
            pbar = tqdm(list(EPSILONS), desc="epsilon: ", leave=False)
            for epsilon in pbar:
                for exp_idx in tqdm(range(N_TRAIN), desc="train iteration: ", leave=False):
                    dpart_model = engine(
                        epsilon=epsilon,
                        bounds=bounds
                    )
                    dpart_model.fit(train_df)

                    pbar.set_description(f"epsilon: {dpart_model.epsilon:.3f}")

                    for gen_idx in tqdm(range(N_GEN), desc="gen iteration: ", leave=False):
                        exp_results = {
                            "engine": engine_name,
                            "exp_idx": exp_idx,
                            "epsilon": epsilon,
                            "gen_idx": gen_idx,
                        }
                        synth_df = dpart_model.sample(train_df.shape[0])
                        exp_scores = evaluate(synth_df, test_df)
                        exp_results.update(exp_scores)
                        exp_results["similarity"] = similarity(train_df, synth_df)

                        results.append(exp_results)

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_PATH, index=False)
