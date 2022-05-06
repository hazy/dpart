import numpy as np
from itertools import combinations
from scipy.special import logsumexp


def mi_sensitivity(n_rows):
    r"""
    Computing Sensitivity for Mutual Information


    Parameters
    ----------
    n_rows: int
        Number of rows available with the DataFrame.
    Returns
    -------
    sensitivity: float
        Sensitivity score computed.
    See Also
    --------
    None
    Notes
    -----
    -
    Examples
    --------
    ```
    ```
    """
    a = (2 / n_rows) * np.log((n_rows + 1) / 2)
    b = (1 - 1 / n_rows) * np.log(1 + 2 / (n_rows - 1))
    return a + b


def entropy(pp):
    return -np.sum(pp * np.log2(pp))


def score_candidate(df, parents, child):
    y_pp = (df[parents].groupby(parents, observed=True).size() / df.shape[0]).to_numpy()
    x_pp = df[child].value_counts(normalize=True).to_numpy()
    xy_pp = (df[parents + [child]].groupby(parents + [child], observed=True).size() / df.shape[0]).to_numpy()

    return entropy(y_pp) + entropy(x_pp) - entropy(xy_pp)


def exponential_mechanism(input_vector, delta):
    dp_scores = np.array(input_vector) / (2 * delta)
    log_normalised = dp_scores - logsumexp(dp_scores)
    scores = np.exp(log_normalised)
    return scores / scores.sum()


def select_candidate(df, parents, n_parents=3, epsilon=None):
    n_parents = min(n_parents, len(parents))
    candidates = [(list(p), child) for p in combinations(parents, n_parents) for child in df.columns if child not in parents]
    scores = [score_candidate(df, parents=p, child=c) for p, c in candidates]

    if epsilon is not None:
        sensitivity = mi_sensitivity(df.shape[0])
        delta = sensitivity / epsilon
        dp_scores = exponential_mechanism(scores, delta=delta, )
        best_idx = np.random.choice(dp_scores.shape[0], p=dp_scores)
    else:
        best_idx = np.argmax(scores)

    return candidates[best_idx]
