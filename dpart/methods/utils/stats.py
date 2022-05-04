import numpy as np


def normalise_proba(dist, conditional=False, _clip=False):
    r"""
    Normalises input probability vector to ensure values are positive and sum up to 1
    Parameters
    ----------
    dist : array_like
        Input data. Probability vector to normalise
    conditional: bool
        When set to True, treats the input distribution as a conditional distribution and normalises the distribution along the last axis.
        Othewise normalises along all axis.
    _clip : bool
        When set to True, clips the `dist` values so they are non-negative.
    conditional

    Returns
    -------
    p_dist : array_like
        Normalised probability distribution.
    See Also
    --------
    None
    Notes
    -----

    Examples
    --------
    ```
    from dpart.methods.utils.stats import normalise_proba
    normalise_proba([.15, .49, .5])
    # array([0.13157895, 0.42982456, 0.43859649])

    # fails
    normalise_proba([.15, .49, -.5])
    # AssertionError: ValueError: dist values contains negative values
    # DETAILS: [-0.5] are negative.

    # succeeds
    normalise_proba([.15, .49, -.5], _clip=True)
    # array([0.234375, 0.765625, 0.      ])
    ```
    """
    dist = np.array(dist).astype(float)

    msg = "ValueError: dist values contains NaN values"
    assert not np.isnan(dist).any(), msg

    if _clip:
        dist = np.clip(dist, a_min=0, a_max=None)

    msg = (
        "ValueError: dist values contains negative values"
        "\n"
        f"DETAILS: {dist[dist < 0]} are negative."
    )
    assert (dist >= 0).all(), msg

    if conditional:
        normaliser = np.expand_dims(np.sum(dist, axis=-1), axis=-1)
        null_map = normaliser == 0
        normaliser += null_map * 10 ** (-6)
        dist /= normaliser
        dist = (1 - null_map) * dist + null_map * 1 / dist.shape[-1]

    else:
        _sum = dist.sum()
        if _sum == 0:
            dist[:] = 1 / dist.size
        else:
            dist /= _sum

    return dist


def pchoice(p, values=None):
    r"""
    Samples from a value for a list of provided values using a vectorized probability matrix. (parallel choice)
    Parameters
    ----------
    p : array_like
        two dimensional numpy array [n_samples, n_values].
    values : list
        list of values to sample from.
    Returns
    -------
        _choice: np.Array with one value for each row within the probability matrix provided.
    See Also
    --------
    None.
    Notes
    -----
    Supports vectorised inputs.
    Examples
    --------
    ```
    import numpy as np
    from ml_core.utils.stats import pchoice
    np.random.seed(2019)
    p = [[0.3, 0.3, 0.4], [0.8, 0.1, 0.1], [0.05, 0.8, 0.15]]
    values = ['dog', 'cat', 'bird']
    pchoice(p, values=values)
    # array(['bird', 'dog', 'cat'], dtype='<U4')
    """

    p = np.array(p)

    cum_probs = np.cumsum(p, axis=1)
    raw_sample = np.random.uniform(0, 1, size=p.shape[0])

    # Parallised version of np.discretize that allows to choose a different set of bins for each row.
    _choice = (raw_sample[:, np.newaxis] < cum_probs).argmax(axis=1)

    if values is not None:
        values = np.array(values)
        _choice = values[_choice]

    return _choice
