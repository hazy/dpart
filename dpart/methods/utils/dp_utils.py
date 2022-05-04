import numpy as np


def laplace_mechanism(
    counts: np.ndarray, sensitivity: float, epsilon: float,
) -> np.ndarray:
    r"""
    Apply laplace mechanism to provided counts

    Parameters
    ----------
    counts: np.ndarray
        Counts Array.
    sensitivity: float
        sensitivity to use when injecting noise.
    epsilon: float
        privacy budget to use.
    normalise: bool
        when set to True will normalise the counts along the last axis.

    Returns
    -------
    scale: np.ndarray
        noised - up count counts
    See Also
    --------
    None
    Notes
    -----
    -
    Examples
    --------
    ```
    from dpart.methods.utils.pd_utils import laplace_mechanism
    laplace_mechanism(
        counts=[[1,2], [0, 0], [3, 5]],
        sensitivity=0.2,
        epsilon=0.05
    )
    # array([[0.67240764, 0.32759236],
    #        [1.        , 0.        ],
    #        [0.46115546, 0.53884454]])

    ```
    """
    msg = f"ValueError: epsilon value is incorrect.\nDETAILS: expected strictly positive float got {epsilon} instead."
    assert epsilon > 0, msg

    counts = np.array(counts)
    noise = np.random.laplace(0, scale=sensitivity / epsilon, size=counts.shape)
    noised_counts = counts + noise

    return noised_counts
