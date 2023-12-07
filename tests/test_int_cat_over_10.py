import numpy as np
import pandas as pd
from dpart.engines import Independent


def test_categorical_int_over_10():
    size = 13
    data = {"a": np.arange(size), "b": np.random.randint(5, size=size)}
    df = pd.DataFrame(data, dtype="category")

    independent = Independent(
        bounds={
            "a": {"categories": list(range(size))},
            "b": {"categories": list(range(5))},
        }
    )
    independent.fit(df)
    independent.sample(df.shape[0])
