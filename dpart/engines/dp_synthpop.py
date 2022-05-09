from dpart.dpart import dpart
from dpart.methods import LogisticRegression, LinearRegression


class DPsynthpop(dpart):
    default_numerical = LinearRegression
    default_categorical = LogisticRegression

    def __init__(self,
                 epsilon: dict = None,
                 methods: dict = None,
                 visit_order: list = None,
                 bounds: dict = None):
        super().__init__(methods=methods, epsilon=epsilon, visit_order=visit_order, bounds=bounds)
