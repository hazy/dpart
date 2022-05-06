from dpart.dpart import dpart


class Synthpop(dpart):
    def __init__(self,
                 epsilon: dict = None,
                 methods: dict = None,
                 visit_order: list = None,
                 bounds: dict = None):
        super().__init__(methods=methods)
