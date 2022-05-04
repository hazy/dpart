from dpart.methods.base.sampler import Sampler


class NumericalSampler(Sampler):
    category_support: bool = False
    numerical_support: bool = True
