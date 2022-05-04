from dpart.methods.base.sampler import Sampler


class CategorySampler(Sampler):
    category_support: bool = True
    numerical_support: bool = False
