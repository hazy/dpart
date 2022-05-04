from diffprivlib.models import LinearRegression as DPLR

from dpar.methods.base import RegressorSampler


class LinearRegression(RegressorSampler):
    reg_class = DPLR
