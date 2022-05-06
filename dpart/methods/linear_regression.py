from diffprivlib.models import LinearRegression as DPLR
from sklearn.linear_model import LinearRegression as LR

from dpart.methods.base import RegressorSampler


class LinearRegression(RegressorSampler):
    dp_reg_class = DPLR
    reg_class = LR
