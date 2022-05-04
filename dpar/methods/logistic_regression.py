from diffprivlib.models import LogisticRegression as DPLR
from dpar.methods.base import ClassifierSampler


class LogisticRegression(ClassifierSampler):
    clf_class = DPLR
