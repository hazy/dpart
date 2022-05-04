from diffprivlib.models import LogisticRegression as DPLR
from dpart.methods.base import ClassifierSampler


class LogisticRegression(ClassifierSampler):
    clf_class = DPLR
