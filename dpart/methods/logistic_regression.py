from sklearn.linear_model import LogisticRegression as LR
from diffprivlib.models import LogisticRegression as DPLR

from dpart.methods.base import ClassifierSampler


class LogisticRegression(ClassifierSampler):
    dp_clf_class = DPLR
    clf_class = LR
