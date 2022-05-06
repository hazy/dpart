from diffprivlib.models import LogisticRegression as DPLR
from sklearn.linear_model import LogisticRegression as LR
from dpart.methods.base import ClassifierSampler


class LogisticRegression(ClassifierSampler):
    dp_clf_class = DPLR
    dp_class = LR
