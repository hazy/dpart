from sklearn.ensemble import RandomForestClassifier as RFC
from diffprivlib.models import RandomForestClassifier as DPRFC

from dpart.methods.base import ClassifierSampler


class RandomForestClassifier(ClassifierSampler):
    dp_clf_class = DPRFC
    clf_class = RFC
