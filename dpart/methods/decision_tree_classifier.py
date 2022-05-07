from sklearn.tree import DecisionTreeClassifier as DTC
from diffprivlib.models.forest import DecisionTreeClassifier as DPDTC

from dpart.methods.base import ClassifierSampler


class DecisionTreeClassifier(ClassifierSampler):
    dp_clf_class = DPDTC
    clf_class = DTC
