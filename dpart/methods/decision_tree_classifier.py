from diffprivlib.models.forest import DecisionTreeClassifier as DPDTC
from sklearn.tree import DecisionTreeClassifier as DTC
from dpart.methods.base import ClassifierSampler


class DecisionTreeClassifier(ClassifierSampler):
    dp_clf_class = DPDTC
    clf_class = DTC
