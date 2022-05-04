from diffprivlib.models.forest import DecisionTreeClassifier as DTC
from dpar.methods.base import ClassifierSampler


class DecisionTreeClassifier(ClassifierSampler):
    clf_class = DTC
