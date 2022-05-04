from diffprivlib.models.forest import DecisionTreeClassifier as DTC
from dpart.methods.base import ClassifierSampler


class DecisionTreeClassifier(ClassifierSampler):
    clf_class = DTC
