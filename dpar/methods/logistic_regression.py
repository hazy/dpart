from diffprivlib.models import RandomForestClassifier as RFC
from dpar.methods.base import ClassifierSampler


class RandomForestClassifier(ClassifierSampler):
    clf_class = RFC
