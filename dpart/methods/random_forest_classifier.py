from diffprivlib.models import RandomForestClassifier as RFC

from dpart.methods.base import ClassifierSampler


class RandomForestClassifier(ClassifierSampler):
    clf_class = RFC
