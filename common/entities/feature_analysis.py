import json
from json import JSONEncoder

import numpy as np


class FeatureAnalysis:

    def __init__(self):
        self.categorical = Analyze()
        self.numerical = Analyze()


class Analyze:

    def __init__(self):
        self.count = 0
        self.features = []


class NumericalFeature:

    def __init__(self, name, missing, zeros, min, max, mean, median, count, correlation, values):
        self.name = name
        self.missing = missing
        self.zeros = zeros
        self.min = min
        self.max = max
        self.mean = mean
        self.median = median
        self.count = count
        self.correlation = correlation
        self.values = values


class CategoricalFeature:

    def __init__(self, name, missing, top, frequency_top, unique, count, values):
        self.name = name
        self.missing = missing
        self.top = top
        self.frequencyTop = frequency_top
        self.unique = unique
        self.count = count
        self.values = values


class PreparationEncoder(JSONEncoder):
    def default(self, object):
        if isinstance(object, FeatureAnalysis):
            return object.__dict__
        elif isinstance(object, Analyze):
            return object.__dict__
        elif isinstance(object, CategoricalFeature):
            return object.__dict__
        elif isinstance(object, NumericalFeature):
            return object.__dict__
        elif isinstance(object, np.integer):
            return int(object)
        elif isinstance(object, np.floating):
            return float(object)
        elif isinstance(object, np.ndarray):
            return object.tolist()
        else:
            return json.JSONEncoder.default(self, object)