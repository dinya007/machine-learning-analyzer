import json
from json import JSONEncoder

import numpy as np

from common.entities.model_analysis import ModelAnalysis


class TrainResponse:

    def __init__(self, status, model_analysis):
        self.status = status
        self.modelAnalysis = model_analysis


class TrainResponseEncoder(JSONEncoder):
    def default(self, object):
        if isinstance(object, TrainResponse):
            return object.__dict__
        elif isinstance(object, ModelAnalysis):
            return object.__dict__
        elif isinstance(object, np.integer):
            return int(object)
        elif isinstance(object, np.floating):
            return float(object)
        elif isinstance(object, np.ndarray):
            return object.tolist()
        else:
            return json.JSONEncoder.default(self, object)
