import os
import pickle
import sys

from regression.entities.preparation import PreparationEncoder
from regression.xgboost.xgboost_regressor import XGBoost


class Jobs:

    def analyze(self, data_path, result_path):
        try:
            model = XGBoost()
            result = PreparationEncoder().encode(model.analyze(data_path))
            dirname = os.path.dirname(result_path)
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
            with open(result_path, 'w') as file:
                file.write(result)
            return {'status': 'ANALYZED'}
        except:
            e = sys.exc_info()[0]
            print("Errro %s" % e)
            return {'status': 'ERROR'}

    def train(self, data_path, model_path):
        try:
            model = XGBoost()
            model.train(data_path)
            os.makedirs(os.path.dirname(model_path))
            with open(model_path, 'wb') as file:
                pickle.dump(model, file)
            return {'status': 'SUCCESS'}
        except:
            e = sys.exc_info()[0]
            print("Errro %s" % e)
            return {'status': 'ERROR'}

    def predict(self, data_path, model_path, result_path):
        try:
            with open(model_path, 'rb') as file:
                model = pickle.load(file)
                predictions = model.predict(data_path)
                self.write_results(result_path, predictions)
                return {'status': 'SUCCESS'}
        except:
            e = sys.exc_info()[0]
            print("Errro %s" % e)
            return {'status': 'ERROR'}

    def write_results(self, result_file_path, results):
        directory = os.path.dirname(result_file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(result_file_path, 'w') as file:
            for line in results:
                file.write("{}\n".format(line))
