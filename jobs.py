import os
import sys

from regression.xgboost.xgboost_regressor import XGBoost
import pickle

class Jobs:

    def train_regressor(self, job_id, file):
        try:
            model = XGBoost()
            model.train(file)

            with open('/Users/denis/IdeaProjects/machine-learning-analyzer/model/xgboost.pickle', 'wb') as file:
                pickle.dump(model, file)

            return {'status': 'SUCCESS'}
        except:
            e = sys.exc_info()[0]
            print("Errro %s" % e)
            return {'status': 'ERROR'}

    def predict(self, job_id, predict_data_file_path, result_file_path):
        try:
            with open('/Users/denis/IdeaProjects/machine-learning-analyzer/model/xgboost.pickle', 'rb') as handle:
                model = pickle.load(handle)
                predictions = model.predict(predict_data_file_path)
                self.write_results(result_file_path, predictions)
                return {'status': 'SUCCESS'}
        except:
            e = sys.exc_info()[0]
            print("Errro %s" % e)
            return {'status': 'ERROR'}

    def getRegressor(self):
        return XGBoost()

    def write_results(self, result_file_path, results):
        directory = os.path.dirname(result_file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(result_file_path, 'w') as file:
            for line in results:
                file.write("{}\n".format(line))
