import os
import sys

from regression.xgboost.xgboost_regressor import XGBoost


class Jobs:
    jobs = {}

    def train_regressor(self, job_id, file):
        try:
            xg_boost = XGBoost()
            xg_boost.train(file)
            self.jobs[job_id] = xg_boost
            return {'status': 'SUCCESS'}
        except:
            e = sys.exc_info()[0]
            print("Errro %s" % e)
            return {'status': 'ERROR'}

    def predict(self, job_id, predict_data_file_path, result_file_path):
        try:
            predictions = self.jobs[job_id].predict(predict_data_file_path)
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
