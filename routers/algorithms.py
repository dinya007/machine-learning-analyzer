import json
import os
import pickle
import sys

from common.entities.train_response import TrainResponse


class Algorithms:

    def __init__(self, default_model_class):
        self.default_model_class = default_model_class

    def train(self, data_path, model_path):
        try:
            model = self.default_model_class()
            model_analysis = model.train(data_path)
            os.makedirs(os.path.dirname(model_path))
            with open(model_path, 'wb') as file:
                pickle.dump(model, file)
            return TrainResponse('SUCCESS', model_analysis)
        except:
            e = sys.exc_info()[0]
            print("Errro %s" % e)
            return {'status': 'ERROR'}

    def predict(self, data_path, model_path, result_path):
        try:
            with open(model_path, 'rb') as file:
                model = pickle.load(file)
                id_column_name, target_column_name, ids, predictions = model.predict(data_path)
                self.write_results(result_path, id_column_name, target_column_name, ids, predictions)
                return {'status': 'SUCCESS'}
        except:
            e = sys.exc_info()[0]
            print("Errro %s" % e)
            return {'status': 'ERROR'}

    def write_results(self, result_file_path, id_column_name, target_column_name, ids, predictions):
        directory = os.path.dirname(result_file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(result_file_path, 'w') as file:
            file.write("{},{}\n".format(id_column_name, target_column_name))
            for i in range(len(predictions)):
                file.write("{},{}\n".format(ids[i], predictions[i]))

    def write_model_analysis(self, model_analysis, model_analysis_path):
        directory = os.path.dirname(model_analysis_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(model_analysis_path, 'w') as file:
            file.write("{}".format(json.dumps(model_analysis)))
