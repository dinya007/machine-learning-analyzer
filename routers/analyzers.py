import os
import sys

from feature_analyze.feature_analyzer import analyze
from common.entities.feature_analysis import PreparationEncoder


class Analyzers:

    def analyze(self, data_path, result_path):
        try:
            result = PreparationEncoder().encode(analyze(data_path))
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
