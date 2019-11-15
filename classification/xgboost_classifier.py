from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from xgboost import XGBClassifier

from common.ModelTemplate import ModelTemplate


class XGBoostClassifier:

    def __init__(self):
        self.model_template = ModelTemplate()

    def train(self, train_file):
        return self.model_template.train(train_file, self.get_model)

    def predict(self, file_path):
        return self.model_template.predict(file_path)

    def get_model(self, X_train, y_train):
        xg_boost_classifier = XGBClassifier(n_estimators=100, random_state=0, objective='reg:squarederror')

        # self.optimizeHyperParameters(X_train, y_train)
        return xg_boost_classifier

    def optimizeHyperParameters(self, X_train, y_train):
        random_forest_regressor = RandomForestRegressor()

        random_forest_parameters = [
            {'n_estimators': [100, 200]},
        ]
        grid_search = GridSearchCV(estimator=random_forest_regressor, param_grid=random_forest_parameters,
                                   scoring='neg_mean_squared_error', cv=10, n_jobs=-1)
        grid_search = grid_search.fit(X_train, y_train)
        print("Best score Random Forest: {}".format(grid_search.best_score_))
        print("Best params Random Forest: {}".format(grid_search.best_params_))
        svr_regressor = SVR(gamma='auto')
        svr_parameters = [
            {'degree': [1, 2, 3]}
        ]
        svr_grid_search = GridSearchCV(estimator=svr_regressor, param_grid=svr_parameters,
                                       scoring='neg_mean_squared_error', cv=10, n_jobs=-1)
        svr_grid_search = svr_grid_search.fit(X_train, y_train)
        print("Best score SVR: {}".format(svr_grid_search.best_score_))
        print("Best params SVR: {}".format(svr_grid_search.best_params_))


# classifier = XGBoostClassifier()
# classifier.train('train.csv')
# id_column_name, target_column_name, ids, predictions = classifier.predict('test.csv')
#
# with open('result.csv', 'w') as file:
#     file.write("{},{}\n".format(id_column_name, target_column_name))
#     for i in range(len(predictions)):
#         file.write("{},{}\n".format(ids[i], predictions[i]))
