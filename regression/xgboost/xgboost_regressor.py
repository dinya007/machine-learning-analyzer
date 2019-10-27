import category_encoders as ce
import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_numeric_dtype
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from statsmodels.api import OLS
from xgboost import XGBRegressor


class XGBoost:

    def __init__(self):
        self.regressor = None
        self.numeric_columns = None
        self.string_columns = None
        self.numeric_imputer = None
        self.string_imputer = None
        self.one_hot_encoder = None
        self.standard_scaler = None
        self.columns_to_delete = []

    def train(self, train_file):
        X_train, y_train = self.import_dataset(train_file)

        # X_train = self.backward_elimination(x_train=X_train, y_train=y_train, siginificance_level=0.05)

        print("Training model...")
        random_forest_regressor = XGBRegressor(n_estimators=300, random_state=0, objective='reg:squarederror')

        # self.optimizeHyperParameters(X_train, y_train)

        random_forest_regressor.fit(X_train, y_train)
        print("Model trained...")

        # accuracies = cross_val_score(random_forest_regressor, X_train, y_train, cv=10)
        # print("Mean: {}".format(accuracies.mean()))
        # print("Std: {}".format(accuracies.std()))
        self.regressor = random_forest_regressor

    def is_ready(self):
        return self.regressor is not None

    def predict(self, file_path):
        dataset = pd.read_csv(file_path)

        rows = dataset.iloc[:, :].values
        rows[:, self.numeric_columns] = self.numeric_imputer.transform(rows[:, self.numeric_columns])
        rows[:, self.string_columns] = self.string_imputer.transform(rows[:, self.string_columns])
        rows = self.one_hot_encoder.transform(rows)
        rows = self.standard_scaler.transform(rows)
        for i in self.columns_to_delete:
            rows = np.delete(rows, i, 1)
        return self.regressor.predict(rows)

    def import_dataset(self, train_file):
        dataset = pd.read_csv(train_file)
        columns_number = dataset.shape[1]
        numeric_columns, string_columns = self.get_column_types(dataset)

        X_train = dataset.iloc[:, :columns_number - 1].values
        y_train = dataset.iloc[:, columns_number - 1:].values

        numeric_imputer = SimpleImputer(strategy='mean')
        X_train[:, numeric_columns] = numeric_imputer.fit_transform(X_train[:, numeric_columns])

        string_imputer = SimpleImputer(strategy='most_frequent')
        X_train[:, string_columns] = string_imputer.fit_transform(X_train[:, string_columns])

        one_hot_encoder = ce.OneHotEncoder(drop_invariant=True, use_cat_names=True)
        X_train = one_hot_encoder.fit_transform(X_train)

        standard_scaler = StandardScaler()
        X_train = standard_scaler.fit_transform(X_train)

        y_train = y_train.flatten()

        self.numeric_columns = numeric_columns
        self.string_columns = string_columns
        self.numeric_imputer = numeric_imputer
        self.string_imputer = string_imputer
        self.one_hot_encoder = one_hot_encoder
        self.standard_scaler = standard_scaler

        return X_train, y_train

    def get_column_types(self, dataset):
        numeric_columns = []
        string_columns = []
        columns_number = dataset.shape[1]
        for i in range(0, columns_number - 1):
            if is_numeric_dtype(dataset.iloc[:, i].values):
                numeric_columns.append(i)
            else:
                string_columns.append(i)
        return numeric_columns, string_columns

    def backward_elimination(self, x_train, y_train, siginificance_level):
        x_train = np.append(arr=np.ones((x_train.shape[0], 1)).astype(int), axis=1, values=x_train)
        numVars = x_train.shape[1]
        for i in range(0, numVars):
            regressor_OLS = OLS(y_train, x_train).fit()
            maxVar = max(regressor_OLS.pvalues).astype(float)
            if maxVar > siginificance_level:
                for j in range(0, numVars - i):
                    if regressor_OLS.pvalues[j].astype(float) == maxVar:
                        x_train = np.delete(x_train, j, 1)
                        self.columns_to_delete.append(j)
        print(regressor_OLS.summary())
        return x_train

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

# regressor = XGBoost()
# regressor.train('train.csv')
#
# X_test = pd.read_csv('test.csv')
#
# results = ['Id,SalePrice']
#
# predicts = regressor.predict(X_test)
#
# for i in range(1461, 2920):
#     results.append("{},{}".format(i, predicts[i - 1461]))
#
# with open('results.txt', 'w') as f:
#     for line in results:
#         f.write("{}\n".format(line))

# pyplot.bar(range(len(regressor.regressor.feature_importances_)), regressor.regressor.feature_importances_)
# pyplot.show()
