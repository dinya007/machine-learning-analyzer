import category_encoders as ce
import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_numeric_dtype
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from statsmodels.api import OLS

from common.entities.model_analysis import ModelAnalysis


class ModelTemplate:

    def __init__(self):
        self.model = None
        self.id_column_name = None
        self.target_column_name = None
        self.numeric_columns = None
        self.string_columns = None
        self.numeric_imputer = None
        self.string_imputer = None
        self.one_hot_encoder = None
        self.standard_scaler = None
        self.columns_to_delete = []

    def train(self, train_file, get_optimized_model):
        X_train, y_train = self.import_dataset(train_file)

        model = get_optimized_model(X_train, y_train)

        print("Training model...")
        model.fit(X_train, y_train)
        print("Model trained...")

        print("Estimate model...")
        accuracies = cross_val_score(model, X_train, y_train, cv=10, n_jobs=-1)
        print("Mean: {}".format(accuracies.mean()))
        print("Std: {}".format(accuracies.std()))
        self.model = model
        return ModelAnalysis(accuracies.mean(), accuracies.std())

    def predict(self, file_path):
        print("Predicting...")

        dataset = pd.read_csv(file_path)

        rows = dataset.iloc[:, :].values
        rows[:, self.numeric_columns] = self.numeric_imputer.transform(rows[:, self.numeric_columns])
        rows[:, self.string_columns] = self.string_imputer.transform(rows[:, self.string_columns])
        rows = self.one_hot_encoder.transform(rows)
        rows = self.standard_scaler.transform(rows)
        for i in self.columns_to_delete:
            rows = np.delete(rows, i, 1)
        result = self.model.predict(rows)
        print("Predicted...")
        return self.id_column_name, self.target_column_name, dataset.iloc[:, 0].values, result

    def import_dataset(self, train_file):
        dataset = pd.read_csv(train_file)
        columns_number = dataset.shape[1]
        self.target_column_name = dataset.columns[columns_number - 1]
        self.id_column_name = dataset.columns[0]

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
