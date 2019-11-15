import numpy as np
import pandas as pd

from common.entities.feature_analysis import FeatureAnalysis, NumericalFeature, CategoricalFeature


def analyze(train_file):
    dataset = pd.read_csv(train_file)
    types = dataset.dtypes
    counts = dataset.apply(lambda x: x.count())
    nulls = dataset.apply(lambda x: x.isnull().sum())
    columns_number = dataset.shape[1]
    preparation = FeatureAnalysis()
    target_feature_name = dataset.columns[columns_number - 1]

    for i in range(len(types)):
        feature_name = types.keys()[i]
        if types[i] in ('int64', 'float64'):
            preparation.numerical.count += 1
            preparation.numerical.features.append(
                NumericalFeature(feature_name,
                                 nulls[feature_name],
                                 np.count_nonzero(dataset[feature_name] == 0),
                                 dataset[feature_name].min(),
                                 dataset[feature_name].max(),
                                 dataset[feature_name].mean(),
                                 dataset[feature_name].median(),
                                 counts[feature_name],
                                 dataset[feature_name].corr(dataset[target_feature_name]),
                                 dataset[feature_name].values))
        else:
            (unique_name, unique_counts) = np.unique(dataset[feature_name].astype(str).values, return_counts=True)
            most_frequent_index = np.argmax(unique_counts)
            preparation.categorical.count += 1
            values = []
            for i in range(len(unique_name)):
                values.append({'name': unique_name[i], 'count': unique_counts[i]})
            preparation.categorical.features.append(
                CategoricalFeature(feature_name,
                                   nulls[feature_name],
                                   unique_name[most_frequent_index],
                                   unique_counts[most_frequent_index],
                                   unique_name.shape[0],
                                   counts[feature_name],
                                   values))
    return preparation
