from typing import Dict, Tuple
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib
import tensorflow.compat.v2.feature_column as fcv2
import tensorflow as tf
from tensorflow import feature_column as fc


# Ref: https://www.tensorflow.org/tutorials/estimator/linear?hl=ja
TRAINSET_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
EVALSET_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"
CATEGORICAL_COLUMNS = [
    "sex",
    "n_siblings_spouses",
    "parch",
    "class",
    "deck",
    "embark_town",
    "alone",
]
NUMERIC_COLUMNS = ["age", "fare"]


def load_dataset(URL: str, target: str = "survived") -> Tuple[pd.DataFrame, pd.Series]:
    df_X = pd.read_csv(URL)
    y = df_X.pop(target)
    return df_X, y


def drop_missing_values(df):
    # TODO: missing valueを含むデータを用意して検証
    for col in df.columns:
        if len(df.loc[df[col].isnull() == True]) != 0:
            if df[col].dtype == "float64" or df[col].dtype == "int64":
                df.loc[df[col].isnull() == True, col] = df[col].mean()
            else:
                df.loc[df[col].isnull() == True, col] = df[col].mode()[0]


def create_features(df: pd.DataFrame, feature_types: Dict[str, list]) -> list:
    feature_cols = []
    for feat_type, cols in feature_types.items():
        if feat_type == "categorical":
            for col in cols:
                vocabulary = df[col].unique()
                categ_feat = fc.categorical_column_with_vocabulary_list(col, vocabulary)
                feature_cols.append(categ_feat)
        if feat_type == "numeric":
            for col in cols:
                feature_cols.append(fc.numeric_column(col, dtype=tf.float32))
    return feature_cols


def make_input_fn(df_data, df_label, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(df_data), df_label))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds

    return input_function


def train(train_input_fn, feature_columns):
    linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
    return linear_est.train(train_input_fn)


def predict(model: tf.estimator.LinearClassifier, eval_input_fn):
    return model.evaluate(eval_input_fn)


def main():
    df_train_X, train_y = load_dataset(TRAINSET_URL)
    df_eval_X, eval_y = load_dataset(EVALSET_URL)
    feature_types = {"categorical": CATEGORICAL_COLUMNS, "numeric": NUMERIC_COLUMNS}
    feature_columns = create_features(df_train_X, feature_types)
    train_input_fn = make_input_fn(df_train_X, train_y)
    eval_input_fn = make_input_fn(df_eval_X, eval_y, num_epochs=1, shuffle=False)
    model = train(train_input_fn, feature_columns=feature_columns)
    result = predict(model, eval_input_fn)
    print(result)


if __name__ == "__main__":
    main()
