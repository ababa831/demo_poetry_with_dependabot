import numpy as np
import pytest

from train import TRAINSET_URL, EVALSET_URL
from train import load_dataset, make_input_fn


@pytest.fixture()
def unittest_dataset():
    objective_col = "survived"
    df_train_X, train_y = load_dataset(TRAINSET_URL, objective_col)
    df_eval_X, eval_y = load_dataset(EVALSET_URL, objective_col)
    train_set = (df_train_X, train_y)
    eval_set = (df_eval_X, eval_y)
    return train_set, eval_set, objective_col


@pytest.fixture()
def expected_dtypes():
    return {
        "sex": np.dtype("O"),
        "age": np.dtype("float64"),
        "n_siblings_spouses": np.dtype("int64"),
        "parch": np.dtype("int64"),
        "fare": np.dtype("float64"),
        "class": np.dtype("O"),
        "deck": np.dtype("O"),
        "embark_town": np.dtype("O"),
        "alone": np.dtype("O"),
    }


def test_load_dataset(unittest_dataset):
    train_set = unittest_dataset[0]
    eval_set = unittest_dataset[1]
    objective_col = unittest_dataset[2]
    for X, y in [train_set, eval_set]:
        errmsg = "The number of input/output rows does not match"
        assert X.shape[0] == y.shape[0], errmsg
        assert objective_col == y.name


def test_make_input_fn(unittest_dataset, expected_dtypes):
    df_train_X, train_y = unittest_dataset[0]
    ds = make_input_fn(df_train_X, train_y)()

    expected_cols = df_train_X.columns
    for feature_batch, _ in ds.take(1):
        for k in feature_batch.keys():
            errmsg = f"{k} not in feature"
            assert k in expected_cols, errmsg
            errmsg = f"Invalid dtypes "
            # pytestのオプションで不一致時に数値比較が出るので,
            # errmsg内には数値比較を明示していない
            assert feature_batch[k].dtype == expected_dtypes[k], errmsg


"""
def test_drop_missing_values(arg):
    # TODO: 欠損値データのあるデータを用意
    pass
"""