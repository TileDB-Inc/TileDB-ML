import pytest
import os

import sklearn.base

from tiledb.ml.models.sklearn import SklearnTileDB


def iter_models(*modules):
    for estim_name, estim_class in sklearn.utils.all_estimators():
        if str(estim_class).split(".")[1] in modules:
            yield estim_class


@pytest.mark.parametrize(
    "net",
    list(iter_models("svm", "linear_model", "naive_bayes", "tree")),
)
def test_save_load(tmpdir, net):
    tiledb_array = os.path.join(tmpdir, "test_array")
    model = net()
    tiledb_sklearn_obj = SklearnTileDB(uri=tiledb_array)
    tiledb_sklearn_obj.save(model=model)
    loaded_model = tiledb_sklearn_obj.load()
    assert all([a == b for a, b in zip(model.get_params(), loaded_model.get_params())])


@pytest.mark.parametrize(
    "net",
    list(iter_models("svm", "linear_model", "naive_bayes", "tree")),
)
def test_preview(tmpdir, net):
    tiledb_array = os.path.join(tmpdir, "test_array")
    model = net()
    tiledb_sklearn_obj = SklearnTileDB(uri=tiledb_array)
    print(tiledb_sklearn_obj.preview(model))
    assert type(tiledb_sklearn_obj.preview(model)) == str
