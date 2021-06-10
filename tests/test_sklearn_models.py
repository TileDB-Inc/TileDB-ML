import pytest
import os

import sklearn.base

from tiledb.ml.models.sklearn import SklearnTileDB

tested_modules = ["svm", "linear_model", "naive_bayes", "tree"]


def model_explorer():
    model_list = []
    all_estim = sklearn.utils.all_estimators()
    for estim_name, estim_class in all_estim:
        if str(estim_class).split(".")[1] in tested_modules:
            model_list.append(estim_class)
    return model_list


@pytest.mark.parametrize(
    "net",
    [
        *model_explorer(),
    ],
)
def test_save_load(tmpdir, net):
    tiledb_array = os.path.join(tmpdir, "test_array")
    model = net()
    tiledb_sklearn_obj = SklearnTileDB(uri=tiledb_array)
    tiledb_sklearn_obj.save(model=model)
    loaded_model = tiledb_sklearn_obj.load()
    assert all([a == b for a, b in zip(model.get_params(), loaded_model.get_params())])
