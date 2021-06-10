# import tempfile
import inspect
import pytest
import re
import os

import sklearn.svm as svms
import sklearn.naive_bayes as nb
import sklearn.linear_model as lm
import sklearn.tree as tree

from tiledb.ml.models.sklearn import SklearnTileDB


def model_explorer(package_name):
    model_list = []
    for name, obj in inspect.getmembers(package_name):
        if (
            inspect.isclass(obj)
            and hasattr(obj, "get_params")
            and callable(getattr(obj, "get_params"))
            and not re.search("Base", name)
        ):
            model_list.append(getattr(package_name, name))
    return model_list


@pytest.mark.parametrize(
    "net",
    [
        *model_explorer(svms),
        *model_explorer(lm),
        *model_explorer(nb),
        *model_explorer(tree),
    ],
)
def test_save_load(tmpdir, net):
    tiledb_array = os.path.join(tmpdir, "test_array")
    model = net()
    tiledb_sklearn_obj = SklearnTileDB(uri=tiledb_array)
    tiledb_sklearn_obj.save(model=model)
    loaded_model = tiledb_sklearn_obj.load()
    assert all([a == b for a, b in zip(model.get_params(), loaded_model.get_params())])
