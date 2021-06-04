import tempfile
import inspect
import pytest
import re
import sklearn.svm as svms
import sklearn.naive_bayes as nb
import sklearn.linear_model as lm
import sklearn.tree as tree

from tiledb.ml.models.sklearn import SklearnTileDB


@pytest.fixture
def temp_uri():
    """
    Returns a temporary directory instance
    """
    return tempfile.TemporaryDirectory()


@pytest.mark.parametrize(
    "net",
    [
        *(
            getattr(svms, name)
            for name, obj in inspect.getmembers(svms)
            if inspect.isclass(obj)
            and hasattr(obj, "get_params")
            and callable(getattr(obj, "get_params"))
            and not re.search("Base", name)
        ),
        *(
            getattr(lm, name)
            for name, obj in inspect.getmembers(lm)
            if inspect.isclass(obj)
            and hasattr(obj, "get_params")
            and callable(getattr(obj, "get_params"))
            and not re.search("Base", name)
        ),
        *(
            getattr(nb, name)
            for name, obj in inspect.getmembers(nb)
            if inspect.isclass(obj)
            and hasattr(obj, "get_params")
            and callable(getattr(obj, "get_params"))
            and not re.search("Base", name)
        ),
        *(
            getattr(tree, name)
            for name, obj in inspect.getmembers(tree)
            if inspect.isclass(obj)
            and hasattr(obj, "get_params")
            and callable(getattr(obj, "get_params"))
            and not re.search("Base", name)
        ),
    ],
)
def test_save_load(temp_uri, net):
    with temp_uri as tiledb_array:
        model = net()

        tiledb_sklearn_obj = SklearnTileDB(uri=tiledb_array)
        tiledb_sklearn_obj.save(model=model)

        loaded_model = tiledb_sklearn_obj.load()

        assert all(
            [a == b for a, b in zip(model.get_params(), loaded_model.get_params())]
        )
