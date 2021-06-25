"""Tests for TileDB Sklearn model save and load."""

import pytest
import os
import platform

import sklearn
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
    # With model as argument
    tiledb_array = os.path.join(tmpdir, "test_array")
    model = net()
    tiledb_sklearn_obj = SklearnTileDB(uri=tiledb_array, model=model)
    assert type(tiledb_sklearn_obj.preview()) == str

    # Without model as argumet
    tiledb_sklearn_obj = SklearnTileDB(uri=tiledb_array)
    assert type(tiledb_sklearn_obj.preview()) == str


def test_file_properties_in_tiledb_cloud_case(tmpdir, mocker):
    mocker.patch("tiledb.ml._cloud_utils.get_s3_prefix", return_value="")

    tiledb_array = os.path.join(tmpdir, "model_array")
    tiledb_obj = SklearnTileDB(uri=tiledb_array, namespace="test_namespace")

    assert tiledb_obj.file_properties["ML_FRAMEWORK"] == "SKLEARN"
    assert tiledb_obj.file_properties["STAGE"] == "STAGING"
    assert tiledb_obj.file_properties["PYTHON_VERSION"] == platform.python_version()
    assert tiledb_obj.file_properties["ML_FRAMEWORK_VERSION"] == sklearn.__version__
    assert tiledb_obj.file_properties["PREVIEW"] == ""
