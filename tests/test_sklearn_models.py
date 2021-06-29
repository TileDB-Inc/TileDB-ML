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
class TestSklearnModel:
    def test_save_load(self, tmpdir, net):
        tiledb_array = os.path.join(tmpdir, "test_array")
        model = net()
        tiledb_sklearn_obj = SklearnTileDB(uri=tiledb_array)
        tiledb_sklearn_obj.save(model=model)
        loaded_model = tiledb_sklearn_obj.load()
        assert all(
            [a == b for a, b in zip(model.get_params(), loaded_model.get_params())]
        )

    def test_preview(self, tmpdir, net):
        # With model as argument
        tiledb_array = os.path.join(tmpdir, "test_array")
        model = net()
        tiledb_sklearn_obj = SklearnTileDB(uri=tiledb_array, model=model)
        assert type(tiledb_sklearn_obj.preview()) == str

        # Without model as argumet
        tiledb_sklearn_obj = SklearnTileDB(uri=tiledb_array)
        assert type(tiledb_sklearn_obj.preview()) == str

    def test_file_properties(self, tmpdir, net):
        model = net()
        tiledb_array = os.path.join(tmpdir, "model_array")
        tiledb_obj = SklearnTileDB(uri=tiledb_array)
        tiledb_obj.save(model=model)

        assert tiledb_obj._file_properties["ML_FRAMEWORK"] == "SKLEARN"
        assert tiledb_obj._file_properties["STAGE"] == "STAGING"
        assert (
            tiledb_obj._file_properties["PYTHON_VERSION"] == platform.python_version()
        )
        assert (
            tiledb_obj._file_properties["ML_FRAMEWORK_VERSION"] == sklearn.__version__
        )
        assert tiledb_obj._file_properties["PREVIEW"] == ""

    def test_file_properties_in_tiledb_cloud_case(self, tmpdir, net, mocker):
        model = net()
        tiledb_array = os.path.join(tmpdir, "model_array")

        mocker.patch(
            "tiledb.ml.models.base.TileDBModel.get_cloud_uri", return_value=tiledb_array
        )
        mocker.patch("tiledb.ml._cloud_utils.update_file_properties")

        tiledb_obj = SklearnTileDB(uri=tiledb_array, namespace="test_namespace")
        tiledb_obj.save(model=model)

        assert tiledb_obj._file_properties["ML_FRAMEWORK"] == "SKLEARN"
        assert tiledb_obj._file_properties["STAGE"] == "STAGING"
        assert (
            tiledb_obj._file_properties["PYTHON_VERSION"] == platform.python_version()
        )
        assert (
            tiledb_obj._file_properties["ML_FRAMEWORK_VERSION"] == sklearn.__version__
        )
        assert tiledb_obj._file_properties["PREVIEW"] == ""

    def test_exception_raise_file_property_in_meta_error(self, tmpdir, net):
        model = net()
        tiledb_array = os.path.join(tmpdir, "model_array")
        tiledb_obj = SklearnTileDB(uri=tiledb_array)
        with pytest.raises(ValueError):
            tiledb_obj.save(model=model, meta={"ML_FRAMEWORK": "ML_FRAMEWORK"})
