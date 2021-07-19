"""Tests for TileDB Sklearn model save and load."""

import pytest
import os
import platform

import sklearn
import sklearn.base

from itertools import zip_longest

from tiledb.ml.models.sklearn import SklearnTileDBModel


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
        tiledb_sklearn_obj = SklearnTileDBModel(uri=tiledb_array, model=model)
        tiledb_sklearn_obj.save()
        loaded_model = tiledb_sklearn_obj.load()
        assert all(
            [
                a == b
                for a, b in zip_longest(model.get_params(), loaded_model.get_params())
            ]
        )

    def test_preview(self, tmpdir, net):
        # With model as argument
        tiledb_array = os.path.join(tmpdir, "test_array")
        model = net()
        tiledb_sklearn_obj = SklearnTileDBModel(uri=tiledb_array, model=model)
        assert type(tiledb_sklearn_obj.preview()) == str
        tiledb_sklearn_obj_none = SklearnTileDBModel(uri=tiledb_array, model=None)
        assert tiledb_sklearn_obj_none.preview() == ""

    def test_get_cloud_uri(self, tmpdir, net, mocker):
        tiledb_array = os.path.join(tmpdir, "test_array")
        model = net()
        tiledb_sklearn_obj = SklearnTileDBModel(uri=tiledb_array, model=model)

        mocker.patch("tiledb.ml._cloud_utils.get_s3_prefix", return_value=None)
        with pytest.raises(ValueError):
            tiledb_sklearn_obj.get_cloud_uri(tiledb_array)

        mocker.patch("tiledb.ml._cloud_utils.get_s3_prefix", return_value="bar")
        actual = tiledb_sklearn_obj.get_cloud_uri(tiledb_array)
        expected = "tiledb://{}/{}".format(
            tiledb_sklearn_obj.namespace, os.path.join("bar", tiledb_array)
        )
        assert actual == expected

    def test_file_properties(self, tmpdir, net):
        model = net()
        tiledb_array = os.path.join(tmpdir, "model_array")
        tiledb_obj = SklearnTileDBModel(uri=tiledb_array, model=model)
        tiledb_obj.save()

        assert tiledb_obj._file_properties["TILEDB_ML_MODEL_ML_FRAMEWORK"] == "SKLEARN"
        assert tiledb_obj._file_properties["TILEDB_ML_MODEL_STAGE"] == "STAGING"
        assert (
            tiledb_obj._file_properties["TILEDB_ML_MODEL_PYTHON_VERSION"]
            == platform.python_version()
        )
        assert (
            tiledb_obj._file_properties["TILEDB_ML_MODEL_ML_FRAMEWORK_VERSION"]
            == sklearn.__version__
        )
        assert tiledb_obj._file_properties["TILEDB_ML_MODEL_PREVIEW"] == str(model)

    def test_file_properties_in_tiledb_cloud_case(self, tmpdir, net, mocker):
        model = net()
        tiledb_array = os.path.join(tmpdir, "model_array")

        mocker.patch(
            "tiledb.ml.models.base.TileDBModel.get_cloud_uri", return_value=tiledb_array
        )
        mocker.patch("tiledb.ml._cloud_utils.update_file_properties")

        tiledb_obj = SklearnTileDBModel(
            uri=tiledb_array, namespace="test_namespace", model=model
        )
        tiledb_obj.save()

        assert tiledb_obj._file_properties["TILEDB_ML_MODEL_ML_FRAMEWORK"] == "SKLEARN"
        assert tiledb_obj._file_properties["TILEDB_ML_MODEL_STAGE"] == "STAGING"
        assert (
            tiledb_obj._file_properties["TILEDB_ML_MODEL_PYTHON_VERSION"]
            == platform.python_version()
        )
        assert (
            tiledb_obj._file_properties["TILEDB_ML_MODEL_ML_FRAMEWORK_VERSION"]
            == sklearn.__version__
        )
        assert tiledb_obj._file_properties["TILEDB_ML_MODEL_PREVIEW"] == str(model)

    def test_exception_raise_file_property_in_meta_error(self, tmpdir, net):
        model = net()
        tiledb_array = os.path.join(tmpdir, "model_array")
        tiledb_obj = SklearnTileDBModel(uri=tiledb_array, model=model)
        with pytest.raises(ValueError):
            tiledb_obj.save(
                meta={"TILEDB_ML_MODEL_ML_FRAMEWORK": "TILEDB_ML_MODEL_ML_FRAMEWORK"},
            )
