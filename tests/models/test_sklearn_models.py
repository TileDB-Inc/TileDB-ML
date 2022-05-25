"""Tests for TileDB Sklearn model save and load."""

import os
import platform
from itertools import zip_longest

import pytest
import sklearn
import sklearn.base

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


class TestSklearnModelCloud:
    def test_get_cloud_uri_call_for_models_on_tiledb_cloud(self, tmpdir, mocker):
        model = sklearn.linear_model.LinearRegression()
        uri = os.path.join(tmpdir, "model_array")

        mock_get_cloud_uri = mocker.patch(
            "tiledb.ml.models._base.get_cloud_uri", return_value=uri
        )

        _ = SklearnTileDBModel(uri=uri, namespace="test_namespace", model=model)

        mock_get_cloud_uri.assert_called_once_with(uri, "test_namespace")

    def test_get_s3_prefix_call_for_models_on_tiledb_cloud(self, tmpdir, mocker):
        model = sklearn.linear_model.LinearRegression()
        uri = os.path.join(tmpdir, "model_array")

        mock_get_s3_prefix = mocker.patch(
            "tiledb.ml.models._cloud_utils.get_s3_prefix", return_value="s3 prefix"
        )

        _ = SklearnTileDBModel(uri=uri, namespace="test_namespace", model=model)

        mock_get_s3_prefix.assert_called_once_with("test_namespace")

    def test_update_file_properties_call(self, tmpdir, mocker):
        model = sklearn.linear_model.LinearRegression()
        uri = os.path.join(tmpdir, "model_array")

        mocker.patch("tiledb.ml.models._base.get_cloud_uri", return_value=uri)

        tiledb_obj = SklearnTileDBModel(
            uri=uri, namespace="test_namespace", model=model
        )

        mock_update_file_properties = mocker.patch(
            "tiledb.ml.models.sklearn.update_file_properties", return_value=None
        )
        mocker.patch("tiledb.ml.models.sklearn.SklearnTileDBModel._write_array")

        tiledb_obj.save()

        file_properties_dict = {
            "TILEDB_ML_MODEL_ML_FRAMEWORK": "SKLEARN",
            "TILEDB_ML_MODEL_ML_FRAMEWORK_VERSION": sklearn.__version__,
            "TILEDB_ML_MODEL_STAGE": "STAGING",
            "TILEDB_ML_MODEL_PYTHON_VERSION": platform.python_version(),
            "TILEDB_ML_MODEL_PREVIEW": str(model),
        }

        mock_update_file_properties.assert_called_once_with(uri, file_properties_dict)

    def test_exception_raise_file_property_in_meta_error(self, tmpdir):
        model = sklearn.linear_model.LinearRegression()
        tiledb_array = os.path.join(tmpdir, "model_array")
        tiledb_obj = SklearnTileDBModel(uri=tiledb_array, model=model)
        with pytest.raises(ValueError) as ex:
            tiledb_obj.save(
                meta={"TILEDB_ML_MODEL_ML_FRAMEWORK": "TILEDB_ML_MODEL_ML_FRAMEWORK"},
            )

        assert "Please avoid using file property key names as metadata keys!" in str(
            ex.value
        )
