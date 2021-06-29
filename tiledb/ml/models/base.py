"""Base Class for saving and loading machine learning models."""

import abc
import os
import tiledb
import json
import platform

from typing import Optional, Union
from enum import Enum, unique

from torch.nn import Module
from tensorflow.python.keras.models import Model
from sklearn.base import BaseEstimator


@unique
class ModelFileProperties(Enum):
    """
    Enum Class that contains all model array file properties.
    """

    ML_FRAMEWORK = "ML_FRAMEWORK"
    ML_FRAMEWORK_VERSION = "ML_FRAMEWORK_VERSION"
    STAGE = "STAGE"
    PYTHON_VERSION = "PYTHON_VERSION"
    PREVIEW = "PREVIEW"


class TileDBModel(abc.ABC):
    """
    This is the base class for all TileDB model storage functionalities, i.e,
    store machine learning models (Tensorflow, PyTorch, etc) as TileDB arrays.
    """

    Framework = None
    FrameworkVersion = None

    def __init__(
        self,
        uri: str,
        namespace: str = None,
        ctx: tiledb.Ctx = None,
        model: Optional[Union[Module, Model, BaseEstimator]] = None,
    ):
        """
        Base class for saving machine learning models as TileDB arrays
        and loading machine learning models from TileDB arrays. In case we need to interact
        with TileDB-Cloud we have to pass user's TileDB-Cloud namespace. If we don't
        models will be saved locally.
        :param uri: str. TileDB array uri
        :param namespace: str. In case we want to interact (save, load, update, check) with models on
        TileDB-Cloud we need the user's namespace on TileDB-Cloud. Moreover, array's uri must have an s3 prefix.
        :param ctx: tiledb.Ctx. TileDB Context.
        :param model: Machine learning model based on the framework we are using.
        """
        self.namespace = namespace
        self.ctx = ctx
        self.model = model
        self._file_properties = {}
        self.uri = self.get_cloud_uri(uri) if self.namespace else uri

    @abc.abstractmethod
    def save(self, **kwargs):
        """
        Abstract method that saves a machine learning model as a TileDB array.
        Must be implemented per machine learning framework, i.e, Tensorflow,
        PyTorch etc.
        """

    @abc.abstractmethod
    def load(self, **kwargs):
        """
        Abstract method that loads a machine learning model from a model TileDB array.
        Must be implemented per machine learning framework.
        """

    @abc.abstractmethod
    def preview(self, **kwargs):
        """
        Abstract method that previews a machine learning model.
        Must be implemented per machine learning framework, i.e, Tensorflow,
        PyTorch etc.
        """

    def set_file_properties(self):
        """
        Method that sets model array's file properties.
        """
        self._file_properties = {
            ModelFileProperties.ML_FRAMEWORK.value: self.Framework,
            ModelFileProperties.ML_FRAMEWORK_VERSION.value: self.FrameworkVersion,
            ModelFileProperties.STAGE.value: "STAGING",
            ModelFileProperties.PYTHON_VERSION.value: platform.python_version(),
            ModelFileProperties.PREVIEW.value: self.preview(),
        }

    def update_model_metadata(self, array: tiledb.Array, meta: Optional[dict] = {}):
        """
        This method updates the metadata in a TileDB model array. File properties also go in the metadata section.
        :param array: tiledb.Array. A TileDB model array.
        :param meta: dict. A dictionary with the <key, value> pairs that will be inserted in array's metadata.
        """
        # Raise ValueError in case users provide metadata with the same keys as file properties.
        if meta.keys() & self._file_properties.keys():
            raise ValueError(
                "Please avoid using file property key names as metadata keys!"
            )
        else:
            for key, value in {**meta, **self._file_properties}.items():
                array.meta[key] = json.dumps(value).encode("utf8")

    def get_cloud_uri(self, uri: str) -> str:
        from tiledb.ml._cloud_utils import get_s3_prefix

        s3_prefix = get_s3_prefix(self.namespace)

        if s3_prefix is None:
            raise ValueError(
                "You must set the default s3 prefix path for ML models in {} profile settings on TileDB-Cloud".format(
                    self.namespace
                )
            )

        return "tiledb://{}/{}".format(self.namespace, os.path.join(s3_prefix, uri))
