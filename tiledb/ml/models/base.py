"""Base Class for saving and loading machine learning models."""

import os
import platform
from abc import ABC, abstractmethod
from enum import Enum, unique
from typing import Any, Generic, Mapping, Optional, Tuple, TypeVar

import tiledb


@unique
class ModelFileProperties(Enum):
    """
    Enum Class that contains all model array file properties.
    """

    TILEDB_ML_MODEL_ML_FRAMEWORK = "TILEDB_ML_MODEL_ML_FRAMEWORK"
    TILEDB_ML_MODEL_ML_FRAMEWORK_VERSION = "TILEDB_ML_MODEL_ML_FRAMEWORK_VERSION"
    TILEDB_ML_MODEL_STAGE = "TILEDB_ML_MODEL_STAGE"
    TILEDB_ML_MODEL_PYTHON_VERSION = "TILEDB_ML_MODEL_PYTHON_VERSION"
    TILEDB_ML_MODEL_PREVIEW = "TILEDB_ML_MODEL_PREVIEW"


Model = TypeVar("Model")
Meta = Mapping[str, Any]
Timestamp = Tuple[int, int]


class TileDBModel(ABC, Generic[Model]):
    """
    This is the base class for all TileDB model storage functionalities, i.e,
    store machine learning models (Tensorflow, PyTorch, etc) as TileDB arrays.
    """

    Framework: Optional[str] = None
    FrameworkVersion: Optional[str] = None

    def __init__(
        self,
        uri: str,
        namespace: Optional[str] = None,
        ctx: Optional[tiledb.Ctx] = None,
        model: Optional[Model] = None,
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
        self.uri = self.get_cloud_uri(uri) if self.namespace else uri
        self._file_properties = {
            ModelFileProperties.TILEDB_ML_MODEL_ML_FRAMEWORK.value: self.Framework,
            ModelFileProperties.TILEDB_ML_MODEL_ML_FRAMEWORK_VERSION.value: self.FrameworkVersion,
            ModelFileProperties.TILEDB_ML_MODEL_STAGE.value: "STAGING",
            ModelFileProperties.TILEDB_ML_MODEL_PYTHON_VERSION.value: platform.python_version(),
            ModelFileProperties.TILEDB_ML_MODEL_PREVIEW.value: self.preview(),
        }

    @abstractmethod
    def save(self, **kwargs: Any) -> None:
        """
        Abstract method that saves a machine learning model as a TileDB array.
        Must be implemented per machine learning framework, i.e, Tensorflow,
        PyTorch etc.
        """

    @abstractmethod
    def load(self, **kwargs: Any) -> Model:
        """
        Abstract method that loads a machine learning model from a model TileDB array.
        Must be implemented per machine learning framework.
        """

    @abstractmethod
    def preview(self, **kwargs: Any) -> str:
        """
        Abstract method that previews a machine learning model.
        Must be implemented per machine learning framework, i.e, Tensorflow,
        PyTorch etc.
        """

    def update_model_metadata(
        self, array: tiledb.Array, meta: Optional[Meta] = None
    ) -> None:
        """
        This method updates the metadata in a TileDB model array. File properties also go in the metadata section.
        :param array: tiledb.Array. A TileDB model array.
        :param meta: dict. A dictionary with the <key, value> pairs that will be inserted in array's metadata.
        """
        # Raise ValueError in case users provide metadata with the same keys as file properties.
        if meta:
            if not meta.keys().isdisjoint(self._file_properties.keys()):
                raise ValueError(
                    "Please avoid using file property key names as metadata keys!"
                )
            for key, value in meta.items():
                array.meta[key] = value
        for key, value in self._file_properties.items():
            array.meta[key] = value

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
