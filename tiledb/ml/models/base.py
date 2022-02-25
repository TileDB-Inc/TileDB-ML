"""Base Class for saving and loading machine learning models."""

import platform
import time
from abc import ABC, abstractmethod
from enum import Enum, unique
from typing import Any, Generic, Mapping, Optional, Tuple, TypeVar

import tiledb

from ._cloud_utils import get_cloud_uri

Model = TypeVar("Model")
Meta = Mapping[str, Any]
Timestamp = Tuple[int, int]


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


def current_milli_time() -> int:
    return round(time.time() * 1000)


class TileDBModel(ABC, Generic[Model]):
    """
    This is the base class for all TileDB model storage functionalities, i.e,
    store machine learning models (Tensorflow, PyTorch, etc) as TileDB arrays.
    """

    Framework: str
    FrameworkVersion: str

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

        :param uri: TileDB array uri
        :param namespace: In case we want to interact (save, load, update, check) with
            models on TileDB-Cloud we need the user's namespace on TileDB-Cloud.
            Moreover, array's uri must have an s3 prefix.
        :param ctx: TileDB Context.
        :param model: Machine learning model based on the framework we are using.
        """
        self.namespace = namespace
        self.ctx = ctx
        self.model = model
        self.uri = get_cloud_uri(uri, namespace) if namespace else uri
        self._file_properties = {
            ModelFileProperties.TILEDB_ML_MODEL_ML_FRAMEWORK.value: self.Framework,
            ModelFileProperties.TILEDB_ML_MODEL_ML_FRAMEWORK_VERSION.value: self.FrameworkVersion,
            ModelFileProperties.TILEDB_ML_MODEL_STAGE.value: "STAGING",
            ModelFileProperties.TILEDB_ML_MODEL_PYTHON_VERSION.value: platform.python_version(),
            ModelFileProperties.TILEDB_ML_MODEL_PREVIEW.value: self.preview(),
        }

    @abstractmethod
    def save(self, *, update: bool = False, meta: Optional[Meta] = None) -> None:
        """Abstract method that saves a machine learning model as a TileDB array.

        :param update: Whether we should update any existing TileDB array model at the
            target location.
        :param meta: Extra metadata to save in a TileDB array.
        """

    @abstractmethod
    def load(self, *, timestamp: Optional[Timestamp] = None) -> Model:
        """Abstract method that loads a machine learning model from a TileDB array.

        :param timestamp: Range of timestamps to load fragments of the array which live
            in the specified time range.
        """

    @abstractmethod
    def preview(self) -> str:
        """Abstract method that previews a machine learning model."""

    def update_model_metadata(
        self, array: tiledb.Array, meta: Optional[Meta] = None
    ) -> None:
        """
        Update the metadata in a TileDB model array. File properties also go in the metadata section.

        :param array: A TileDB model array.
        :param meta: A mapping with the <key, value> pairs to be inserted in array's metadata.
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
