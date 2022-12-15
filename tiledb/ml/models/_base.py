"""Base Class for saving and loading machine learning models."""

import glob
import os
import pickle
import platform
from abc import ABC, abstractmethod
from typing import (
    Any,
    Generic,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import numpy as np

import tiledb

from .. import __version__
from ._cloud_utils import get_cloud_uri, update_file_properties
from ._file_properties import ModelFileProperties

Artifact = TypeVar("Artifact")
Meta = Mapping[str, Any]
Timestamp = Tuple[int, int]

Weights = Union[Sequence[np.ndarray], Mapping[str, Any]]


class TileDBArtifact(ABC, Generic[Artifact]):
    """
    This is the base class for all TileDB model storage functionalities, i.e,
    store machine learning models (Tensorflow, PyTorch, etc) as TileDB arrays.
    """

    Name: str
    Version: str

    def __init__(
        self,
        uri: str,
        namespace: Optional[str] = None,
        ctx: Optional[tiledb.Ctx] = None,
        artifact: Optional[Artifact] = None,
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
        :param artifact: Machine learning artifact (e.g. machine learning model) based on the framework we are using.
        """
        self.namespace = namespace
        self.ctx = ctx
        self.artifact = artifact
        self.uri = get_cloud_uri(uri, namespace) if namespace else uri
        self._file_properties = {
            ModelFileProperties.TILEDB_ML_MODEL_ML_FRAMEWORK.value: self.Name,
            ModelFileProperties.TILEDB_ML_MODEL_ML_FRAMEWORK_VERSION.value: self.Version,
            ModelFileProperties.TILEDB_ML_MODEL_STAGE.value: "STAGING",
            ModelFileProperties.TILEDB_ML_MODEL_PYTHON_VERSION.value: platform.python_version(),
            ModelFileProperties.TILEDB_ML_MODEL_PREVIEW.value: self.preview(),
            ModelFileProperties.TILEDB_ML_MODEL_VERSION.value: __version__,
        }

    @abstractmethod
    def save(self, *, update: bool = False, meta: Optional[Meta] = None) -> None:
        """Abstract method that saves a machine learning model as a TileDB array.

        :param update: Whether we should update any existing TileDB array model at the
            target location.
        :param meta: Extra metadata to save in a TileDB array.
        """

    @abstractmethod
    def load(self, *, timestamp: Optional[Timestamp] = None) -> Artifact:
        """Abstract method that loads a machine learning model from a TileDB array.

        :param timestamp: Range of timestamps to load fragments of the array which live
            in the specified time range.
        """

    def get_weights(self, timestamp: Optional[Timestamp] = None) -> Weights:
        """
        Returns model's weights. Works for Tensorflow Keras and PyTorch
        """
        with tiledb.open(self.uri, ctx=self.ctx, timestamp=timestamp) as model_array:
            return cast(Weights, self._get_model_param(model_array, "model"))

    def get_optimizer_weights(self, timestamp: Optional[Timestamp] = None) -> Weights:
        """
        Returns optimizer's weights. Works for Tensorflow Keras and PyTorch
        """
        with tiledb.open(self.uri, ctx=self.ctx, timestamp=timestamp) as model_array:
            return cast(Weights, self._get_model_param(model_array, "optimizer"))

    @abstractmethod
    def preview(self) -> str:
        """
        Creates a string representation of a machine learning model.
        """

    def _create_array(self, fields: Sequence[str]) -> None:
        """Internal method that creates a TileDB array based on the model's spec."""

        # The array will be be 1 dimensional with domain of 0 to max uint64. We use a tile extent of 1024 bytes
        dom = tiledb.Domain(
            tiledb.Dim(
                name="offset",
                domain=(0, np.iinfo(np.uint64).max - 1025),
                tile=1024,
                dtype=np.uint64,
                ctx=self.ctx,
            ),
        )

        attrs = [
            tiledb.Attr(
                name=field,
                dtype=np.uint8,
                filters=tiledb.FilterList([tiledb.ZstdFilter()]),
                ctx=self.ctx,
            )
            for field in fields
        ]

        schema = tiledb.ArraySchema(
            domain=dom,
            sparse=False,
            attrs=attrs,
            ctx=self.ctx,
        )

        tiledb.Array.create(self.uri, schema, ctx=self.ctx)

        # In case we are on TileDB-Cloud we have to update model array's file properties
        if self.namespace:
            update_file_properties(self.uri, self._file_properties)

    def _write_array(
        self,
        model_params: Mapping[str, bytes],
        tensorboard_log_dir: Optional[str] = None,
        meta: Optional[Meta] = None,
    ) -> None:
        if tensorboard_log_dir:
            tensorboard = self._serialize_tensorboard(tensorboard_log_dir)
        else:
            tensorboard = b""
        model_params = dict(tensorboard=tensorboard, **model_params)

        if meta is None:
            meta = {}
        if not meta.keys().isdisjoint(self._file_properties.keys()):
            raise ValueError(
                "Please avoid using file property key names as metadata keys!"
            )

        with tiledb.open(self.uri, "w", ctx=self.ctx) as model_array:
            one_d_buffers = {}
            max_len = 0
            for key, value in model_params.items():
                one_d_buffer = np.frombuffer(value, dtype=np.uint8)
                one_d_buffer_len = len(one_d_buffer)
                one_d_buffers[key] = one_d_buffer
                # Write size only in case is greater than 0.
                if one_d_buffer_len:
                    model_array.meta[key + "_size"] = one_d_buffer_len
                if one_d_buffer_len > max_len:
                    max_len = one_d_buffer_len

            model_array[0:max_len] = {
                key: np.pad(value, (0, max_len - len(value)))
                for key, value in one_d_buffers.items()
            }
            for mapping in meta, self._file_properties:
                for key, value in mapping.items():
                    model_array.meta[key] = value

    def _get_model_param(self, model_array: tiledb.Array, key: str) -> Any:
        size_key = key + "_size"
        try:
            size = model_array.meta[size_key]
        except KeyError:
            raise Exception(
                f"{size_key} metadata entry not present in {self.uri}"
                f" (existing keys: {set(model_array.meta.keys())})"
            )
        return pickle.loads(model_array.query(attrs=(key,))[0:size][key].tobytes())

    @staticmethod
    def _serialize_tensorboard(log_dir: str) -> bytes:
        """Serialize all Tensorboard files."""
        if not os.path.exists(log_dir):
            raise ValueError(f"{log_dir} does not exist")
        tensorboard_files = {}
        for path in glob.glob(f"{log_dir}/*tfevents*"):
            with open(path, "rb") as f:
                tensorboard_files[path] = f.read()
        return pickle.dumps(tensorboard_files, protocol=4)

    def _load_tensorboard(self, model_array: tiledb.Array) -> None:
        """
        Write Tensorboard files to directory. Works for Tensorflow-Keras and PyTorch.
        """
        tensorboard_files = self._get_model_param(model_array, "tensorboard")
        for path, file_bytes in tensorboard_files.items():
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(file_bytes)

    def _use_legacy_schema(self, model_array: tiledb.Array) -> bool:
        # TODO: Decide based on tiledb-ml version and not on schema characteristics, like "offset".
        return str(model_array.schema.domain.dim(0).name) != "offset"
