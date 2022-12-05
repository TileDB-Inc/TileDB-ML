"""Base Class for saving and loading machine learning models."""

import glob
import os
import pickle
import platform
import time
from abc import ABC, abstractmethod
from typing import Any, Generic, Mapping, Optional, Sequence, Tuple, TypeVar

import numpy as np

import tiledb

from ._cloud_utils import get_cloud_uri, update_file_properties
from ._file_properties import ModelFileProperties

Artifact = TypeVar("Artifact")
Meta = Mapping[str, Any]
Timestamp = Tuple[int, int]


def current_milli_time() -> int:
    return round(time.time() * 1000)


def group_create(uri: str, ctx: tiledb.Ctx) -> None:
    tiledb.group_create(f"{uri}-group", ctx)
    grp = tiledb.Group(f"{uri}-group", mode="w", ctx=ctx)
    grp.add(uri)
    grp.add(f"{uri}-tensorboard")


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
        self._file_properties = self._get_file_properties()

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

    def preview(self) -> str:
        """
        Creates a string representation of a machine learning model.
        """
        return ""

    def _get_file_properties(self) -> Mapping[str, str]:
        return {
            ModelFileProperties.TILEDB_ML_MODEL_ML_FRAMEWORK.value: self.Name,
            ModelFileProperties.TILEDB_ML_MODEL_ML_FRAMEWORK_VERSION.value: self.Version,
            ModelFileProperties.TILEDB_ML_MODEL_STAGE.value: "STAGING",
            ModelFileProperties.TILEDB_ML_MODEL_PYTHON_VERSION.value: platform.python_version(),
            ModelFileProperties.TILEDB_ML_MODEL_PREVIEW.value: self.preview(),
        }

    def _create_array(
        self,
        fields: Sequence[str],
    ) -> None:
        """Internal method that creates a TileDB array based on the model's spec."""

        # The array will be be 1 dimensional with domain of 0 to max uint64. We use a tile extent of 1024 bytes
        dom = tiledb.Domain(
            tiledb.Dim(
                name="position",
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

    @staticmethod
    def _serialize_tensorboard_files(log_dir: str = "") -> bytes:
        """Serialize all Tensorboard files."""

        if not os.path.exists(log_dir):
            raise ValueError(f"{log_dir} does not exist")

        event_files = {}
        for path in glob.glob(f"{log_dir}/*tfevents*"):
            with open(path, "rb") as f:
                event_files[path] = f.read()

        return pickle.dumps(event_files, protocol=4)

    @staticmethod
    def _write_tensorboard_files(array: tiledb.Array):  # type: ignore
        meta = dict(array.meta.items())

        try:
            tensorboard_size = meta["tensorboard_size"]
        except KeyError:
            raise Exception(
                f"tensorboard_size metadata entry not present in"
                f" (existing keys: {set(meta)})"
            )

        tensorboard_files = pickle.loads(array[0:tensorboard_size]["tensorboard"])

        for path, file_bytes in tensorboard_files:
            log_dir = os.path.dirname(path)
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)
            with open(os.path.join(log_dir, os.path.basename(path)), "wb") as f:
                f.write(file_bytes)
