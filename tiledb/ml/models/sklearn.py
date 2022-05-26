"""Functionality for saving and loading Sklearn models as TileDB arrays"""

import pickle
from typing import Optional

import numpy as np
import sklearn
from sklearn import config_context
from sklearn.base import BaseEstimator

import tiledb

from ._base import Meta, TileDBModel, Timestamp, current_milli_time
from ._cloud_utils import update_file_properties


class SklearnTileDBModel(TileDBModel[BaseEstimator]):
    """
    Class that implements all functionality needed to save Sklearn models as
    TileDB arrays and load Sklearn models from TileDB arrays.
    """

    Framework = "SKLEARN"
    FrameworkVersion = sklearn.__version__

    def save(self, *, update: bool = False, meta: Optional[Meta] = None) -> None:
        """
        Save a Sklearn model as a TileDB array.

        :param update: Whether we should update any existing TileDB array model at the
            target location.
        :param meta: Extra metadata to save in a TileDB array.
        """
        # Serialize model
        serialized_model = self._serialize_model()

        # Create TileDB model array
        if not update:
            self._create_array()

        self._write_array(serialized_model=serialized_model, meta=meta)

    def load(self, *, timestamp: Optional[Timestamp] = None) -> BaseEstimator:
        """
        Load a Sklearn model from a TileDB array.

        :param timestamp: Range of timestamps to load fragments of the array which live
            in the specified time range.
        :return: A Sklearn model object.
        """
        # TODO: Change timestamp when issue in core is resolved

        model_array = tiledb.open(self.uri, ctx=self.ctx, timestamp=timestamp)
        model_array_results = model_array[:]
        model = pickle.loads(model_array_results["model_params"].item(0))
        return model

    def preview(self, *, display: str = "text") -> str:
        """
        Create a text representation of the model.

        :param display. If ‘diagram’, estimators will be displayed as a diagram in an
            HTML format when shown in a jupyter notebook. If ‘text’, estimators will be
            displayed as text.
        :return. A string representation of the models internal configuration.
        """
        if self.model:
            with config_context(display=display):
                return str(self.model)
        else:
            return ""

    def _create_array(self) -> None:
        """Create a TileDB array for a Sklearn model."""
        dom = tiledb.Domain(
            tiledb.Dim(
                name="model", domain=(1, 1), tile=1, dtype=np.int32, ctx=self.ctx
            ),
        )

        attrs = [
            tiledb.Attr(
                name="model_params",
                dtype=bytes,
                var=True,
                filters=tiledb.FilterList([tiledb.ZstdFilter()]),
                ctx=self.ctx,
            ),
        ]

        schema = tiledb.ArraySchema(domain=dom, sparse=False, attrs=attrs, ctx=self.ctx)

        tiledb.Array.create(self.uri, schema, ctx=self.ctx)

        # In case we are on TileDB-Cloud we have to update model array's file properties
        if self.namespace:
            update_file_properties(self.uri, self._file_properties)

    def _write_array(self, serialized_model: bytes, meta: Optional[Meta]) -> None:
        """
        Write a Sklearn model to a TileDB array.

        :param serialized_model: A pickled sklearn model.
        :param meta: Extra metadata to save in a TileDB array.
        """
        # TODO: Change timestamp when issue in core is resolved

        with tiledb.open(
            self.uri, "w", timestamp=current_milli_time(), ctx=self.ctx
        ) as tf_model_tiledb:
            # Insertion in TileDB array
            tf_model_tiledb[:] = {"model_params": np.array([serialized_model])}
            self.update_model_metadata(array=tf_model_tiledb, meta=meta)

    def _serialize_model(self) -> bytes:
        """
        Serialize a Sklearn model with pickle.

        :return: Pickled Sklearn model.
        """
        return pickle.dumps(self.model, protocol=4)
