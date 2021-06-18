"""Functionality for saving and loading Sklearn models as TileDB arrays"""

import pickle
import numpy as np
import json
import tiledb

from typing import Optional, Tuple

import sklearn
from sklearn.base import BaseEstimator

from .base import TileDBModel
import platform
from . import (
    FilePropertyName_ML_FRAMEWORK,
    FilePropertyName_STAGE,
    FilePropertyName_PYTHON_VERSION,
    FilePropertyName_ML_FRAMEWORK_VERSION,
)


class SklearnTileDB(TileDBModel):
    """
    Class that implements all functionality needed to save Sklearn models as
    TileDB arrays and load Sklearn models from TileDB arrays.
    """

    def save(
        self, model: BaseEstimator, update: bool = False, meta: Optional[dict] = None
    ):
        """
        Saves a Sklearn model as a TileDB array.
        :param model: An Sklearn Estimator object. Model to store as TileDB array.
        :param update: Boolean. Whether we should update any existing TileDB array
        model at the target location.
        :param meta: Dict. Extra metadata to save in a TileDB array.
        """
        # Serialize model
        serialized_model = self._serialize_model(model)

        # Create TileDB model array
        if not update:
            self._create_array()

        self._write_array(serialized_model=serialized_model, meta=meta)

    def load(self, timestamp: Optional[Tuple[int, int]] = None) -> BaseEstimator:
        """
        Loads a Sklearn model from a TileDB array.
        :param timestamp: Tuple of int. In case we want to use TileDB time travelling, we can provide a range of
        timestamps in order to load fragments of the array which live in the specified time range.
        :return: BaseEstimator. A Sklearn model object.
        """
        model_array = tiledb.open(self.uri, ctx=self.ctx, timestamp=timestamp)
        model_array_results = model_array[:]
        model = pickle.loads(model_array_results["model_params"].item(0))
        return model

    def _create_array(self):
        """
        Creates a TileDB array for a Sklearn model.
        """
        dom = tiledb.Domain(
            tiledb.Dim(
                name="model", domain=(1, 1), tile=1, dtype=np.int32, ctx=self.ctx
            ),
        )

        attrs = [
            tiledb.Attr(
                name="model_params",
                dtype="S1",
                var=True,
                filters=tiledb.FilterList([tiledb.ZstdFilter()]),
                ctx=self.ctx,
            ),
        ]

        schema = tiledb.ArraySchema(domain=dom, sparse=False, attrs=attrs, ctx=self.ctx)

        tiledb.Array.create(self.uri, schema, ctx=self.ctx)

        # In case we are on TileDB-Cloud we have to update model array's file properties
        if self.namespace:
            from tiledb.ml._cloud_utils import update_file_properties

            file_properties = {
                FilePropertyName_ML_FRAMEWORK: "SKLEARN",
                FilePropertyName_STAGE: "STAGING",
                FilePropertyName_PYTHON_VERSION: platform.python_version(),
                FilePropertyName_ML_FRAMEWORK_VERSION: sklearn.__version__,
            }
            update_file_properties(self.uri, file_properties)

    def _write_array(self, serialized_model: bytes, meta: Optional[dict]):
        """
        Writes a Sklearn model to a TileDB array.
        :param serialized_model: Bytes. A pickled sklearn model.
        :param meta: Dict. A dictionary that can contain any kind of metadata in a (key, value) form.
        """
        with tiledb.open(self.uri, "w", ctx=self.ctx) as tf_model_tiledb:
            # Insertion in TileDB array
            tf_model_tiledb[:] = {"model_params": np.array([serialized_model])}

            # Add extra metadata given by the user to array's metadata
            if meta:
                for key, value in meta.items():
                    tf_model_tiledb.meta[key] = json.dumps(value).encode("utf8")

    @staticmethod
    def _serialize_model(model: BaseEstimator) -> bytes:
        """
        Serializes a Sklearn model with pickle.
        :param model: A Sklearn Estimator object.
        :return: Bytes. Pickled Sklearn model.
        """
        return pickle.dumps(model, protocol=4)
