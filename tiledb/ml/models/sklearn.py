"""Functionality for saving and loading Sklearn models as TileDB arrays"""

import logging
import pickle
import platform
import numpy as np
import json
import tiledb
import tiledb.cloud

from typing import Optional

import sklearn
from sklearn.base import BaseEstimator

from .base import TileDBModel
from . import FILETYPE_ML_MODEL, FilePropertyName_ML_FRAMEWORK, FilePropertyName_STAGE


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

    def load(self) -> BaseEstimator:
        """
        Loads a Sklearn model from a TileDB array.
        """
        model_array = tiledb.open(self.uri, ctx=self.ctx)
        model_array_results = model_array[:]
        model = pickle.loads(model_array_results["model_params"].item(0))
        return model

    def _create_array(self):
        """
        Creates a TileDB array for a Sklearn model.
        """
        try:
            dom = tiledb.Domain(
                tiledb.Dim(
                    name="model",
                    domain=(1, 1),
                    tile=1,
                    dtype=np.int32,
                ),
            )

            attrs = [
                tiledb.Attr(
                    name="model_params",
                    dtype="S1",
                    var=True,
                    filters=tiledb.FilterList([tiledb.ZstdFilter()]),
                ),
            ]

            schema = tiledb.ArraySchema(domain=dom, sparse=False, attrs=attrs)

            tiledb.Array.create(self.uri, schema, ctx=self.ctx)

            # In case we are on TileDB-Cloud we have to update model array's file properties
            if self.namespace:
                tiledb.cloud.array.update_file_properties(
                    uri=self.uri,
                    file_type=FILETYPE_ML_MODEL,
                    file_properties={
                        FilePropertyName_ML_FRAMEWORK: "SKLEARN",
                        FilePropertyName_STAGE: "STAGING",
                    },
                )
        except tiledb.TileDBError as error:
            if "Error while listing with prefix" in str(error):
                # It is possible to land here if user sets wrong default s3 credentials
                # with respect to default s3 path
                raise Exception(
                    f"Error creating file, {error} Are your S3 credentials valid?"
                )

            if "already exists" in str(error):
                logging.warning(
                    "TileDB array already exists but update=False. "
                    "Next time set update=True. Returning"
                )
                raise error

    def _write_array(self, serialized_model: bytes, meta: Optional[dict]):
        """
        Writes a Sklearn model to a TileDB array.
        """
        with tiledb.open(self.uri, "w", ctx=self.ctx) as tf_model_tiledb:
            # Insertion in TileDB array
            tf_model_tiledb[:] = {"model_params": np.array([serialized_model])}

            # Add Python version to metadata
            tf_model_tiledb.meta["python_version"] = platform.python_version()

            # Add Sklearn version to metadata
            tf_model_tiledb.meta["sklearn_version"] = sklearn.__version__

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
