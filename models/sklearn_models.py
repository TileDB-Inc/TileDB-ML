"""Functionality for saving and loading Sklearn models as TileDB arrays"""

import logging
import pickle
import platform
import numpy as np
import json
import tiledb

from urllib.error import HTTPError
from typing import Optional

import sklearn
from sklearn.base import BaseEstimator

from models.base_model import TileDBModel


class SklearnTileDB(TileDBModel):
    """
    Class that implements all functionality needed to save Sklearn models as
    TileDB arrays and load Sklearn models from TileDB arrays.
    """

    def __init__(self, **kwargs):
        super(SklearnTileDB, self).__init__(**kwargs)

    def save(self, model: BaseEstimator, update: Optional[bool] = False, meta: Optional[dict] = None):
        """
        Saves a Sklearn model as a TileDB array.
        :param model: An Sklearn Estimator object. Model to store as TileDB array.
        :param update: Whether we should update any existing TileDB array
        model at the target location.
        :param meta: Dict. Extra metadata to save in a TileDB array.
        """

        # Serialize model
        serialized_model = self._serialize_model(model)

        # Create TileDB model array
        if not update:
            self._create_array()

        self._write_array(serialized_model, meta=meta)

    def load(self) -> BaseEstimator:
        """
        Loads a Sklearn model from a TileDB array.
        """
        try:
            model_array = tiledb.open(self.uri)
            model_array_results = model_array[:]
            model = pickle.loads(model_array_results['model_parameters'].item(0))
            return model
        except tiledb.TileDBError as error:
            raise error
        except HTTPError as error:
            raise error
        except Exception as error:
            raise error

    def _create_array(self):
        """
        Creates a TileDB array for a Sklearn model.
        """
        try:
            dom = tiledb.Domain(
                tiledb.Dim(name="model",
                           domain=(1, 1),
                           tile=1,
                           dtype=np.int32
                           ),
            )

            attrs = [
                tiledb.Attr(name='model_parameters',
                            dtype="S1",
                            var=True,
                            filters=tiledb.FilterList([tiledb.ZstdFilter()]),
                            ),
            ]

            schema = tiledb.ArraySchema(domain=dom,
                                        sparse=False,
                                        attrs=attrs,
                                        )

            tiledb.Array.create(self.uri, schema)
        except tiledb.TileDBError as error:
            if "Error while listing with prefix" in str(error):
                # It is possible to land here if user sets wrong default s3 credentials
                # with respect to default s3 path
                raise HTTPError(code=400, msg="Error creating file, %s Are your S3 "
                                              "credentials valid?" % str(error))

            if "already exists" in str(error):
                logging.warning('TileDB array already exists but update=False. '
                                'Next time set update=True. Returning')
                raise error
        except HTTPError as error:
            raise error
        except Exception as error:
            raise error

    def _write_array(self, serialized_model: bytes, meta: Optional[dict]):
        """
        Writes a Sklearn model to a TileDB array.
        """
        with tiledb.open(self.uri, 'w') as tf_model_tiledb:
            # Insertion in TileDB array
            tf_model_tiledb[:] = {'model_parameters': np.array([serialized_model])}

            # Add Python version to metadata
            tf_model_tiledb.meta['python_version'] = platform.python_version()

            # Add Sklearn version to metadata
            tf_model_tiledb.meta['sklearn_version'] = sklearn.__version__

            # Add extra metadata given by the user to array's metadata
            if meta:
                for key, value in meta.items():
                    if isinstance(value, (dict, list, tuple)):
                        tf_model_tiledb.meta[key] = json.dumps(value).encode('utf8')
                    else:
                        tf_model_tiledb.meta[key] = value

    @staticmethod
    def _serialize_model(model: BaseEstimator) -> bytes:
        """
        Serializes a Sklearn model with pickle.
        :param model: A Sklearn Estimator object.
        :return: Bytes. Pickled Sklearn model.
        """
        return pickle.dumps(model, protocol=4)
