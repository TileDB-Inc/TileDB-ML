"""Functionality for saving and loading PytTorch models as TileDB arrays"""

import logging
import pickle
import numpy as np
import tiledb

from urllib.error import HTTPError
from typing import Optional

from torch.optim import Optimizer
from torch.nn import Module

from models.base_model import TileDBModel

tiledb.default_ctx()


class PyTorchTileDB(TileDBModel):
    """
    Class that implements all functionality needed to save PyTorch models as
    TileDB arrays and load PyTorch models from TileDB arrays.
    """

    def __init__(self, **kwargs):
        super(PyTorchTileDB, self).__init__(**kwargs)

    def save(self, model_info: dict, update: Optional[bool] = False):
        """
        Saves a PyTorch model as a TileDB array.
        :param model_info: Python dictionary. Contains all model info like,
        model.state_dict(), optimizer.state_dict(), loss, etc.
        :param update: Whether we should update any existing TileDB array
        model at the target location.
        """

        # Serialize model and optimizer
        serialized_model_info = self._serialize_model_info(model_info)

        # Create TileDB model array
        if not update:
            self._create_array(serialized_model_info)

        self._write_array(serialized_model_info)

    def load(self, model: Module, optimizer: Optimizer) -> dict:
        """
        Loads a PyTorch model from a TileDB array.
        :param model: Pytorch Module. A defined PyTorch model.
        :param optimizer: PyTorch Optimizer. A defined PyTorch optimizer.
        :return: PyTorch Module. A PyTorch model.
        """
        try:
            model_array = tiledb.open(self.uri)
            schema = model_array.schema

            model_state_dict = pickle.loads(model_array[:]['model_state_dict'].item(0))
            optimizer_state_dict = pickle.loads(model_array[:]['optimizer_state_dict'].item(0))

            # Load model's state and optimizer dictionaries
            model.load_state_dict(model_state_dict)
            optimizer.load_state_dict(optimizer_state_dict)

            # Get the rest of the attributes
            out_dict = {}
            for idx in range(schema.nattr):
                attr_name = schema.attr(idx).name
                if schema.attr(idx).name != 'model_state_dict' and schema.attr(idx).name != 'optimizer_state_dict':
                    out_dict[attr_name] = pickle.loads(model_array[:][attr_name].item(0))
            return out_dict
        except tiledb.TileDBError as error:
            raise error
        except HTTPError as error:
            raise error
        except Exception as error:
            raise error

    def _create_array(self, serialized_model_info: dict):
        """
        Creates a TileDB array for a PyTorch model
        """
        try:
            dom = tiledb.Domain(
                tiledb.Dim(name="model",
                           domain=(1, 1),
                           tile=1,
                           dtype=np.int32
                           ),
            )

            attrs = []

            for key in serialized_model_info:
                attrs.append(
                    tiledb.Attr(name=key,
                                dtype="S1",
                                var=True,
                                filters=tiledb.FilterList([tiledb.ZstdFilter()]),
                                ),)

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
            raise HTTPError(code=400, msg="Error creating file %s " % str(error))

    def _write_array(self, serialized_model_info: dict):
        """
        Writes a PyTorch model to a TileDB array.
        """
        with tiledb.open(self.uri, 'w') as tf_model_tiledb:
            # Insertion in TileDB array
            insertion_dict = {}

            for key, value in serialized_model_info.items():
                insertion_dict[key] = np.array([value])

            tf_model_tiledb[:] = insertion_dict

    @staticmethod
    def _serialize_model_info(model_info: dict) -> dict:
        """
        Serializes all key values in model_info dictionary.
        :param model_info: Python dictionary. Contains all model info like,
        model.state_dict(), optimizer.state_dict(), loss, etc.
        :return: Python dictionary. Contains pickled model information.
        """
        serialized_model_info = {}

        for key, value in model_info.items():
            serialized_model_info[key] = pickle.dumps(value, protocol=4)

        return serialized_model_info
