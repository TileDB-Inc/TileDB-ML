"""Functionality for saving and loading PytTorch models as TileDB arrays"""

import logging
import pickle
import platform
import json
import numpy as np
import tiledb

from urllib.error import HTTPError
from typing import Optional

import torch
from torch.optim import Optimizer
from torch.nn import Module

from .base import TileDBModel


class PyTorchTileDB(TileDBModel):
    """
    Class that implements all functionality needed to save PyTorch models as
    TileDB arrays and load PyTorch models from TileDB arrays.
    """

    def save(self, model_info: dict, update: bool = False, meta: Optional[dict] = None):
        """
        Saves a PyTorch model as a TileDB array.
        :param model_info: Python dictionary. Contains all model info like,
        model.state_dict(), optimizer.state_dict(), loss, etc.
        :param update: Whether we should update any existing TileDB array
        model at the target location.
        :param meta: Dict. Extra metadata to save in a TileDB array.
        """

        # Serialize model information
        serialized_model_info = {
            key: pickle.dumps(value, protocol=4) for key, value in model_info.items()
        }

        # Create TileDB model array
        if not update:
            self._create_array(serialized_model_info)

        self._write_array(serialized_model_info=serialized_model_info, meta=meta)

    def load(self, model: Module, optimizer: Optimizer) -> dict:
        """
        Loads a PyTorch model from a TileDB array.
        :param model: Pytorch Module. A defined PyTorch model.
        :param optimizer: PyTorch Optimizer. A defined PyTorch optimizer.
        :return: Dict. A dictionary with attributes other than model or optimizer
        state_dict.
        """

        model_array = tiledb.open(self.uri)
        model_array_results = model_array[:]
        schema = model_array.schema

        model_state_dict = pickle.loads(model_array_results["model_state_dict"].item(0))
        optimizer_state_dict = pickle.loads(
            model_array_results["optimizer_state_dict"].item(0)
        )

        # Load model's state and optimizer dictionaries
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)

        # Get the rest of the attributes
        out_dict = {}
        for idx in range(schema.nattr):
            attr_name = schema.attr(idx).name
            if (
                schema.attr(idx).name != "model_state_dict"
                and schema.attr(idx).name != "optimizer_state_dict"
            ):
                out_dict[attr_name] = pickle.loads(
                    model_array_results[attr_name].item(0)
                )
        return out_dict

    def _create_array(self, serialized_model_info: dict):
        """
        Creates a TileDB array for a PyTorch model
        """
        try:
            dom = tiledb.Domain(
                tiledb.Dim(name="model", domain=(1, 1), tile=1, dtype=np.int32),
            )

            attrs = []

            for key in serialized_model_info:
                attrs.append(
                    tiledb.Attr(
                        name=key,
                        dtype="S1",
                        var=True,
                        filters=tiledb.FilterList([tiledb.ZstdFilter()]),
                    ),
                )

            schema = tiledb.ArraySchema(
                domain=dom,
                sparse=False,
                attrs=attrs,
            )

            tiledb.Array.create(self.uri, schema)
        except tiledb.TileDBError as error:
            if "Error while listing with prefix" in str(error):
                # It is possible to land here if user sets wrong default s3 credentials
                # with respect to default s3 path
                raise HTTPError(
                    code=400,
                    msg=f"Error creating file, {error} Are your S3 credentials valid?",
                )

            if "already exists" in str(error):
                logging.warning(
                    "TileDB array already exists but update=False. "
                    "Next time set update=True. Returning"
                )
                raise error

    def _write_array(self, serialized_model_info: dict, meta: Optional[dict]):
        """
        Writes a PyTorch model to a TileDB array.
        """
        with tiledb.open(self.uri, "w") as tf_model_tiledb:
            # Insertion in TileDB array
            insertion_dict = {
                key: np.array([value]) for key, value in serialized_model_info.items()
            }

            tf_model_tiledb[:] = insertion_dict

            # Add Python version to metadata
            tf_model_tiledb.meta["python_version"] = platform.python_version()

            # Add PyTorch version to metadata
            tf_model_tiledb.meta["pytorch_version"] = torch.__version__

            # Add extra metadata given by the user to array's metadata
            if meta:
                for key, value in meta.items():
                    try:
                        tf_model_tiledb.meta[key] = json.dumps(value).encode("utf8")
                    except:
                        logging.warning(
                            "Exception occurred during Json serialization of metadata!"
                        )
