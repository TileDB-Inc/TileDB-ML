"""Functionality for saving and loading PytTorch models as TileDB arrays"""

import pickle
import json
import numpy as np
import tiledb

from typing import Optional, Tuple

import torch
from torch.optim import Optimizer
from torch.nn import Module

from .base import TileDBModel
import platform
from . import (
    FilePropertyName_ML_FRAMEWORK,
    FilePropertyName_STAGE,
    FilePropertyName_PYTHON_VERSION,
    FilePropertyName_ML_FRAMEWORK_VERSION,
)


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

    def load(
        self,
        model: Module,
        optimizer: Optimizer,
        timestamp: Optional[Tuple[int, int]] = None,
    ) -> dict:
        """
        Loads a PyTorch model from a TileDB array.
        :param model: Pytorch Module. A defined PyTorch model.
        :param optimizer: PyTorch Optimizer. A defined PyTorch optimizer.
        :param timestamp: Tuple of int. In case we want to use TileDB time travelling, we can provide a range of
        timestamps in order to load fragments of the array which live in the specified time range.
        :return: Dict. A dictionary with attributes other than model or optimizer
        state_dict.
        """
        model_array = tiledb.open(self.uri, ctx=self.ctx, timestamp=timestamp)
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
        :param serialized_model_info: Dict. A dictionary with serialized (pickled) information of a PyTorch model.
        """
        dom = tiledb.Domain(
            tiledb.Dim(
                name="model", domain=(1, 1), tile=1, dtype=np.int32, ctx=self.ctx
            ),
        )

        attrs = []

        for key in serialized_model_info:
            attrs.append(
                tiledb.Attr(
                    name=key,
                    dtype="S1",
                    var=True,
                    filters=tiledb.FilterList([tiledb.ZstdFilter()]),
                    ctx=self.ctx,
                ),
            )

        schema = tiledb.ArraySchema(
            domain=dom,
            sparse=False,
            attrs=attrs,
            ctx=self.ctx,
        )

        tiledb.Array.create(self.uri, schema, ctx=self.ctx)

        # In case we are on TileDB-Cloud we have to update model array's file properties
        if self.namespace:
            from tiledb.ml._cloud_utils import update_file_properties

            file_properties = {
                FilePropertyName_ML_FRAMEWORK: "PYTORCH",
                FilePropertyName_STAGE: "STAGING",
                FilePropertyName_PYTHON_VERSION: platform.python_version(),
                FilePropertyName_ML_FRAMEWORK_VERSION: torch.__version__,
            }
            update_file_properties(self.uri, file_properties)

    def _write_array(self, serialized_model_info: dict, meta: Optional[dict]):
        """
        Writes a PyTorch model to a TileDB array.
        :param serialized_model_info: Dict. A dictionary with serialized (pickled) information of a PyTorch model.
        :param meta: Optional Dict. Extra metadata the user will save as model array's metadata.
        """
        with tiledb.open(self.uri, "w", ctx=self.ctx) as tf_model_tiledb:
            # Insertion in TileDB array
            insertion_dict = {
                key: np.array([value]) for key, value in serialized_model_info.items()
            }

            tf_model_tiledb[:] = insertion_dict

            # Add extra metadata given by the user to array's metadata
            if meta:
                for key, value in meta.items():
                    tf_model_tiledb.meta[key] = json.dumps(value).encode("utf8")
