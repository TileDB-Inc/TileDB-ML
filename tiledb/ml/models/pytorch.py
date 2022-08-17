"""Functionality for saving and loading PytTorch models as TileDB arrays"""
import os
import pickle
from typing import Any, Mapping, Optional

import numpy as np
import torch
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

import tiledb

from ._base import Meta, TileDBArtifact, Timestamp, current_milli_time, group_create
from ._tensorboard import TensorBoardTileDB


class PyTorchTileDBModel(TileDBArtifact[torch.nn.Module]):
    """
    Class that implements all functionality needed to save PyTorch models as
    TileDB arrays and load PyTorch models from TileDB arrays.
    """

    Name = "PYTORCH"
    Version = torch.__version__

    def __init__(
        self,
        uri: str,
        namespace: Optional[str] = None,
        ctx: Optional[tiledb.Ctx] = None,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[Optimizer] = None,
    ):
        super().__init__(uri, namespace, ctx, model)
        self.optimizer = optimizer

    def save(
        self,
        *,
        update: bool = False,
        meta: Optional[Meta] = None,
        model_info: Optional[Mapping[str, Any]] = None,
        summary_writer: Optional[SummaryWriter] = None,
    ) -> None:
        """
        Save a PyTorch model as a TileDB array.

        :param summary_writer:
        :param update: Whether we should update any existing TileDB array model at the
            target location.
        :param meta: Extra metadata to save in a TileDB array.
        :param model_info: Contains model info like loss, epoch etc, that could be needed
            to save a model's general checkpoint for inference and/or resuming training.
        :param summary_writer: Contains summary writer object for storing tensorboard metadata
                                in array's metadata
        """
        if self.artifact is None:
            raise RuntimeError("Model is not initialized")

        # Serialize model information
        serialized_model_info = (
            {key: pickle.dumps(value, protocol=4) for key, value in model_info.items()}
            if model_info
            else {}
        )

        serialized_model_dict = {
            "model_state_dict": pickle.dumps(self.artifact.state_dict(), protocol=4)
        }

        serialized_optimizer_dict = (
            {
                "optimizer_state_dict": pickle.dumps(
                    self.optimizer.state_dict(), protocol=4
                )
            }
            if self.optimizer
            else {}
        )

        # Create TileDB model array
        if not update:
            self.__create_array(serialized_model_info=serialized_model_info)

        self._write_array(
            {
                **serialized_model_dict,
                **serialized_optimizer_dict,
                **serialized_model_info,
            },
            meta=meta,
        )

        # Summary writer
        if summary_writer:
            tb = TensorBoardTileDB(f"{self.uri}-tensorboard", self.ctx, self.namespace)
            tb.save(log_dir=os.path.join(summary_writer.log_dir), update=update)

            # Create group for first time when callback is activated
            if not update:
                group_create(self.uri, self.ctx)

    # FIXME: This method should change to return the model, not the model_info dict
    def load(
        self,
        *,
        timestamp: Optional[Timestamp] = None,
        model: torch.nn.Module = None,
        optimizer: Optimizer = None,
        callback: bool = False,
    ) -> Optional[Mapping[str, Any]]:
        """
        Load a PyTorch model from a TileDB array.

        :param callback: Boolean variable if True will store Callback data into saved directory
        :param timestamp: Range of timestamps to load fragments of the array which live
            in the specified time range.
        :param model: A defined PyTorch model.
        :param optimizer: A defined PyTorch optimizer.
        :return: A dictionary with attributes other than model or optimizer state_dict.
        """

        # TODO: Change timestamp when issue in core is resolved
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

        if callback:
            try:
                with tiledb.open(f"{self.uri}-tensorboard") as tb_array:
                    for path, file_bytes in pickle.loads(
                        tb_array[:]["tensorboard_data"][0]
                    ).items():
                        log_dir = os.path.dirname(path)
                        if not os.path.exists(log_dir):
                            os.mkdir(log_dir)
                        with open(
                            os.path.join(log_dir, os.path.basename(path)), "wb"
                        ) as f:
                            f.write(file_bytes)
            except FileNotFoundError:
                print(f"Array {self.uri}-tensorboard does not exist")

        return out_dict

    def preview(self) -> str:
        """
        Create a string representation of the model.

        :return: str. A string representation of the models internal configuration.
        """
        return str(self.artifact) if self.artifact else ""

    def __create_array(self, serialized_model_info: Mapping[str, bytes]) -> None:
        """
        Create a TileDB array for a PyTorch model
        :param serialized_model_info: A mapping with pickled information of a PyTorch model.
        """
        domain_info = ("model", (1, 1))
        fields = ["model_state_dict"]
        if self.optimizer:
            fields.append("optimizer_state_dict")
        if serialized_model_info:
            fields.extend(serialized_model_info.keys())
        super()._create_array(domain_info, fields)

    def _write_array(
        self, serialized_model_dict: Mapping[str, bytes], meta: Optional[Meta]
    ) -> None:
        """
        Write a PyTorch model to a TileDB array.

        :param serialized_model_dict: A mapping with pickled information (model state,
            optimizer state, extra model information) of a PyTorch model.
        :param meta: Extra metadata to save in a TileDB array.
        """

        # TODO: Change timestamp when issue in core is resolved
        with tiledb.open(
            self.uri, "w", timestamp=current_milli_time(), ctx=self.ctx
        ) as tf_model_tiledb:
            # Insertion in TileDB array
            tf_model_tiledb[:] = {
                key: np.array([value]) for key, value in serialized_model_dict.items()
            }
            self.update_model_metadata(array=tf_model_tiledb, meta=meta)
