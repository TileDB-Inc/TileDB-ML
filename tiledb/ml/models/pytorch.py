"""Functionality for saving and loading PytTorch models as TileDB arrays"""
import os
import pickle
from typing import Any, Mapping, Optional

import torch
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

import tiledb

from ._base import Meta, TileDBArtifact, Timestamp


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
        summary_writer: Optional[SummaryWriter] = None,
    ) -> None:
        """
        Save a PyTorch model as a TileDB array.

        :param summary_writer:
        :param update: Whether we should update any existing TileDB array model at the
            target location.
        :param meta: Extra metadata to save in a TileDB array.
        :param summary_writer: Contains summary writer object for storing Tensorboard data.
        """
        if self.artifact is None:
            raise RuntimeError("Model is not initialized")

        # Serialize model state dictionary.
        serialized_model_dict = pickle.dumps(self.artifact.state_dict(), protocol=4)

        # Serialize model's optimizer dictionary.
        if self.optimizer:
            serialized_optimizer_dict = pickle.dumps(
                self.optimizer.state_dict(), protocol=4
            )
        else:
            serialized_optimizer_dict = b""

        # Serialize Tensorboard files
        if summary_writer:
            tensorboard = self._serialize_tensorboard_files(
                log_dir=summary_writer.log_dir
            )
        else:
            tensorboard = b""

        # Create TileDB model array
        if not update:
            self._create_array(fields=["model", "optimizer", "tensorboard"])

        self._write_array(
            model_params={
                "model": serialized_model_dict,
                "optimizer": serialized_optimizer_dict,
                "tensorboard": tensorboard,
            },
            meta=meta or {},
        )

    def load(
        self,
        *,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[Optimizer] = None,
        timestamp: Optional[Timestamp] = None,
        callback: bool = False,
    ) -> Any:
        """
        Load switch, i.e, decide between __load (TileDB-ML<=0.8.0) or __load_v2 (TileDB-ML>0.8.0).

        :param model: A defined PyTorch model.
        :param optimizer: A defined PyTorch optimizer.
        :param timestamp: Range of timestamps to load fragments of the array which live in the specified time range.
        :param callback: Boolean variable if True will store Callback data into saved directory
        :return: A dictionary with attributes other than model or optimizer state_dict.
        """

        load = (
            self.__load_legacy
            if self._use_legacy_schema(timestamp=timestamp)
            else self.__load
        )
        return load(
            model=model, optimizer=optimizer, timestamp=timestamp, callback=callback
        )

    def __load_legacy(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        timestamp: Optional[Timestamp],
        callback: bool,
    ) -> Optional[Mapping[str, Any]]:
        """
        Load a PyTorch model from a TileDB array.
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

    def __load(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        timestamp: Optional[Timestamp],
        callback: bool,
    ) -> None:
        """
        Load a PyTorch model from a TileDB array.
        """

        model_state_dict = self.get_weights(timestamp=timestamp)
        model.load_state_dict(model_state_dict)

        # Load model's state dictionary
        if optimizer:
            opt_state_dict = self.get_optimizer_weights(timestamp=timestamp)
            optimizer.load_state_dict(opt_state_dict)

        if callback:
            self._load_tensorboard(timestamp=timestamp)

    def preview(self) -> str:
        """
        Create a string representation of the model.

        :return: str. A string representation of the models internal configuration.
        """
        return str(self.artifact) if self.artifact else ""
