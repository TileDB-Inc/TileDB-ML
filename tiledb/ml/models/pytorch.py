"""Functionality for saving and loading PytTorch models as TileDB arrays"""
import os
import pickle
from typing import Any, Mapping, Optional

import numpy as np
import torch
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

import tiledb

from ._base import Meta, TileDBArtifact, Timestamp, current_milli_time


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
                log_dir=os.path.join(summary_writer.log_dir)
            )
        else:
            tensorboard = b""

        # Create TileDB model array
        if not update:
            fields = ["model_state_dict", "optimizer_state_dict", "tensorboard"]
            super()._create_array(fields=fields)

        self._write_array(
            serialized_model_dict=serialized_model_dict,
            serialized_optimizer_dict=serialized_optimizer_dict,
            serialized_tb_files=tensorboard,
            meta=meta,
        )

    def load(
        self,
        *,
        model: torch.nn.Module = None,
        optimizer: Optimizer = None,
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

        with tiledb.open(self.uri, ctx=self.ctx, timestamp=timestamp) as model_array:
            # Check if we try to load models with the old 1-cell schema.
            if model_array.schema.domain.size < np.iinfo(np.uint64).max - 1025:
                return self.__load(
                    timestamp=timestamp,
                    model=model,
                    optimizer=optimizer,
                    callback=callback,
                )
            return self.__load_v2(
                timestamp=timestamp,
                model=model,
                optimizer=optimizer,
                callback=callback,
            )

    def __load(
        self,
        model: torch.nn.Module = None,
        optimizer: Optimizer = None,
        timestamp: Optional[Timestamp] = None,
        callback: bool = False,
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

    def __load_v2(
        self,
        model: torch.nn.Module = None,
        optimizer: Optimizer = None,
        timestamp: Optional[Timestamp] = None,
        callback: bool = False,
    ) -> None:
        """
        Load a PyTorch model from a TileDB array.
        """

        with tiledb.open(self.uri, ctx=self.ctx, timestamp=timestamp) as model_array:
            model_meta = dict(model_array.meta.items())

            try:
                model_state_dict_size = model_meta["model_state_dict_size"]
            except KeyError:
                raise Exception(
                    f"model_state_dict_size metadata entry not present in {self.uri}"
                    f" (existing keys: {set(model_meta)})"
                )

            # Load model's state dictionary
            model_state_dict = pickle.loads(
                model_array[0:model_state_dict_size]["model_state_dict"]
            )
            model.load_state_dict(model_state_dict)

            # Load model's state dictionary
            if optimizer:
                try:
                    optimizer_state_dict_size = model_meta["optimizer_state_dict_size"]
                except KeyError:
                    raise Exception(
                        f"optimizer_state_dict_size metadata entry not present in {self.uri}"
                        f" (existing keys: {set(model_meta)})"
                    )

                optimizer_state_dict = pickle.loads(
                    model_array[0:optimizer_state_dict_size]["optimizer_state_dict"]
                )
                optimizer.load_state_dict(optimizer_state_dict)

            if callback:
                self._write_tensorboard_files(array=model_array)

    def preview(self) -> str:
        """
        Create a string representation of the model.

        :return: str. A string representation of the models internal configuration.
        """
        return str(self.artifact) if self.artifact else ""

    def _write_array(
        self,
        serialized_model_dict: bytes,
        serialized_optimizer_dict: bytes,
        serialized_tb_files: bytes,
        meta: Optional[Meta],
    ) -> None:
        """
        Write a PyTorch model to a TileDB array.
        """
        # TODO: Change timestamp when issue in core is resolved
        with tiledb.open(
            self.uri, "w", timestamp=current_milli_time(), ctx=self.ctx
        ) as tf_model_tiledb:

            one_d_buffer = np.frombuffer(serialized_model_dict, dtype=np.uint8)
            tf_model_tiledb[: len(one_d_buffer)] = {"model_state_dict": one_d_buffer}
            tf_model_tiledb.meta["model_state_dict_size"] = len(one_d_buffer)

            if serialized_optimizer_dict:
                one_d_buffer = np.frombuffer(serialized_optimizer_dict, dtype=np.uint8)
                tf_model_tiledb[: len(one_d_buffer)] = {
                    "optimizer_state_dict": one_d_buffer
                }
                tf_model_tiledb.meta["optimizer_state_dict_size"] = len(one_d_buffer)

            if serialized_tb_files:
                one_d_buffer = np.frombuffer(serialized_tb_files, dtype=np.uint8)
                tf_model_tiledb[: len(one_d_buffer)] = {"tensorboard": one_d_buffer}
                tf_model_tiledb.meta["tensorboard_size"] = len(one_d_buffer)

            # Insert all model metadata
            self.update_model_metadata(array=tf_model_tiledb, meta=meta)
