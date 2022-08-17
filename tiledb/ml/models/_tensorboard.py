import glob
import os
import pickle
from typing import Any, Mapping, Optional

import numpy as np
import tensorflow as tf

import tiledb

from ._base import TileDBArtifact, Timestamp, current_milli_time
from ._file_properties import TensorboardFileProperties


class TensorBoardTileDB(TileDBArtifact[tf.keras.callbacks.TensorBoard]):
    Name = "TensorBoard"
    Version = "1.0"
    _KEY = "tensorboard_data"

    def __init__(
        self, uri: str, ctx: tiledb.Ctx = None, namespace: Optional[str] = None
    ) -> None:

        super().__init__(uri, namespace, ctx)

    def _get_file_properties(
        self,
    ) -> Mapping[str, str]:
        return {
            TensorboardFileProperties.TILEDB_ML_MODEL_TENSORBOARD_FRAMEWORK.value: self.Name,
            TensorboardFileProperties.TILEDB_ML_MODEL_TENSORBOARD_VERSION.value: self.Version,
        }

    def __create_array(self) -> None:
        """Create a TileDB array for a TensorBoard"""
        domain_info = ("tensorboard", (1, 1))
        fields = [self._KEY]
        super()._create_array(domain_info, fields)

    def save(self, *, log_dir: str = "", update: bool = False, **kwargs: Any) -> None:
        """
        :param update:
        :param log_dir: The directory to find the tensorboard tfevents data
        """

        if not os.path.exists(log_dir):
            raise ValueError(f"{log_dir} does not exist")

        # Create TileDB model array
        if not update:
            self.__create_array()

        event_files = {}
        for path in glob.glob(f"{log_dir}/*tfevents*"):
            with open(path, "rb") as f:
                event_files[path] = f.read()

        with tiledb.open(
            self.uri, "w", timestamp=current_milli_time(), ctx=self.ctx
        ) as tensorboard_array:
            # Insertion in TileDB array
            tensorboard_array[:] = {
                self._KEY: np.array(pickle.dumps(event_files, protocol=4))
            }

    def load(
        self,
        uri: str = "",
        *,
        ctx: Optional[tiledb.Ctx] = None,
        target_dir: Optional[str] = None,
        timestamp: Optional[Timestamp] = None,
    ) -> None:
        """
        :param uri: The array uri that holds tensorboard data in its metadata
        :param ctx: TileDB CTX for TileDB cloud registered arrays
        :param target_dir: The local target directory to store the tensorboard data
        :param timestamp: Range of timestamps to load fragments of the array which live in the specified time range.
        :return: Loads tensorboard metadata in local directory
        """
        with tiledb.open(uri, ctx=ctx, timestamp=timestamp) as tensorboard_array:
            for path, file_bytes in pickle.loads(tensorboard_array[:]).items():
                log_dir = target_dir if target_dir else os.path.dirname(path)
                if not os.path.exists(log_dir):
                    os.mkdir(log_dir)
                with open(os.path.join(log_dir, os.path.basename(path)), "wb") as f:
                    f.write(file_bytes)
