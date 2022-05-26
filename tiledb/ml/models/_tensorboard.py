import glob
import os
import pickle
from typing import Mapping, Optional

import tiledb

from ._base import Timestamp

_KEY = "__TENSORBOARD__"


def save_tensorboard(log_dir: str) -> Mapping[str, bytes]:
    """
    :param log_dir: The directory to find the tensorboard tfevents data
    """
    event_files = {}
    for path in glob.glob(f"{log_dir}/*tfevents*"):
        with open(path, "rb") as f:
            event_files[path] = f.read()
    return {_KEY: pickle.dumps(event_files, protocol=4)}


def load_tensorboard(
    uri: str,
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
    with tiledb.open(uri, ctx=ctx, timestamp=timestamp) as model_array:
        for path, file_bytes in pickle.loads(model_array.meta[_KEY]).items():
            log_dir = target_dir if target_dir else os.path.dirname(path)
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)
            with open(os.path.join(log_dir, os.path.basename(path)), "wb") as f:
                f.write(file_bytes)
