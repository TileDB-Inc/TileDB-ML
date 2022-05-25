import glob
import os
import pickle
from typing import Mapping, Optional

import tiledb

from ._base import Timestamp

_KEY = "__TENSORBOARD__"


def save_tensorboard(log_dir: str) -> Mapping[str, bytes]:
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
    with tiledb.open(uri, ctx=ctx, timestamp=timestamp) as model_array:
        for path, file_bytes in pickle.loads(model_array.meta[_KEY]).items():
            log_dir = target_dir if target_dir else os.path.dirname(path)
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)
            with open(os.path.join(log_dir, os.path.basename(path)), "wb") as f:
                f.write(file_bytes)
