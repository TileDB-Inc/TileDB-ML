"""Functionality for loading data from TileDB arrays to the PyTorch Dataloader API."""

import math
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data._utils.worker import WorkerInfo, get_worker_info

import tiledb

from ._pytorch_batch import PyTorchBatch


class PyTorchTileDBDataset(torch.utils.data.IterableDataset[Tuple[torch.Tensor, ...]]):
    """Loads data from TileDB to the PyTorch Dataloader API."""

    def __init__(
        self,
        x_array: tiledb.Array,
        y_array: tiledb.Array,
        batch_size: int,
        buffer_size: Optional[int],
        batch_shuffle: bool = False,
        within_batch_shuffle: bool = False,
        x_attribute_names: Sequence[str] = (),
        y_attribute_names: Sequence[str] = (),
    ):
        super().__init__()
        if x_array.shape[0] != y_array.shape[0]:
            raise ValueError(
                "X and Y should have the same number of rows, i.e., the 1st dimension "
                "of TileDB arrays X, Y should be of equal domain extent."
            )

        if buffer_size is None:
            buffer_size = batch_size
        elif buffer_size < batch_size:
            raise ValueError("Buffer size should be greater or equal to batch size")

        # If no attribute names are passed explicitly, return all attributes
        if not x_attribute_names:
            x_attribute_names = _get_attr_names(x_array)
        if not y_attribute_names:
            y_attribute_names = _get_attr_names(y_array)

        self.x = x_array
        self.y = y_array
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.batch_shuffle = batch_shuffle
        self.within_batch_shuffle = within_batch_shuffle
        self.x_attrs = x_attribute_names
        self.y_attrs = y_attribute_names

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, ...]]:
        batch_size = self.batch_size
        buffer_size = self.buffer_size
        rows = self.x.shape[0]
        worker_info = get_worker_info()
        if worker_info is not None and isinstance(self.x, tiledb.SparseArray):
            raise NotImplementedError("https://github.com/pytorch/pytorch/issues/20248")

        x_batch = PyTorchBatch(self.x, self.x_attrs, batch_size)
        y_batch = PyTorchBatch(self.y, self.y_attrs, batch_size)
        with ThreadPoolExecutor(max_workers=2) as executor:
            for offset in _get_offset_range(rows, buffer_size, worker_info):
                x_buffer, y_buffer = executor.map(
                    lambda array: array[offset : offset + buffer_size],  # type: ignore
                    (self.x, self.y),
                )
                x_batch.set_buffer_offset(x_buffer, offset)
                y_batch.set_buffer_offset(y_buffer, offset)

                # Split the buffer_size into batch_size chunks
                batch_offsets = np.arange(0, buffer_size, batch_size)
                if self.batch_shuffle:
                    np.random.shuffle(batch_offsets)

                for batch_offset in batch_offsets:
                    batch_slice = slice(batch_offset, batch_offset + batch_size)
                    x_batch.set_batch_slice(batch_slice)
                    y_batch.set_batch_slice(batch_slice)
                    if len(x_batch) != len(y_batch):
                        raise ValueError(
                            "x_array and y_array should have the same number of rows, "
                            "i.e. the first dimension of x_array and y_array should be "
                            "of equal domain extent inside the batch"
                        )
                    if x_batch:
                        if self.within_batch_shuffle:
                            idx = np.arange(len(x_batch))
                            np.random.shuffle(idx)
                        else:
                            idx = Ellipsis

                        yield x_batch.get_tensors(idx) + y_batch.get_tensors(idx)

    def __len__(self) -> int:
        return int(self.x.shape[0])


def _get_attr_names(array: tiledb.Array) -> Sequence[str]:
    return tuple(array.schema.attr(idx).name for idx in range(array.schema.nattr))


def _get_offset_range(rows: int, step: int, worker_info: Optional[WorkerInfo]) -> range:
    if worker_info is None:
        # Single worker - return full range
        start = 0
        stop = rows
    else:
        # Multiple workers - split range
        per_worker = int(math.ceil(rows / worker_info.num_workers))
        worker_id = worker_info.id
        start = worker_id * per_worker
        stop = min(start + per_worker, rows)
    return range(start, stop, step)
