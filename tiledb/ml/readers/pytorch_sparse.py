"""Functionality for loading data from TileDB arrays to the PyTorch Dataloader API."""

from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, Tuple

import numpy as np
import torch
from torch.utils.data._utils.worker import get_worker_info

from ._pytorch_batch import PyTorchBatch
from .pytorch import PyTorchTileDBDataset, _get_offset_range


class PyTorchTileDBSparseDataset(PyTorchTileDBDataset):
    """Loads data from TileDB to the PyTorch Sparse Dataloader API."""

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, ...]]:
        batch_size = self.batch_size
        buffer_size = self.buffer_size
        rows = self.x.schema.domain.shape[0]
        worker_info = get_worker_info()
        if worker_info is not None:
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
                            "i.e. the first dimension of x_array and y_array should be of "
                            "equal domain extent inside the batch"
                        )
                    if x_batch:
                        yield x_batch.get_tensors() + y_batch.get_tensors()
