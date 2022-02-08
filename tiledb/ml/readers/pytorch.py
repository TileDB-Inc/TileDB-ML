"""Functionality for loading data from TileDB arrays to the PyTorch Dataloader API."""

import math
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, Optional, Sequence, Tuple

import numpy as np
import torch

import tiledb

DataType = Tuple[torch.Tensor, ...]


class PyTorchTileDBDataset(torch.utils.data.IterableDataset[DataType]):
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
        if x_array.schema.domain.shape[0] != y_array.schema.domain.shape[0]:
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
        self.x_attribute_names = x_attribute_names
        self.y_attribute_names = y_attribute_names


class PyTorchTileDBDenseDataset(PyTorchTileDBDataset):
    """Loads data from TileDB to the PyTorch Dataloader API."""

    def __iter__(self) -> Iterator[DataType]:
        worker_info = torch.utils.data.get_worker_info()

        # Get number of observations
        rows = self.x.schema.domain.shape[0]

        if worker_info is None:
            # Single worker - return full iterator
            iter_start = 0
            iter_end = rows
        else:
            # Multiple workers - split workload
            per_worker = int(math.ceil(rows / worker_info.num_workers))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, rows)

        batch_size = self.batch_size
        buffer_size = self.buffer_size
        offsets = np.arange(iter_start, iter_end, buffer_size)

        # Loop over batches
        with ThreadPoolExecutor(max_workers=2) as executor:
            for offset in offsets:
                x_buffer, y_buffer = executor.map(
                    lambda array: array[offset : offset + buffer_size],  # type: ignore
                    (self.x, self.y),
                )

                # Split the buffer_size into batch_size chunks
                batch_offsets = np.arange(0, buffer_size, batch_size)

                # Shuffle offsets in case we need batch shuffling
                if self.batch_shuffle:
                    np.random.shuffle(batch_offsets)

                for batch_offset in batch_offsets:
                    x_batch = {
                        attr: data[batch_offset : batch_offset + batch_size]
                        for attr, data in x_buffer.items()
                    }
                    y_batch = {
                        attr: data[batch_offset : batch_offset + batch_size]
                        for attr, data in y_buffer.items()
                    }

                    if self.within_batch_shuffle:
                        # We get batch length based on the first attribute,
                        # because last batch might be smaller than the batch size
                        rand_permutation = np.arange(
                            x_batch[self.x_attribute_names[0]].shape[0]
                        )

                        np.random.shuffle(rand_permutation)

                        # Yield the next training batch
                        yield tuple(
                            x_batch[attr][rand_permutation]
                            for attr in self.x_attribute_names
                        ) + tuple(
                            y_batch[attr][rand_permutation]
                            for attr in self.y_attribute_names
                        )
                    else:
                        # Yield the next training batch
                        yield tuple(
                            x_batch[attr] for attr in self.x_attribute_names
                        ) + tuple(y_batch[attr] for attr in self.y_attribute_names)

    def __len__(self) -> int:
        return int(self.x.shape[0])


def _get_attr_names(array: tiledb.Array) -> Sequence[str]:
    return tuple(array.schema.attr(idx).name for idx in range(array.schema.nattr))
