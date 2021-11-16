"""Functionality for loading data directly from dense TileDB arrays into the PyTorch Dataloader API."""

import math
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, Optional, Sequence, Tuple

import numpy as np
import torch

import tiledb
from tiledb.ml._parallel_utils import run_io_tasks_in_parallel

DataType = Tuple[np.ndarray, ...]


class PyTorchTileDBDenseDataset(torch.utils.data.IterableDataset[DataType]):
    """
    Class that implements all functionality needed to load data from TileDB directly to the
    PyTorch Dataloader API.
    """

    def __init__(
        self,
        x_array: tiledb.DenseArray,
        y_array: tiledb.DenseArray,
        batch_size: int,
        buffer_size: Optional[int],
        batch_shuffle: bool = False,
        within_batch_shuffle: bool = False,
        x_attribute_names: Sequence[str] = (),
        y_attribute_names: Sequence[str] = (),
    ):
        """
        Initialize a PyTorchTileDBDenseDataset.

        For optimal reads from a TileDB array, it is recommended to set the batch size
        equal to the tile extent of the dimension we query (here, we always query the
        first dimension of a TileDB array) in order to get a slice (batch) of the data.
        For example, in case the tile extent of the first dimension of a TileDB array
        (x or y) is equal to 32, it's recommended to set batch_size=32. Any batch size
        will work, but in case it's not equal the tile extent of the first dimension of
        the TileDB array, you won't achieve highest read speed. For more details on tiles,
        tile extent and indices in TileDB, please check here:
        https://docs.tiledb.com/main/how-to/performance/performance-tips/choosing-tiling-and-cell-layout#dense-arrays

        :param x_array: Array that contains features.
        :param y_array: Array that contains labels.
        :param batch_size: The size of the batch that the generator will return. Remember
            to set batch_size=None when calling the PyTorch Dataloader API, because
            batching is taking place inside the TileDB IterableDataset.
        :param buffer_size: The size of the buffer that will hold the records returned from tiledb backend. This optional
            argument provides an optimization over small batch sizes. Remember that when using this argument
            it is recommended to set the buffer size equal to the tile extent of the dimension we query
        :param batch_shuffle: True if we want to shuffle batches.
        :param within_batch_shuffle: True if we want to shuffle records in each batch.
        :param x_attribute_names: The attribute names of x_array.
        :param y_attribute_names: The attribute names of y_array.
        """

        # Check that x and y have the same number of rows
        if x_array.schema.domain.shape[0] != y_array.schema.domain.shape[0]:
            raise ValueError(
                "X and Y should have the same number of rows, i.e., the 1st dimension "
                "of TileDB arrays X, Y should be of equal domain extent."
            )

        self.x = x_array
        self.y = y_array
        self.batch_size = batch_size
        self.buffer_size = buffer_size or batch_size
        self.batch_shuffle = batch_shuffle
        self.within_batch_shuffle = within_batch_shuffle

        # If a user doesn't pass explicit attribute names to return per batch, we return all attributes.
        self.x_attribute_names = x_attribute_names or tuple(
            x_array.schema.attr(idx).name for idx in range(x_array.schema.nattr)
        )

        self.y_attribute_names = y_attribute_names or tuple(
            y_array.schema.attr(idx).name for idx in range(y_array.schema.nattr)
        )

        if buffer_size is not None and buffer_size < batch_size:
            raise ValueError("Buffer size should be geq to the batch size.")

    def __iter__(self) -> Iterator[DataType]:
        worker_info = torch.utils.data.get_worker_info()  # type: ignore

        # Get number of observations
        rows = self.x.schema.domain.shape[0]

        # Single worker - return full iterator
        if worker_info is None:
            iter_start = 0
            iter_end = rows

        # Multiple workers - split workload
        else:
            per_worker = int(math.ceil(rows / worker_info.num_workers))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, rows)

        offsets = np.arange(iter_start, iter_end, self.buffer_size)

        # Loop over batches
        with ThreadPoolExecutor(max_workers=2) as executor:
            for offset in offsets:
                # Summon the buffer_sized data from back-end in case buffer_size is enabled
                x_buffer, y_buffer = run_io_tasks_in_parallel(
                    executor,
                    (self.x, self.y),
                    self.buffer_size,
                    offset,
                )

                # Split the buffer_size into batch_size chunks
                batch_offsets = np.arange(0, self.buffer_size, self.batch_size)

                # Shuffle offsets in case we need batch shuffling
                if self.batch_shuffle:
                    np.random.shuffle(batch_offsets)

                for batch_offset in batch_offsets:
                    x_batch = {
                        attr: data[batch_offset : batch_offset + self.batch_size]
                        for attr, data in x_buffer.items()
                    }
                    y_batch = {
                        attr: data[batch_offset : batch_offset + self.batch_size]
                        for attr, data in y_buffer.items()
                    }

                    if self.within_batch_shuffle:
                        # We get batch length based on the first attribute, because last batch might be smaller than the
                        # batch size
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
