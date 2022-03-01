"""Functionality for loading data from TileDB arrays to the PyTorch Dataloader API."""

import math
from typing import Iterator, Optional, Sequence

import numpy as np
import torch

import tiledb

from ._batch_utils import BaseDenseBatch, BaseSparseBatch, tensor_generator


class PyTorchTileDBDataset(torch.utils.data.IterableDataset[Sequence[torch.Tensor]]):
    """Loads data from TileDB to the PyTorch Dataloader API."""

    def __init__(
        self,
        x_array: tiledb.Array,
        y_array: tiledb.Array,
        batch_size: int,
        buffer_bytes: Optional[int] = None,
        batch_shuffle: bool = False,
        within_batch_shuffle: bool = False,
        x_attrs: Sequence[str] = (),
        y_attrs: Sequence[str] = (),
    ):
        """Return an IterableDataset for loading data from TileDB arrays.

        :param x_array: TileDB array of the features.
        :param y_array: TileDB array of the labels.
        :param batch_size: Size of each batch.
        :param buffer_bytes: Size (in bytes) of the buffer used to read from each array.
            If not given, it is determined automatically.
        :param batch_shuffle: True for shuffling batches.
        :param within_batch_shuffle: True for shuffling records in each batch.
        :param x_attrs: Attribute names of x_array.
        :param y_attrs: Attribute names of y_array.
        """
        super().__init__()
        rows: int = x_array.shape[0]
        if rows != y_array.shape[0]:
            raise ValueError("X and Y arrays must have the same number of rows")

        self._rows = rows
        self._generator_kwargs = dict(
            dense_batch_cls=PyTorchDenseBatch,
            sparse_batch_cls=PyTorchSparseBatch,
            x_array=x_array,
            y_array=y_array,
            batch_size=batch_size,
            buffer_bytes=buffer_bytes,
            batch_shuffle=batch_shuffle,
            within_batch_shuffle=within_batch_shuffle,
            x_attrs=x_attrs,
            y_attrs=y_attrs,
        )

    def __iter__(self) -> Iterator[Sequence[torch.Tensor]]:
        kwargs = self._generator_kwargs.copy()
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            for array_key in "x_array", "y_array":
                if isinstance(kwargs[array_key], tiledb.SparseArray):
                    raise NotImplementedError(
                        "https://github.com/pytorch/pytorch/issues/20248"
                    )
            per_worker = int(math.ceil(self._rows / worker_info.num_workers))
            start_offset = worker_info.id * per_worker
            stop_offset = min(start_offset + per_worker, self._rows)
            kwargs.update(start_offset=start_offset, stop_offset=stop_offset)
        return tensor_generator(**kwargs)


class PyTorchDenseBatch(BaseDenseBatch[torch.Tensor]):
    @staticmethod
    def _tensor_from_numpy(data: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(data)


class PyTorchSparseBatch(BaseSparseBatch[torch.Tensor]):
    @staticmethod
    def _tensor_from_coo(
        data: np.ndarray,
        coords: np.ndarray,
        dense_shape: Sequence[int],
        dtype: np.dtype,
    ) -> torch.Tensor:
        return torch.sparse_coo_tensor(
            torch.tensor(coords).t(), data, dense_shape, requires_grad=False
        )
