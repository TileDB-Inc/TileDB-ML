"""Functionality for loading data from TileDB arrays to the PyTorch Dataloader API."""

from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, Mapping

import numpy as np
import scipy.sparse
import torch
from torch.utils.data._utils.worker import get_worker_info

import tiledb

from .pytorch import DataType, PyTorchTileDBDataset, _get_offset_range


class PyTorchTileDBSparseDataset(PyTorchTileDBDataset):
    """Loads data from TileDB to the PyTorch Sparse Dataloader API."""

    def __iter__(self) -> Iterator[DataType]:
        batch_size = self.batch_size
        buffer_size = self.buffer_size
        rows = self.x.schema.domain.shape[0]
        worker_info = get_worker_info()
        if worker_info is not None:
            raise NotImplementedError("https://github.com/pytorch/pytorch/issues/20248")
        x_shape = (batch_size, self.x.schema.domain.shape[1])
        y_shape = (batch_size, self.y.schema.domain.shape[1])
        with ThreadPoolExecutor(max_workers=2) as executor:
            for offset in _get_offset_range(rows, buffer_size, worker_info):
                x_buffer, y_buffer = executor.map(
                    lambda array: array[offset : offset + buffer_size],  # type: ignore
                    (self.x, self.y),
                )

                # COO to CSR transformation for batching and row slicing
                x_buffer_csr = _to_csr(self.x, self.x_attrs[0], x_buffer, offset)
                if isinstance(self.y, tiledb.SparseArray):
                    y_buffer_csr = _to_csr(self.y, self.y_attrs[0], y_buffer, offset)

                # Split the buffer_size into batch_size chunks
                batch_offsets = np.arange(0, buffer_size, batch_size)
                if self.batch_shuffle:
                    np.random.shuffle(batch_offsets)

                for batch_offset in batch_offsets:
                    x_batch = x_buffer_csr[batch_offset : batch_offset + batch_size]
                    if x_batch.data.size == 0:
                        return

                    if isinstance(self.y, tiledb.SparseArray):
                        y_batch = y_buffer_csr[batch_offset : batch_offset + batch_size]
                    else:
                        y_batch = {
                            attr: data[batch_offset : batch_offset + batch_size]
                            for attr, data in y_buffer.items()
                        }

                    # Transform back to COO for torch.sparse_coo_tensor to digest
                    x_batch_coo = x_batch.tocoo()
                    x_coords = np.stack((x_batch_coo.row, x_batch_coo.col), axis=-1)

                    # TODO: Sparse labels are not supported by Pytorch during this
                    # iteration for completeness we support the ingestion of sparseArray
                    # in labels, but loss and backward will fail due to SparseCPU backend
                    x_tensor = tuple(
                        torch.sparse_coo_tensor(
                            torch.tensor(x_coords).t(),
                            x_batch_coo.data,
                            x_shape,
                            requires_grad=False,
                        )
                        for attr in self.x_attrs
                    )

                    # Identify the label array hence ingest it as sparse tensor or simple tensor
                    if isinstance(self.y, tiledb.SparseArray):
                        y_batch_coo = y_batch.tocoo()
                        y_coords = np.stack((y_batch_coo.row, y_batch_coo.col), axis=-1)

                        if len(np.unique(x_batch.indptr[:-1])) != len(
                            np.unique(y_batch.indptr[:-1])
                        ):
                            raise ValueError(
                                "X and Y should have the same number of rows, i.e., the 1st dimension "
                                "of TileDB arrays X, Y should be of equal domain extent inside the batch."
                            )

                        y_tensor = tuple(
                            torch.sparse_coo_tensor(
                                torch.tensor(y_coords).t(),
                                y_batch_coo.data,
                                y_shape,
                                requires_grad=False,
                            )
                            for attr in self.y_attrs
                        )
                    else:
                        if len(np.unique(x_batch.indptr[:-1])) != len(
                            y_batch[self.y_attrs[0]]
                        ):
                            raise ValueError(
                                "X and Y should have the same number of rows, i.e., the 1st dimension "
                                "of TileDB arrays X, Y should be of equal domain extent inside the batch."
                            )

                        y_tensor = tuple(y_batch[attr] for attr in self.y_attrs)

                    yield x_tensor + y_tensor


def _to_csr(
    array: tiledb.SparseArray, attr: str, buffer: Mapping[str, np.array], offset: int
) -> scipy.sparse.csr_matrix:
    dim = array.schema.domain.dim
    row = buffer[dim(0).name]
    col = buffer[dim(1).name]
    row_size_norm = row.max() - row.min() + 1
    col_size_norm = col.max() + 1
    return scipy.sparse.csr_matrix(
        (buffer[attr], (row - offset, col)),
        shape=(row_size_norm, col_size_norm),
    )
