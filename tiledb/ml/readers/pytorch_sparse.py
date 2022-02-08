"""Functionality for loading data from TileDB arrays to the PyTorch Dataloader API."""

from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, Mapping

import numpy as np
import scipy.sparse
import torch

import tiledb

from .pytorch import DataType, PyTorchTileDBDataset


class PyTorchTileDBSparseDataset(PyTorchTileDBDataset):
    """Loads data from TileDB to the PyTorch Sparse Dataloader API."""

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
            raise NotImplementedError("https://github.com/pytorch/pytorch/issues/20248")

        batch_size = self.batch_size
        buffer_size = self.buffer_size
        offsets = np.arange(iter_start, iter_end, buffer_size)

        x_shape = self.x.schema.domain.shape[1:]
        y_shape = self.y.schema.domain.shape[1:]

        # Loop over batches
        with ThreadPoolExecutor(max_workers=2) as executor:
            for offset in offsets:
                x_buffer, y_buffer = executor.map(
                    lambda array: array[offset : offset + buffer_size],  # type: ignore
                    (self.x, self.y),
                )

                # COO to CSR transformation for batching and row slicing
                x_buffer_csr = _to_csr(
                    self.x, self.x_attribute_names[0], x_buffer, offset
                )
                if isinstance(self.y, tiledb.SparseArray):
                    y_buffer_csr = _to_csr(
                        self.y, self.y_attribute_names[0], y_buffer, offset
                    )

                # Split the buffer_size into batch_size chunks
                batch_offsets = np.arange(0, buffer_size, batch_size)

                # Shuffle offsets in case we need batch shuffling
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

                    # Keep row records number for cross-check between X and Y batches.
                    # Last index excluded shows to empty
                    samples_num_x = x_batch.indptr[:-1]

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
                            (batch_size, x_shape[0]),
                            requires_grad=False,
                        )
                        for attr in self.x_attribute_names
                    )

                    # Identify the label array hence ingest it as sparse tensor or simple tensor
                    if isinstance(self.y, tiledb.SparseArray):
                        # Keep row records number for cross-check between X and Y batches.
                        # Last index excluded shows to empty
                        samples_num_y = y_batch.indptr[:-1]

                        # Transform back to COO for torch.sparse_coo_tensor to digest
                        y_batch_coo = y_batch.tocoo()
                        y_coords = np.stack((y_batch_coo.row, y_batch_coo.col), axis=-1)

                        _check_row_dims(self.y, samples_num_x, samples_num_y)

                        y_tensor = tuple(
                            torch.sparse_coo_tensor(
                                torch.tensor(y_coords).t(),
                                y_batch_coo.data,
                                (batch_size, y_shape[0]),
                                requires_grad=False,
                            )
                            for attr in self.y_attribute_names
                        )
                    else:
                        # for the check slice the row dimension of y dense array
                        _check_row_dims(
                            self.y, samples_num_x, y_batch[self.y_attribute_names[0]]
                        )
                        y_tensor = tuple(
                            y_batch[attr] for attr in self.y_attribute_names
                        )

                    yield x_tensor + y_tensor


def _check_row_dims(
    array: tiledb.Array, x_row_idx: np.ndarray, y_row_idx: np.ndarray
) -> None:
    if np.unique(x_row_idx).size != (
        np.unique(y_row_idx).size
        if isinstance(array, tiledb.SparseArray)
        else y_row_idx.shape[0]
    ):
        raise ValueError(
            "X and Y should have the same number of rows, i.e., the 1st dimension "
            "of TileDB arrays X, Y should be of equal domain extent inside the batch."
        )


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
