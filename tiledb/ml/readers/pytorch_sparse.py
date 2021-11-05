"""Functionality for loading data directly from sparse TileDB arrays into the PyTorch Dataloader API."""

from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.sparse import csr_matrix

import tiledb
from tiledb.ml._parallel_utils import run_io_tasks_in_parallel

DataType = Tuple[torch.Tensor, ...]


class PyTorchTileDBSparseDataset(torch.utils.data.IterableDataset[DataType]):
    """
    Class that implements all functionality needed to load data from TileDB directly to the
    PyTorch Sparse Dataloader API.
    """

    def __init__(
        self,
        x_array: tiledb.SparseArray,
        y_array: tiledb.Array,
        batch_size: int,
        buffer_size: Optional[int],
        batch_shuffle: bool = False,
        x_attribute_names: Sequence[str] = (),
        y_attribute_names: Sequence[str] = (),
    ):
        """
        Initialize a PyTorchTileDBSparseDataset.

        :param x_array: Array that contains features.
        :param y_array: Array that contains labels. The sparse tiledb arrays are ingested
            in sparse tensors but torch does not provide full functionality (experimented)
            for this case.
        :param batch_size: The size of the batch that the generator will return. Remember
            to set batch_size=None when calling the PyTorch Sparse Dataloader API,
            because batching is taking place inside the TileDB IterableDataset.
        :param buffer_size: The size of the buffer that will hold the records returned from tiledb backend. This optional
            argument provides an optimization over small batch sizes.
        :param batch_shuffle: True if we want to shuffle batches.
        :param x_attribute_names: The attribute names of x_array.
        :param y_attribute_names: The attribute names of y_array.
        """
        if type(x_array) is tiledb.DenseArray:
            raise TypeError(
                "PyTorchTileDBSparseDataset class should be used with tiledb.SparseArray representation"
            )

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

        self.x_attribute_names = (
            [x_array.schema.attr(idx).name for idx in range(x_array.schema.nattr)]
            if not x_attribute_names
            else x_attribute_names
        )

        self.y_attribute_names = (
            [y_array.schema.attr(idx).name for idx in range(y_array.schema.nattr)]
            if not y_attribute_names
            else y_attribute_names
        )

        if buffer_size is not None and buffer_size < batch_size:
            raise ValueError("Buffer size should be geq to the batch size.")

    def __check_row_dims(self, x_row_idx: np.ndarray, y_row_idx: np.ndarray) -> None:
        """
        Check the row dimensionality of x,y in case y is sparse or not

        :param x_row_idx: The row indices x_coords of x Sparse Array of the dimension
            that is being batched
        :param y_row_idx: if y is sparse array, the row indices y_coords of the dimension
            that is being batched. If y is dense array, data of y
        :raises ValueError: If unique coords idx of x and y mismatch (both-sparse) or
            when unique coords idx of x mismatch y elements when y is Dense
        """
        if np.unique(x_row_idx).size != (
            np.unique(y_row_idx).size
            if isinstance(self.y, tiledb.SparseArray)
            else y_row_idx.shape[0]
        ):
            raise ValueError(
                "X and Y should have the same number of rows, i.e., the 1st dimension "
                "of TileDB arrays X, Y should be of equal domain extent inside the batch."
            )

    def __to_csr(
        self, array_id: str, buffer: Mapping[str, np.array], offset: int
    ) -> csr_matrix:
        """
        :param array_id: The matrix on which the transformation will have effect 'X' for x_array and 'Y' for y_array
        :param buffer: The buffered slice of the matrix to be batched
        :param offset: The starting offset of the buffered slice
        :returns A CSR representation of the buffered slice of the matrix
        """
        if array_id not in ["X", "Y"]:
            raise ValueError(
                "You can either transform in CSR format inner arrays of 'X' or 'Y'"
            )

        # TODO: Only 2d arrays supported for now
        if array_id == "X":
            array = self.x
            attribute_names = self.x_attribute_names[0]
        else:
            array = self.y
            attribute_names = self.y_attribute_names[0]

        dim = array.schema.domain.dim
        row = buffer[dim(0).name]
        col = buffer[dim(1).name]
        # Normalise indices for torch.sparse.Tensor We want the coords indices in every iteration to be
        # in the range of [0, self.batch_size] so the torch.sparse.Tensors can be created batch-wise. If
        # we do not normalise the sparse tensor is being created but with a dimension [0,
        # max(coord_index)], which is overkill
        row_size_norm = row.max() - row.min() + 1
        col_size_norm = col.max() + 1
        buffer_csr = csr_matrix(
            (
                buffer[attribute_names],
                (row - offset, col),
            ),
            shape=(row_size_norm, col_size_norm),
        )
        return buffer_csr

    def __iter__(self) -> Iterator[DataType]:
        worker_info = torch.utils.data.get_worker_info()  # type: ignore

        # Get number of observations
        rows = self.x.schema.domain.shape[0]

        if worker_info is None:
            # Single worker - return full iterator
            iter_start = 0
            iter_end = rows
        else:
            # Multiple workers - split workload
            # TODO: https://github.com/pytorch/pytorch/issues/20248
            # per_worker = math.ceil(rows / worker_info.num_workers)
            # worker_id = worker_info.id
            # iter_start = worker_id * per_worker
            # iter_end = min(iter_start + per_worker, rows)
            raise NotImplementedError("https://github.com/pytorch/pytorch/issues/20248")

        offsets = np.arange(iter_start, iter_end, self.buffer_size)

        x_shape = self.x.schema.domain.shape[1:]
        y_shape = self.y.schema.domain.shape[1:]

        # Loop over batches
        with ThreadPoolExecutor(max_workers=2) as executor:
            for offset in offsets:
                # Yield the next training batch
                # Fetch the buffer_sized data from back-end in case buffer_size is enabled
                x_buffer, y_buffer = run_io_tasks_in_parallel(
                    executor,
                    (self.x, self.y),
                    self.buffer_size,
                    offset,
                )

                # COO to CSR transformation for batching and row slicing
                x_buffer_csr = self.__to_csr("X", x_buffer, offset)

                if isinstance(self.y, tiledb.SparseArray):
                    y_buffer_csr = self.__to_csr("Y", y_buffer, offset)

                # Split the buffer_size into batch_size chunks
                batch_offsets = np.arange(0, self.buffer_size, self.batch_size)

                # Shuffle offsets in case we need batch shuffling
                if self.batch_shuffle:
                    np.random.shuffle(batch_offsets)

                for batch_offset in batch_offsets:
                    x_batch = x_buffer_csr[
                        batch_offset : batch_offset + self.batch_size
                    ]

                    if x_batch.data.size == 0:
                        return

                    if isinstance(self.y, tiledb.SparseArray):
                        y_batch = y_buffer_csr[
                            batch_offset : batch_offset + self.batch_size
                        ]
                    else:
                        y_batch = {
                            attr: data[batch_offset : batch_offset + self.batch_size]
                            for attr, data in y_buffer.items()
                        }

                    # Keep row records number for cross-check between X and Y batches. Last index excluded shows to
                    # empty
                    samples_num_x = x_batch.indptr[:-1]

                    # Transform back to COO for torch.sparse_coo_tensor to digest
                    x_batch_coo = x_batch.tocoo()
                    x_coords = np.stack((x_batch_coo.row, x_batch_coo.col), axis=-1)

                    # TODO: Sparse labels are not supported by Pytorch during this iteration for completeness
                    # we support the ingestion of sparseArray in labels, but loss and backward will fail due to
                    # SparseCPU backend

                    x_tensor = tuple(
                        torch.sparse_coo_tensor(
                            torch.tensor(x_coords).t(),
                            x_batch_coo.data,
                            (self.batch_size, x_shape[0]),
                            requires_grad=False,
                        )
                        for attr in self.x_attribute_names
                    )

                    # Identify the label array hence ingest it as sparse tensor or simple tensor
                    if isinstance(self.y, tiledb.SparseArray):
                        # Keep row records number for cross-check between X and Y batches. Last index excluded shows
                        # to empty
                        samples_num_y = y_batch.indptr[:-1]

                        # Transform back to COO for torch.sparse_coo_tensor to digest
                        y_batch_coo = y_batch.tocoo()
                        y_coords = np.stack((y_batch_coo.row, y_batch_coo.col), axis=-1)

                        self.__check_row_dims(samples_num_x, samples_num_y)

                        y_tensor = tuple(
                            torch.sparse_coo_tensor(
                                torch.tensor(y_coords).t(),
                                y_batch_coo.data,
                                (self.batch_size, y_shape[0]),
                                requires_grad=False,
                            )
                            for attr in self.y_attribute_names
                        )
                    else:
                        # for the check slice the row dimension of y dense array
                        self.__check_row_dims(
                            samples_num_x, y_batch[self.y_attribute_names[0]]
                        )
                        y_tensor = tuple(
                            y_batch[attr] for attr in self.y_attribute_names
                        )

                    yield x_tensor + y_tensor
