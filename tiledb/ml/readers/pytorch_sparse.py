"""Functionality for loading data directly from sparse TileDB arrays into the PyTorch Dataloader API."""
import numpy as np
import tiledb
import torch
from typing import List, Optional


class PyTorchTileDBSparseDataset(torch.utils.data.IterableDataset):
    """
    Class that implements all functionality needed to load data from TileDB directly to the
    PyTorch Sparse Dataloader API.
    """

    def __init__(
        self,
        x_array: tiledb.Array,
        y_array: tiledb.Array,
        batch_size: int,
        x_attribute_names: Optional[List[str]] = [],
        y_attribute_names: Optional[List[str]] = [],
    ):
        """
        Initialises a PyTorchTileDBSparseDataset that inherits from PyTorch IterableDataset.
        :param x_array: TileDB Sparse Array. Array that contains features.
        :param y_array: TileDB Sparse/Dense Array. Array that contains labels. The sparse tiledb arrays are ingested in sparse
        tensors but torch does not provide full functionality (experimented) for this case.
        :param batch_size: Integer. The size of the batch that the generator will return. Remember to set batch_size=None
        when calling the PyTorch Sparse Dataloader API, because batching is taking place inside the TileDB IterableDataset.
        :param x_attribute_names: List of str. A list that contains the attribute names of TileDB array x.
        :param y_attribute_names: List of str. A list that contains the attribute names of TileDB array y.
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

    def __check_row_dims(self, x_row_idx: np.array, y_row_idx: np.array):
        """
        Check the row dimensionality of x,y in case y is sparse or not

        Parameters:
            x_row_idx (np.array): Expects the row indices x_coords of x Sparse Array of the
            dimension that is being batched

            y_row_idx (np.array): if y Sparse Array -> Expects the row indices y_coords of the
            dimension that is being batched else if y is Dense Array -> data of y

        Raises:
            ValueError: If unique coords idx of x and y mismatch (both-sparse) or
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

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

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

        x_shape = self.x.schema.domain.shape[1:]
        y_shape = self.y.schema.domain.shape[1:]

        # Loop over batches
        for offset in range(iter_start, iter_end, self.batch_size):
            # Yield the next training batch
            x_batch = self.x[offset : offset + self.batch_size]
            y_batch = self.y[offset : offset + self.batch_size]

            x_coords = []
            for i in range(0, self.x.schema.domain.ndim):
                dim_name = self.x.schema.domain.dim(i).name
                x_coords.append(x_batch[dim_name])

            # Normalise indices for torch.sparse.Tensor We want the coords indices in every iteration
            # to be in the range of [0, self.batch_size] so the torch.sparse.Tensors can be created batch-wise.
            # If we do not normalise the sparse tensor is being created but with a dimension [0, max(coord_index)],
            # which is overkill
            x_coords[0] -= x_coords[0].min()

            # TODO: Sparse labels are not supported by Pytorch during this iteration for completeness
            # we support the ingestion of sparseArray in labels, but loss and backward will fail due to
            # SparseCPU backend

            # Identify the label array hence ingest it as sparse tensor or simple tensor
            x_tensor = tuple(
                torch.sparse_coo_tensor(
                    torch.tensor(list(zip(*x_coords))).t(),
                    x_batch[attr].ravel(),
                    (self.batch_size, x_shape[0]),
                    requires_grad=False,
                )
                for attr in self.x_attribute_names
            )

            if isinstance(self.y, tiledb.SparseArray):
                y_coords = []
                for i in range(0, self.y.schema.domain.ndim):
                    dim_name = self.y.schema.domain.dim(i).name
                    y_coords.append(y_batch[dim_name])

                # Normalise indices for torch.sparse.Tensor We want the coords indices in every iteration
                # to be in the range of [0, self.batch_size] so the torch.sparse.Tensors can be created batch-wise.
                # If we do not normalise the sparse tensor is being created but with a dimension [0, max(coord_index)],
                # which is overkill
                y_coords[0] -= y_coords[0].min()

                self.__check_row_dims(x_coords[0], y_coords[0])

                y_tensor = tuple(
                    torch.sparse_coo_tensor(
                        torch.tensor(list(zip(*y_coords))).t(),
                        y_batch[attr].ravel(),
                        (self.batch_size, y_shape[0]),
                        requires_grad=False,
                    )
                    for attr in self.y_attribute_names
                )
            else:
                # for the check slice the row dimension of y dense array
                self.__check_row_dims(x_coords[0], y_batch[self.y_attribute_names[0]])
                y_tensor = tuple(
                    self.y[offset : offset + self.batch_size][attr]
                    for attr in self.y_attribute_names
                )

            yield x_tensor + y_tensor
