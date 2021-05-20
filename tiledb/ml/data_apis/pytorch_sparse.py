"""Functionality for loading data directly from dense TileDB arrays into the PyTorch Dataloader API."""
import tiledb
import torch
import math
import numpy as np


class PyTorchTileDBSparseDataset(torch.utils.data.IterableDataset):
    """
    Class that implements all functionality needed to load data from TileDB directly to the
    PyTorch Dataloader API.
    """

    def __init__(self, x_array: tiledb.Array, y_array: tiledb.Array, batch_size: int):
        """
        Initialises a PyTorchTileDBSparseDataset that inherits from PyTorch IterableDataset.
        :param x_array: TileDB Dense Array. Array that contains features.
        :param y_array: TileDB Dense Array. Array that contains labels.
        :param batch_size: Integer. The size of the batch that the generator will return. Remember to set batch_size=None
        when calling the PyTorch Dataloader API, because batching is taking place inside the TileDB IterableDataset.
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

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

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

        # x_attr_name = self.x.schema.attr(0).name
        # y_attr_name = self.y.schema.attr(0).name

        x_shape = self.x.schema.domain.shape[1:]
        y_shape = self.y.schema.domain.shape[1:]

        # Loop over batches
        for offset in range(iter_start, iter_end, self.batch_size):
            # Yield the next training batch
            y_batch = self.y[offset : offset + self.batch_size]
            x_batch = self.x[offset : offset + self.batch_size]

            # TODO: Both for dense case support multiple attributes
            values_y = list(y_batch.items())[0][1]
            values_x = list(x_batch.items())[0][1]

            # Transform to TF COO format y data
            y_data = np.array(values_y).ravel()
            x_data = np.array(values_x).ravel()
            if x_data.shape[0] != y_data.shape[0]:
                raise ValueError(
                    "X and Y should have the same number of rows, i.e., the 1st dimension "
                    "of TileDB arrays X, Y should be of equal domain extent inside the batch"
                )

            y_coords = []
            for i in range(0, self.y.schema.domain.ndim):
                dim_name = self.y.schema.domain.dim(i).name
                y_coords.append(np.array(y_batch[dim_name]))

            # Transform to TF COO format x data
            x_coords = []
            for i in range(0, self.x.schema.domain.ndim):
                dim_name = self.x.schema.domain.dim(i).name
                x_coords.append(np.array(x_batch[dim_name]))

            x_coords[0] = np.vectorize(
                lambda x: x - np.max(x_coords[0]) + self.batch_size - 1
            )(x_coords[0])
            y_coords[0] = np.vectorize(
                lambda y: y - np.max(y_coords[0]) + self.batch_size - 1
            )(y_coords[0])

            yield torch.sparse_coo_tensor(
                torch.tensor(list(zip(*x_coords))).t(),
                x_data,
                (self.batch_size, x_shape[0]),
                requires_grad=False,
            ), torch.sparse_coo_tensor(
                torch.tensor(list(zip(*y_coords))).t(),
                y_data,
                (self.batch_size, y_shape[0]),
                requires_grad=True,
            )
