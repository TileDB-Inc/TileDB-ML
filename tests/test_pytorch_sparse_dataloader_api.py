"""Tests for TileDB integration with Pytorch sparse Data API."""

import os
import torch
import tiledb
import numpy as np
import pytest

import torch.nn as nn
import torch.optim as optim

from tiledb.ml.readers.pytorch_sparse import PyTorchTileDBSparseDataset
from tiledb.ml._utils import ingest_in_tiledb, create_sparse_array_one_hot_2d

# Test parameters
NUM_OF_CLASSES = 5
BATCH_SIZE = 20
ROWS = 1000


class Net(nn.Module):
    def __init__(self, shape):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(np.product(shape), 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, NUM_OF_CLASSES),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


@pytest.mark.parametrize(
    "input_shape",
    [
        (10,),
    ],
)
# TODO: Multiple workers require tiledb.SparseArray to be pickled hence serializable as well
# We test for single and multiple workers
@pytest.mark.parametrize(
    "workers",
    [0],
)
class TestTileDBSparsePyTorchDataloaderAPI:
    def test_tiledb_pytorch_sparse_data_api_train_with_multiple_dim_data(
        self, tmpdir, input_shape, workers
    ):
        dataset_shape_x = (ROWS, input_shape)
        dataset_shape_y = (ROWS,)

        tiledb_uri_x = os.path.join(tmpdir, "x")
        tiledb_uri_y = os.path.join(tmpdir, "y")

        ingest_in_tiledb(
            uri=tiledb_uri_x,
            data=create_sparse_array_one_hot_2d(*dataset_shape_x),
            batch_size=BATCH_SIZE,
            sparse=True,
        )
        ingest_in_tiledb(
            uri=tiledb_uri_y,
            data=np.random.randint(low=0, high=NUM_OF_CLASSES, size=dataset_shape_y),
            batch_size=BATCH_SIZE,
            sparse=False,
        )

        with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:

            tiledb_dataset = PyTorchTileDBSparseDataset(
                x_array=x, y_array=y, batch_size=BATCH_SIZE
            )

            assert isinstance(tiledb_dataset, torch.utils.data.IterableDataset)

            train_loader = torch.utils.data.DataLoader(
                tiledb_dataset, batch_size=None, num_workers=workers
            )

            # Train network
            net = Net(shape=dataset_shape_x[1:])
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = optim.Adam(
                net.parameters(),
                lr=0.001,
                betas=(0.9, 0.999),
                eps=1e-08,
            )

            # loop over the dataset multiple times
            for epoch in range(1):
                for inputs, labels in train_loader:
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward + backward + optimize
                    outputs = net(inputs)
                    loss = criterion(outputs, labels.type(torch.LongTensor))
                    loss.backward()
                    optimizer.step()

    def test_except_with_diff_number_of_x_y_sparse_rows(
        self, tmpdir, input_shape, workers
    ):

        # Add one extra row on X
        dataset_shape_x = (ROWS + 1, input_shape)
        dataset_shape_y = (ROWS, (NUM_OF_CLASSES,))

        tiledb_uri_x = os.path.join(tmpdir, "x")
        tiledb_uri_y = os.path.join(tmpdir, "y")

        ingest_in_tiledb(
            uri=tiledb_uri_x,
            data=create_sparse_array_one_hot_2d(*dataset_shape_x),
            batch_size=BATCH_SIZE,
            sparse=True,
        )
        ingest_in_tiledb(
            uri=tiledb_uri_y,
            data=create_sparse_array_one_hot_2d(*dataset_shape_y),
            batch_size=BATCH_SIZE,
            sparse=True,
        )

        with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:
            with pytest.raises(ValueError):
                PyTorchTileDBSparseDataset(x_array=x, y_array=y, batch_size=BATCH_SIZE)

    def test_except_with_diff_number_of_batch_x_y_rows_empty_record(
        self, tmpdir, input_shape, workers
    ):
        # Add one extra row on X
        dataset_shape_x = (ROWS, input_shape)
        dataset_shape_y = (ROWS, (NUM_OF_CLASSES,))

        tiledb_uri_x = os.path.join(tmpdir, "x")
        tiledb_uri_y = os.path.join(tmpdir, "y")

        spoiled_data = create_sparse_array_one_hot_2d(*dataset_shape_x)
        spoiled_data[np.nonzero(spoiled_data[0])] = 0

        ingest_in_tiledb(
            uri=tiledb_uri_x,
            data=spoiled_data,
            batch_size=BATCH_SIZE,
            sparse=True,
        )
        ingest_in_tiledb(
            uri=tiledb_uri_y,
            data=create_sparse_array_one_hot_2d(*dataset_shape_y),
            batch_size=BATCH_SIZE,
            sparse=False,
        )

        with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:
            with pytest.raises(Exception):
                tiledb_dataset = PyTorchTileDBSparseDataset(
                    x_array=x, y_array=y, batch_size=BATCH_SIZE
                )

                train_loader = torch.utils.data.DataLoader(
                    tiledb_dataset, batch_size=None, num_workers=0
                )

                # Train network
                net = Net(shape=dataset_shape_x[1:])
                criterion = torch.nn.CrossEntropyLoss()
                optimizer = optim.Adam(
                    net.parameters(),
                    lr=0.001,
                    betas=(0.9, 0.999),
                    eps=1e-08,
                )

                # loop over the dataset multiple times
                for epoch in range(1):
                    for inputs, labels in train_loader:
                        # zero the parameter gradients
                        optimizer.zero_grad()
                        # forward + backward + optimize
                        outputs = net(inputs)
                        loss = criterion(outputs, labels.type(torch.LongTensor))
                        loss.backward()
                        optimizer.step()

    def test_except_with_multiple_nz_value_record_of_batch_x_y_rows(
        self, tmpdir, input_shape, workers
    ):
        # Add one extra row on X
        dataset_shape_x = (ROWS, input_shape)
        dataset_shape_y = (ROWS, (NUM_OF_CLASSES,))

        tiledb_uri_x = os.path.join(tmpdir, "x")
        tiledb_uri_y = os.path.join(tmpdir, "y")

        spoiled_data = create_sparse_array_one_hot_2d(*dataset_shape_x)
        spoiled_data[0] += 1

        ingest_in_tiledb(
            uri=tiledb_uri_x,
            data=spoiled_data,
            batch_size=BATCH_SIZE,
            sparse=True,
        )
        ingest_in_tiledb(
            uri=tiledb_uri_y,
            data=create_sparse_array_one_hot_2d(*dataset_shape_y),
            batch_size=BATCH_SIZE,
            sparse=False,
        )

        with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:
            tiledb_dataset = PyTorchTileDBSparseDataset(
                x_array=x, y_array=y, batch_size=BATCH_SIZE
            )

            train_loader = torch.utils.data.DataLoader(
                tiledb_dataset, batch_size=None, num_workers=0
            )

            # Train network
            net = Net(shape=dataset_shape_x[1:])
            optimizer = optim.Adam(
                net.parameters(),
                lr=0.001,
                betas=(0.9, 0.999),
                eps=1e-08,
            )

            # loop over the dataset multiple times
            for epoch in range(1):
                for inputs, labels in train_loader:
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward + backward + optimize
                    net(inputs)
