"""Tests for TileDB integration with Pytorch Data API."""

import torch
import tiledb
import numpy as np
import unittest
import tempfile

import torch.nn as nn
import torch.optim as optim

from tiledb.ml.data_apis.pytorch_sparse import PyTorchTileDBSparseDataset

# Test parameters
NUM_OF_CLASSES = 5
BATCH_SIZE = 20
ROWS = 1000

# We test for 2d
INPUT_SHAPES = [
    (10,),
]
# We test for single and multiple workers
# TODO#1: Multiple workers require tiledb.SparseArray to be pickled hence serializable as well
NUM_OF_WORKERS = [0]


def create_sparse_array_one_hot_2d(rows: int, cols: tuple) -> np.ndarray:
    seed = np.random.randint(low=0, high=cols[0], size=(rows,))
    seed[-1] = cols[0] - 1
    b = np.zeros((seed.size, seed.max() + 1))
    b[np.arange(seed.size), seed] = 1
    return b


def get_schema(data: np.array, batch_size: int, sparse: bool) -> tiledb.ArraySchema:
    dims = [
        tiledb.Dim(
            name="dim_" + str(dim),
            domain=(0, data.shape[dim] - 1),
            tile=data.shape[dim] if dim > 0 else batch_size,
            dtype=np.int32,
        )
        for dim in range(data.ndim)
    ]

    # TileDB schema
    schema = tiledb.ArraySchema(
        domain=tiledb.Domain(*dims),
        sparse=sparse,
        attrs=[tiledb.Attr(name="features", dtype=np.float32)],
    )

    return schema


def ingest_in_tiledb(data: np.array, batch_size: int, uri: str):
    schema = get_schema(data, batch_size, False)

    # Create array
    tiledb.Array.create(uri, schema)

    # Ingest
    with tiledb.open(uri, "w") as tiledb_array:
        tiledb_array[:] = {"features": data}


def ingest_in_tiledb_sparse(data: np.array, batch_size: int, uri: str):
    schema = get_schema(data, batch_size, True)

    # Create the (empty) array on disk.
    tiledb.Array.create(uri, schema)

    # Ingest
    with tiledb.open(uri, "w") as tiledb_array:
        idx = np.nonzero(data)
        I, J = idx[0], idx[1]
        tiledb_array[I, J] = {"features": data[idx]}


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


class TestTileDBSparsePyTorchDataloaderAPI(unittest.TestCase):
    def test_tiledb_pytorch_sparse_data_api_train_with_multiple_dim_data(self):
        for input_shape in INPUT_SHAPES:
            for workers in NUM_OF_WORKERS:
                with self.subTest():
                    with tempfile.TemporaryDirectory() as tiledb_uri_x, tempfile.TemporaryDirectory() as tiledb_uri_y:

                        dataset_shape_x = (ROWS, input_shape)
                        dataset_shape_y = (ROWS,)

                        ingest_in_tiledb_sparse(
                            uri=tiledb_uri_x,
                            data=create_sparse_array_one_hot_2d(
                                dataset_shape_x[0], dataset_shape_x[1]
                            ),
                            batch_size=BATCH_SIZE,
                        )
                        ingest_in_tiledb(
                            uri=tiledb_uri_y,
                            data=np.random.randint(
                                low=0, high=NUM_OF_CLASSES, size=dataset_shape_y
                            ),
                            batch_size=BATCH_SIZE,
                        )

                        with tiledb.open(tiledb_uri_x) as x, tiledb.open(
                            tiledb_uri_y
                        ) as y:

                            tiledb_dataset = PyTorchTileDBSparseDataset(
                                x_array=x, y_array=y, batch_size=BATCH_SIZE
                            )

                            self.assertIsInstance(
                                tiledb_dataset, torch.utils.data.IterableDataset
                            )

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

                            for epoch in range(
                                1
                            ):  # loop over the dataset multiple times
                                for inputs, labels in train_loader:
                                    # zero the parameter gradients
                                    optimizer.zero_grad()
                                    # forward + backward + optimize
                                    outputs = net(inputs)
                                    loss = criterion(
                                        outputs, labels.type(torch.LongTensor)
                                    )
                                    loss.backward()
                                    optimizer.step()

    def test_except_with_diff_number_of_x_y_sparse_rows(self):
        with tempfile.TemporaryDirectory() as tiledb_uri_x, tempfile.TemporaryDirectory() as tiledb_uri_y:
            # Add one extra row on X
            dataset_shape_x = (ROWS + 1, INPUT_SHAPES[0])
            dataset_shape_y = (ROWS, (NUM_OF_CLASSES,))

            ingest_in_tiledb_sparse(
                uri=tiledb_uri_x,
                data=create_sparse_array_one_hot_2d(
                    dataset_shape_x[0], dataset_shape_x[1]
                ),
                batch_size=BATCH_SIZE,
            )
            ingest_in_tiledb_sparse(
                uri=tiledb_uri_y,
                data=create_sparse_array_one_hot_2d(
                    dataset_shape_y[0], dataset_shape_y[1]
                ),
                batch_size=BATCH_SIZE,
            )

            with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:
                with self.assertRaises(Exception):
                    PyTorchTileDBSparseDataset(
                        x_array=x, y_array=y, batch_size=BATCH_SIZE
                    )

    # This test is practically skipped but is linked to the TODO#1
    def test_sparse_no_duplicates_with_multiple_workers(self):
        for workers in NUM_OF_WORKERS[2:]:
            with self.subTest():
                with tempfile.TemporaryDirectory() as tiledb_uri_x, tempfile.TemporaryDirectory() as tiledb_uri_y:

                    dataset_shape_x = (ROWS,) + INPUT_SHAPES[1]
                    dataset_shape_y = (ROWS, (NUM_OF_CLASSES,))

                    ingest_in_tiledb_sparse(
                        uri=tiledb_uri_x,
                        data=create_sparse_array_one_hot_2d(
                            dataset_shape_x[0], dataset_shape_x[1]
                        ),
                        batch_size=BATCH_SIZE,
                    )
                    ingest_in_tiledb_sparse(
                        uri=tiledb_uri_y,
                        data=create_sparse_array_one_hot_2d(
                            dataset_shape_y[0], dataset_shape_y[1]
                        ),
                        batch_size=BATCH_SIZE,
                    )

                    with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:

                        tiledb_dataset = PyTorchTileDBSparseDataset(
                            x_array=x, y_array=y, batch_size=BATCH_SIZE
                        )

                        self.assertIsInstance(
                            tiledb_dataset, torch.utils.data.IterableDataset
                        )

                        train_loader = torch.utils.data.DataLoader(
                            tiledb_dataset, batch_size=None, num_workers=workers
                        )

                        unique_inputs = []
                        unique_labels = []

                        for batchindx, data in enumerate(train_loader):
                            # Keep unique X tensors
                            if not any(
                                np.array_equal(data[0].numpy(), unique_input)
                                for unique_input in unique_inputs
                            ):
                                unique_inputs.append(data[0].numpy())

                            # Keep unique Y tensors
                            if not any(
                                np.array_equal(data[1].numpy(), unique_label)
                                for unique_label in unique_labels
                            ):
                                unique_labels.append(data[1].numpy())

                        self.assertEqual(len(unique_inputs) - 1, batchindx)
                        self.assertEqual(len(unique_labels) - 1, batchindx)
