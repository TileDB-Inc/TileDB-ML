"""Tests for TileDB integration with PyTorch Data API for Dense Arrays."""

import torch
import tiledb
import numpy as np
import unittest
import tempfile

import torch.nn as nn
import torch.optim as optim

from tiledb.ml.data_apis.pytorch import PyTorchTileDBDenseDataset

# Test parameters
NUM_OF_CLASSES = 5
BATCH_SIZE = 20
ROWS = 1000

# We test for 2d, 3d, 4d and 5d data
INPUT_SHAPES = [(10,), (10, 3), (10, 10, 3), (10, 10, 10, 3)]

# We test for single and multiple workers
NUM_OF_WORKERS = [1, 2, 3, 4, 5]


def ingest_in_tiledb(data: np.array, batch_size: int, uri: str):
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
        sparse=False,
        attrs=[tiledb.Attr(name="features", dtype=np.float32)],
    )
    # Create array
    tiledb.Array.create(uri, schema)

    # Ingest
    with tiledb.open(uri, "w") as tiledb_array:
        tiledb_array[:] = {"features": data}


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


class TestTileDBDensePyTorchDataloaderAPI(unittest.TestCase):
    def test_tiledb_pytorch_data_api_train_with_multiple_dim_data(self):
        for input_shape in INPUT_SHAPES:
            for workers in NUM_OF_WORKERS:
                with self.subTest():
                    with tempfile.TemporaryDirectory() as tiledb_uri_x, tempfile.TemporaryDirectory() as tiledb_uri_y:

                        dataset_shape_x = (ROWS,) + input_shape
                        dataset_shape_y = (ROWS,)

                        ingest_in_tiledb(
                            uri=tiledb_uri_x,
                            data=np.random.rand(*dataset_shape_x),
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

                            tiledb_dataset = PyTorchTileDBDenseDataset(
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
                            criterion = nn.CrossEntropyLoss()
                            optimizer = optim.SGD(
                                net.parameters(), lr=0.001, momentum=0.9
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

    def test_except_with_diff_number_of_x_y_rows(self):
        with tempfile.TemporaryDirectory() as tiledb_uri_x, tempfile.TemporaryDirectory() as tiledb_uri_y:
            # Add one extra row on X
            dataset_shape_x = (ROWS + 1,) + INPUT_SHAPES[0]
            dataset_shape_y = (ROWS, NUM_OF_CLASSES)

            ingest_in_tiledb(
                uri=tiledb_uri_x,
                data=np.random.rand(*dataset_shape_x),
                batch_size=BATCH_SIZE,
            )
            ingest_in_tiledb(
                uri=tiledb_uri_y,
                data=np.random.rand(*dataset_shape_y),
                batch_size=BATCH_SIZE,
            )

            with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:
                with self.assertRaises(Exception):
                    PyTorchTileDBDenseDataset(
                        x_array=x, y_array=y, batch_size=BATCH_SIZE
                    )

    def test_no_duplicates_with_multiple_workers(self):
        for workers in NUM_OF_WORKERS[2:]:
            with self.subTest():
                with tempfile.TemporaryDirectory() as tiledb_uri_x, tempfile.TemporaryDirectory() as tiledb_uri_y:

                    dataset_shape_x = (ROWS,) + INPUT_SHAPES[1]
                    dataset_shape_y = (ROWS,)

                    ingest_in_tiledb(
                        uri=tiledb_uri_x,
                        data=np.random.rand(*dataset_shape_x),
                        batch_size=BATCH_SIZE,
                    )
                    ingest_in_tiledb(
                        uri=tiledb_uri_y,
                        data=np.random.randint(
                            low=0, high=NUM_OF_CLASSES, size=dataset_shape_y
                        ),
                        batch_size=BATCH_SIZE,
                    )

                    with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:

                        tiledb_dataset = PyTorchTileDBDenseDataset(
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

    def test_dataset_length(self):
        with tempfile.TemporaryDirectory() as tiledb_uri_x, tempfile.TemporaryDirectory() as tiledb_uri_y:
            dataset_shape_x = (ROWS,) + INPUT_SHAPES[1]
            dataset_shape_y = (ROWS,)

            ingest_in_tiledb(
                uri=tiledb_uri_x,
                data=np.random.rand(*dataset_shape_x),
                batch_size=BATCH_SIZE,
            )
            ingest_in_tiledb(
                uri=tiledb_uri_y,
                data=np.random.randint(
                    low=0, high=NUM_OF_CLASSES, size=dataset_shape_y
                ),
                batch_size=BATCH_SIZE,
            )

            with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:
                tiledb_dataset = PyTorchTileDBDenseDataset(
                    x_array=x, y_array=y, batch_size=BATCH_SIZE
                )

                self.assertEqual(len(tiledb_dataset), ROWS)
