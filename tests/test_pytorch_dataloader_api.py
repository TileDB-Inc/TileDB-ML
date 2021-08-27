"""Tests for TileDB integration with PyTorch Data API for Dense Arrays."""

import os
import torch
import tiledb
import numpy as np
import pytest

import torch.nn as nn
import torch.optim as optim

from tiledb.ml.readers.pytorch import PyTorchTileDBDenseDataset
from tiledb.ml._utils import ingest_in_tiledb

# Test parameters
NUM_OF_CLASSES = 5
BATCH_SIZE = 20
ROWS = 100


class Net(nn.Module):
    def __init__(self, shape):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(np.product(shape), NUM_OF_CLASSES),
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
        (10, 3),
        (10, 10, 3),
    ],
)
# We test for single and multiple workers
@pytest.mark.parametrize(
    "workers",
    [1, 2, 3],
)
@pytest.mark.parametrize(
    "num_of_attributes",
    [1],
)
class TestPytorchDenseDataloader:
    def test_tiledb_pytorch_data_api_train_with_multiple_dim_data(
        self, tmpdir, input_shape, workers, num_of_attributes
    ):
        dataset_shape_x = (ROWS,) + input_shape
        dataset_shape_y = (ROWS,)

        tiledb_uri_x = os.path.join(tmpdir, "x")
        tiledb_uri_y = os.path.join(tmpdir, "y")

        ingest_in_tiledb(
            uri=tiledb_uri_x,
            data=np.random.rand(*dataset_shape_x),
            batch_size=BATCH_SIZE,
            sparse=False,
            num_of_attributes=num_of_attributes,
        )
        ingest_in_tiledb(
            uri=tiledb_uri_y,
            data=np.random.randint(low=0, high=NUM_OF_CLASSES, size=dataset_shape_y),
            batch_size=BATCH_SIZE,
            sparse=False,
            num_of_attributes=num_of_attributes,
        )

        with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:

            tiledb_dataset = PyTorchTileDBDenseDataset(
                x_array=x, y_array=y, batch_size=BATCH_SIZE
            )

            assert isinstance(tiledb_dataset, torch.utils.data.IterableDataset)

            train_loader = torch.utils.data.DataLoader(
                tiledb_dataset, batch_size=None, num_workers=workers
            )

            # Train network
            net = Net(shape=dataset_shape_x[1:])
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

            for epoch in range(1):  # loop over the dataset multiple times
                for inputs, labels in train_loader:
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward + backward + optimize
                    outputs = net(inputs)
                    loss = criterion(outputs, labels.type(torch.LongTensor))
                    loss.backward()
                    optimizer.step()

    def test_except_with_diff_number_of_x_y_rows(
        self, tmpdir, input_shape, workers, num_of_attributes
    ):
        tiledb_uri_x = os.path.join(tmpdir, "x")
        tiledb_uri_y = os.path.join(tmpdir, "y")

        # Add one extra row on X
        dataset_shape_x = (ROWS + 1,) + input_shape
        dataset_shape_y = (ROWS, NUM_OF_CLASSES)

        ingest_in_tiledb(
            uri=tiledb_uri_x,
            data=np.random.rand(*dataset_shape_x),
            batch_size=BATCH_SIZE,
            sparse=False,
            num_of_attributes=num_of_attributes,
        )
        ingest_in_tiledb(
            uri=tiledb_uri_y,
            data=np.random.rand(*dataset_shape_y),
            batch_size=BATCH_SIZE,
            sparse=False,
            num_of_attributes=num_of_attributes,
        )

        with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:
            with pytest.raises(ValueError):
                PyTorchTileDBDenseDataset(x_array=x, y_array=y, batch_size=BATCH_SIZE)

    def test_no_duplicates_with_multiple_workers(
        self, tmpdir, input_shape, workers, mocker, num_of_attributes
    ):

        tiledb_uri_x = os.path.join(tmpdir, "x")
        tiledb_uri_y = os.path.join(tmpdir, "y")

        dataset_shape_x = (ROWS,) + input_shape
        dataset_shape_y = (ROWS,)

        ingest_in_tiledb(
            uri=tiledb_uri_x,
            data=np.random.rand(*dataset_shape_x),
            batch_size=BATCH_SIZE,
            sparse=False,
            num_of_attributes=num_of_attributes,
        )
        ingest_in_tiledb(
            uri=tiledb_uri_y,
            data=np.random.randint(low=0, high=NUM_OF_CLASSES, size=dataset_shape_y),
            batch_size=BATCH_SIZE,
            sparse=False,
            num_of_attributes=num_of_attributes,
        )

        with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:

            tiledb_dataset = PyTorchTileDBDenseDataset(
                x_array=x, y_array=y, batch_size=BATCH_SIZE
            )

            assert isinstance(tiledb_dataset, torch.utils.data.IterableDataset)

            if workers == 1:
                mocker.patch("torch.utils.data.get_worker_info", return_value=None)

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

            assert len(unique_inputs) - 1 == batchindx
            assert len(unique_labels) - 1 == batchindx

    def test_dataset_length(self, tmpdir, input_shape, workers, num_of_attributes):
        tiledb_uri_x = os.path.join(tmpdir, "x")
        tiledb_uri_y = os.path.join(tmpdir, "y")

        dataset_shape_x = (ROWS,) + input_shape
        dataset_shape_y = (ROWS,)

        ingest_in_tiledb(
            uri=tiledb_uri_x,
            data=np.random.rand(*dataset_shape_x),
            batch_size=BATCH_SIZE,
            sparse=False,
            num_of_attributes=num_of_attributes,
        )
        ingest_in_tiledb(
            uri=tiledb_uri_y,
            data=np.random.randint(low=0, high=NUM_OF_CLASSES, size=dataset_shape_y),
            batch_size=BATCH_SIZE,
            sparse=False,
            num_of_attributes=num_of_attributes,
        )

        with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:
            tiledb_dataset = PyTorchTileDBDenseDataset(
                x_array=x, y_array=y, batch_size=BATCH_SIZE
            )

            assert len(tiledb_dataset) == ROWS
