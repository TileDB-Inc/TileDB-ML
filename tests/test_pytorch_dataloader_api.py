"""Tests for TileDB integration with PyTorch Data API for Dense Arrays."""

import os

import numpy as np
import pytest
import torch

import tiledb
from tiledb.ml.readers.pytorch import PyTorchTileDBDenseDataset

from .utils import ingest_in_tiledb

# Test parameters
NUM_OF_CLASSES = 5
BATCH_SIZE = 20
ROWS = 100


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
    [1, 2, 3],
)
@pytest.mark.parametrize(
    "batch_shuffle",
    [True, False],
)
@pytest.mark.parametrize(
    "within_batch_shuffle",
    [True, False],
)
@pytest.mark.parametrize(
    "buffer_size",
    [50, None],
)
class TestPytorchDenseDataloader:
    def test_tiledb_pytorch_data_api_train_with_multiple_dim_data(
        self,
        tmpdir,
        input_shape,
        workers,
        num_of_attributes,
        batch_shuffle,
        within_batch_shuffle,
        buffer_size,
    ):
        tiledb_uri_x = os.path.join(tmpdir, "x")
        tiledb_uri_y = os.path.join(tmpdir, "y")

        ingest_in_tiledb(
            uri=tiledb_uri_x,
            data=np.random.rand(ROWS, *input_shape),
            batch_size=BATCH_SIZE,
            sparse=False,
            num_of_attributes=num_of_attributes,
        )
        ingest_in_tiledb(
            uri=tiledb_uri_y,
            data=np.random.randint(low=0, high=NUM_OF_CLASSES, size=ROWS),
            batch_size=BATCH_SIZE,
            sparse=False,
            num_of_attributes=num_of_attributes,
        )

        with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:
            tiledb_dataset = PyTorchTileDBDenseDataset(
                x_array=x,
                y_array=y,
                batch_size=BATCH_SIZE,
                buffer_size=buffer_size,
                batch_shuffle=batch_shuffle,
                within_batch_shuffle=within_batch_shuffle,
                x_attribute_names=[
                    "features_" + str(attr) for attr in range(num_of_attributes)
                ],
                y_attribute_names=[
                    "features_" + str(attr) for attr in range(num_of_attributes)
                ],
            )
            assert isinstance(tiledb_dataset, torch.utils.data.IterableDataset)

            # Same test without attribute names explicitly provided by the user
            tiledb_dataset = PyTorchTileDBDenseDataset(
                x_array=x,
                y_array=y,
                batch_size=BATCH_SIZE,
                buffer_size=buffer_size,
                batch_shuffle=batch_shuffle,
                within_batch_shuffle=within_batch_shuffle,
            )

            assert isinstance(tiledb_dataset, torch.utils.data.IterableDataset)

    def test_except_with_diff_number_of_x_y_rows(
        self,
        tmpdir,
        input_shape,
        workers,
        num_of_attributes,
        batch_shuffle,
        within_batch_shuffle,
        buffer_size,
    ):
        tiledb_uri_x = os.path.join(tmpdir, "x")
        tiledb_uri_y = os.path.join(tmpdir, "y")

        ingest_in_tiledb(
            uri=tiledb_uri_x,
            # Add one extra row on X
            data=np.random.rand(ROWS + 1, *input_shape),
            batch_size=BATCH_SIZE,
            sparse=False,
            num_of_attributes=num_of_attributes,
        )
        ingest_in_tiledb(
            uri=tiledb_uri_y,
            data=np.random.rand(ROWS, NUM_OF_CLASSES),
            batch_size=BATCH_SIZE,
            sparse=False,
            num_of_attributes=num_of_attributes,
        )

        with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:
            with pytest.raises(ValueError):
                PyTorchTileDBDenseDataset(
                    x_array=x,
                    y_array=y,
                    batch_size=BATCH_SIZE,
                    buffer_size=buffer_size,
                    batch_shuffle=batch_shuffle,
                    within_batch_shuffle=within_batch_shuffle,
                    x_attribute_names=[
                        "features_" + str(attr) for attr in range(num_of_attributes)
                    ],
                    y_attribute_names=[
                        "features_" + str(attr) for attr in range(num_of_attributes)
                    ],
                )

    def test_dataset_length(
        self,
        tmpdir,
        input_shape,
        workers,
        num_of_attributes,
        batch_shuffle,
        within_batch_shuffle,
        buffer_size,
    ):
        tiledb_uri_x = os.path.join(tmpdir, "x")
        tiledb_uri_y = os.path.join(tmpdir, "y")

        ingest_in_tiledb(
            uri=tiledb_uri_x,
            data=np.random.rand(ROWS, *input_shape),
            batch_size=BATCH_SIZE,
            sparse=False,
            num_of_attributes=num_of_attributes,
        )
        ingest_in_tiledb(
            uri=tiledb_uri_y,
            data=np.random.randint(low=0, high=NUM_OF_CLASSES, size=ROWS),
            batch_size=BATCH_SIZE,
            sparse=False,
            num_of_attributes=num_of_attributes,
        )

        with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:
            tiledb_dataset = PyTorchTileDBDenseDataset(
                x_array=x,
                y_array=y,
                batch_size=BATCH_SIZE,
                buffer_size=buffer_size,
                batch_shuffle=batch_shuffle,
                within_batch_shuffle=within_batch_shuffle,
                x_attribute_names=[
                    "features_" + str(attr) for attr in range(num_of_attributes)
                ],
                y_attribute_names=[
                    "features_" + str(attr) for attr in range(num_of_attributes)
                ],
            )

            assert len(tiledb_dataset) == ROWS

            # Same test without attribute names explicitly provided by the user
            tiledb_dataset = PyTorchTileDBDenseDataset(
                x_array=x,
                y_array=y,
                batch_size=BATCH_SIZE,
                buffer_size=buffer_size,
                batch_shuffle=batch_shuffle,
                within_batch_shuffle=within_batch_shuffle,
            )

            assert len(tiledb_dataset) == ROWS

    def test_dataset_generator_batch_output(
        self,
        tmpdir,
        input_shape,
        workers,
        mocker,
        num_of_attributes,
        batch_shuffle,
        within_batch_shuffle,
        buffer_size,
    ):
        tiledb_uri_x = os.path.join(tmpdir, "x")
        tiledb_uri_y = os.path.join(tmpdir, "y")

        ingest_in_tiledb(
            uri=tiledb_uri_x,
            data=np.random.rand(ROWS, *input_shape[1:]),
            batch_size=BATCH_SIZE,
            sparse=False,
            num_of_attributes=num_of_attributes,
        )
        ingest_in_tiledb(
            uri=tiledb_uri_y,
            data=np.random.rand(ROWS, NUM_OF_CLASSES),
            batch_size=BATCH_SIZE,
            sparse=False,
            num_of_attributes=num_of_attributes,
        )

        with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:
            tiledb_dataset = PyTorchTileDBDenseDataset(
                x_array=x,
                y_array=y,
                batch_size=BATCH_SIZE,
                buffer_size=buffer_size,
                batch_shuffle=batch_shuffle,
                within_batch_shuffle=within_batch_shuffle,
            )

            assert isinstance(tiledb_dataset, torch.utils.data.IterableDataset)

            train_loader = torch.utils.data.DataLoader(
                tiledb_dataset, batch_size=None, num_workers=workers
            )

            for batchindx, data in enumerate(train_loader):
                assert len(data) == 2 * num_of_attributes

                for attr in range(num_of_attributes):
                    assert data[attr].shape <= (
                        BATCH_SIZE,
                        *input_shape[1:],
                    )
                    assert data[num_of_attributes + attr].shape <= (
                        BATCH_SIZE,
                        NUM_OF_CLASSES,
                    )

    def test_no_duplicates_with_multiple_workers(
        self,
        tmpdir,
        input_shape,
        workers,
        mocker,
        num_of_attributes,
        batch_shuffle,
        within_batch_shuffle,
        buffer_size,
    ):

        tiledb_uri_x = os.path.join(tmpdir, "x")
        tiledb_uri_y = os.path.join(tmpdir, "y")

        ingest_in_tiledb(
            uri=tiledb_uri_x,
            data=np.random.rand(ROWS, *input_shape),
            batch_size=BATCH_SIZE,
            sparse=False,
            num_of_attributes=num_of_attributes,
        )
        ingest_in_tiledb(
            uri=tiledb_uri_y,
            data=np.random.randint(low=0, high=NUM_OF_CLASSES, size=ROWS),
            batch_size=BATCH_SIZE,
            sparse=False,
            num_of_attributes=num_of_attributes,
        )

        with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:

            tiledb_dataset = PyTorchTileDBDenseDataset(
                x_array=x,
                y_array=y,
                batch_size=BATCH_SIZE,
                buffer_size=buffer_size,
                batch_shuffle=batch_shuffle,
                within_batch_shuffle=within_batch_shuffle,
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
                for attr in range(num_of_attributes):
                    if not any(
                        np.array_equal(data[attr].numpy(), unique_input)
                        for unique_input in unique_inputs
                    ):
                        unique_inputs.append(data[attr].numpy())

                    # Keep unique Y tensors - Y index is attr + num_of_attributes following x attrs
                    if not any(
                        np.array_equal(
                            data[attr + num_of_attributes].numpy(), unique_label
                        )
                        for unique_label in unique_labels
                    ):
                        unique_labels.append(data[attr + num_of_attributes].numpy())

            assert len(unique_inputs) - 1 == batchindx
            assert len(unique_labels) - 1 == batchindx
