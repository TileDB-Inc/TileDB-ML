"""Tests for TileDB integration with PyTorch Data API for Dense Arrays."""

import numpy as np
import pytest
import torch

import tiledb
from tiledb.ml.readers.pytorch import PyTorchTileDBDataset

from .utils import create_rand_labels, ingest_in_tiledb

# Test parameters
NUM_OF_CLASSES = 5
BATCH_SIZE = 20
ROWS = 100


@pytest.mark.parametrize("input_shape", [(10,), (10, 3), (10, 10, 3)])
@pytest.mark.parametrize("num_attrs", [1, 2, 3])
@pytest.mark.parametrize("batch_shuffle", [True, False])
@pytest.mark.parametrize("within_batch_shuffle", [True, False])
@pytest.mark.parametrize("buffer_size", [50, None])
class TestPytorchDenseDataloader:
    def test_data_api_train_with_multiple_dim_data(
        self,
        tmpdir,
        input_shape,
        num_attrs,
        batch_shuffle,
        within_batch_shuffle,
        buffer_size,
    ):
        uri_x, uri_y = ingest_in_tiledb(
            tmpdir,
            data_x=np.random.rand(ROWS, *input_shape),
            data_y=create_rand_labels(ROWS, NUM_OF_CLASSES),
            sparse_x=False,
            sparse_y=False,
            batch_size=BATCH_SIZE,
            num_attrs=num_attrs,
        )
        attrs = [f"features_{attr}" for attr in range(num_attrs)]
        with tiledb.open(uri_x) as x, tiledb.open(uri_y) as y:
            for pass_attrs in True, False:
                dataset = PyTorchTileDBDataset(
                    x_array=x,
                    y_array=y,
                    batch_size=BATCH_SIZE,
                    buffer_size=buffer_size,
                    batch_shuffle=batch_shuffle,
                    within_batch_shuffle=within_batch_shuffle,
                    x_attrs=attrs if pass_attrs else [],
                    y_attrs=attrs if pass_attrs else [],
                )
                assert isinstance(dataset, torch.utils.data.IterableDataset)

    def test_except_with_diff_number_of_x_y_rows(
        self,
        tmpdir,
        input_shape,
        num_attrs,
        batch_shuffle,
        within_batch_shuffle,
        buffer_size,
    ):
        uri_x, uri_y = ingest_in_tiledb(
            tmpdir,
            # Add one extra row on X
            data_x=np.random.rand(ROWS + 1, *input_shape),
            sparse_x=False,
            data_y=np.random.rand(ROWS, NUM_OF_CLASSES),
            sparse_y=False,
            batch_size=BATCH_SIZE,
            num_attrs=num_attrs,
        )
        attrs = [f"features_{attr}" for attr in range(num_attrs)]
        with tiledb.open(uri_x) as x, tiledb.open(uri_y) as y:
            for pass_attrs in True, False:
                with pytest.raises(ValueError):
                    PyTorchTileDBDataset(
                        x_array=x,
                        y_array=y,
                        batch_size=BATCH_SIZE,
                        buffer_size=buffer_size,
                        batch_shuffle=batch_shuffle,
                        within_batch_shuffle=within_batch_shuffle,
                        x_attrs=attrs if pass_attrs else [],
                        y_attrs=attrs if pass_attrs else [],
                    )

    @pytest.mark.parametrize("workers", [1, 2, 3])
    def test_dataset_generator_batch_output(
        self,
        tmpdir,
        input_shape,
        workers,
        num_attrs,
        batch_shuffle,
        within_batch_shuffle,
        buffer_size,
    ):
        uri_x, uri_y = ingest_in_tiledb(
            tmpdir,
            data_x=np.random.rand(ROWS, *input_shape),
            sparse_x=False,
            data_y=np.random.rand(ROWS, NUM_OF_CLASSES),
            sparse_y=False,
            batch_size=BATCH_SIZE,
            num_attrs=num_attrs,
        )
        with tiledb.open(uri_x) as x, tiledb.open(uri_y) as y:
            dataset = PyTorchTileDBDataset(
                x_array=x,
                y_array=y,
                batch_size=BATCH_SIZE,
                buffer_size=buffer_size,
                batch_shuffle=batch_shuffle,
                within_batch_shuffle=within_batch_shuffle,
            )
            assert isinstance(dataset, torch.utils.data.IterableDataset)

            train_loader = torch.utils.data.DataLoader(
                dataset, batch_size=None, num_workers=workers
            )

            for batchindx, data in enumerate(train_loader):
                assert len(data) == 2 * num_attrs

                for attr in range(num_attrs):
                    assert data[attr].shape <= (BATCH_SIZE, *input_shape)
                    assert data[num_attrs + attr].shape <= (BATCH_SIZE, NUM_OF_CLASSES)

    @pytest.mark.parametrize("workers", [1, 2, 3])
    def test_no_duplicates_with_multiple_workers(
        self,
        tmpdir,
        input_shape,
        workers,
        mocker,
        num_attrs,
        batch_shuffle,
        within_batch_shuffle,
        buffer_size,
    ):
        uri_x, uri_y = ingest_in_tiledb(
            tmpdir,
            data_x=np.random.rand(ROWS, *input_shape),
            sparse_x=False,
            data_y=create_rand_labels(ROWS, NUM_OF_CLASSES),
            sparse_y=False,
            batch_size=BATCH_SIZE,
            num_attrs=num_attrs,
        )
        with tiledb.open(uri_x) as x, tiledb.open(uri_y) as y:
            dataset = PyTorchTileDBDataset(
                x_array=x,
                y_array=y,
                batch_size=BATCH_SIZE,
                buffer_size=buffer_size,
                batch_shuffle=batch_shuffle,
                within_batch_shuffle=within_batch_shuffle,
            )
            assert isinstance(dataset, torch.utils.data.IterableDataset)

            if workers == 1:
                mocker.patch("torch.utils.data.get_worker_info", return_value=None)

            train_loader = torch.utils.data.DataLoader(
                dataset, batch_size=None, num_workers=workers
            )

            unique_inputs = []
            unique_labels = []

            for batchindx, data in enumerate(train_loader):
                # Keep unique X tensors
                for attr in range(num_attrs):
                    if not any(
                        np.array_equal(data[attr].numpy(), unique_input)
                        for unique_input in unique_inputs
                    ):
                        unique_inputs.append(data[attr].numpy())

                    # Keep unique Y tensors - Y index is attr + num_attrs following x attrs
                    if not any(
                        np.array_equal(data[attr + num_attrs].numpy(), unique_label)
                        for unique_label in unique_labels
                    ):
                        unique_labels.append(data[attr + num_attrs].numpy())

            assert len(unique_inputs) - 1 == batchindx
            assert len(unique_labels) - 1 == batchindx
