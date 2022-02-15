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


@pytest.mark.parametrize("input_shape", [(10,), (10, 3)])
@pytest.mark.parametrize("num_attrs", [1, 2])
@pytest.mark.parametrize("batch_shuffle", [True, False])
@pytest.mark.parametrize("within_batch_shuffle", [True, False])
@pytest.mark.parametrize("buffer_size", [50, None])
class TestPyTorchTileDBDatasetDense:
    @pytest.mark.parametrize("workers", [0, 2])
    def test_dense_x_dense_y(
        self,
        tmpdir,
        input_shape,
        num_attrs,
        batch_shuffle,
        within_batch_shuffle,
        buffer_size,
        workers,
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
                kwargs = dict(
                    x_array=x,
                    y_array=y,
                    batch_size=BATCH_SIZE,
                    buffer_size=buffer_size,
                    batch_shuffle=batch_shuffle,
                    within_batch_shuffle=within_batch_shuffle,
                    x_attrs=attrs if pass_attrs else [],
                    y_attrs=attrs if pass_attrs else [],
                )

                # Test buffer_size < batch_size
                dataset = PyTorchTileDBDataset(
                    **dict(kwargs, buffer_size=BATCH_SIZE - 1)
                )
                with pytest.raises(Exception) as excinfo:
                    next(iter(dataset))
                assert "Buffer size should be greater or equal to batch size" in str(
                    excinfo.value
                )

                dataset = PyTorchTileDBDataset(**kwargs)
                assert isinstance(dataset, torch.utils.data.IterableDataset)

                train_loader = torch.utils.data.DataLoader(
                    dataset, batch_size=None, num_workers=workers
                )
                unique_inputs = []
                unique_labels = []
                for batchindx, data in enumerate(train_loader):
                    assert len(data) == 2 * num_attrs

                    for attr in range(num_attrs):
                        assert data[attr].shape <= (BATCH_SIZE, *input_shape)
                        assert data[num_attrs + attr].shape <= (
                            BATCH_SIZE,
                            NUM_OF_CLASSES,
                        )
                        # Keep unique X tensors
                        if not any(
                            np.array_equal(data[attr].numpy(), unique_input)
                            for unique_input in unique_inputs
                        ):
                            unique_inputs.append(data[attr].numpy())
                        # Keep unique Y tensors
                        # Y index is attr + num_attrs following x attrs
                        if not any(
                            np.array_equal(data[attr + num_attrs].numpy(), unique_label)
                            for unique_label in unique_labels
                        ):
                            unique_labels.append(data[attr + num_attrs].numpy())
                assert len(unique_inputs) - 1 == batchindx
                assert len(unique_labels) - 1 == batchindx

    def test_unequal_num_rows(
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
            data_y=create_rand_labels(ROWS, NUM_OF_CLASSES),
            sparse_x=False,
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
