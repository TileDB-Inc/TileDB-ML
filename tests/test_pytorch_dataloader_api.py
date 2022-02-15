"""Tests for TileDB integration with PyTorch Data API for Dense Arrays."""

import numpy as np
import pytest
import torch

import tiledb
from tiledb.ml.readers.pytorch import PyTorchTileDBDataset

from .utils import create_rand_labels, ingest_in_tiledb, validate_tensor_generator

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
    @pytest.mark.parametrize("sparse_y", [True, False])
    def test_dense_x(
        self,
        tmpdir,
        input_shape,
        num_attrs,
        batch_shuffle,
        within_batch_shuffle,
        buffer_size,
        workers,
        sparse_y,
    ):
        if sparse_y:
            if within_batch_shuffle:
                pytest.skip("Sparse y not supported with within_batch_shuffle")
            if workers:
                pytest.skip("Sparse y not supported with multiple workers")

        uri_x, uri_y = ingest_in_tiledb(
            tmpdir,
            data_x=np.random.rand(ROWS, *input_shape),
            data_y=create_rand_labels(ROWS, NUM_OF_CLASSES, one_hot=sparse_y),
            sparse_x=False,
            sparse_y=sparse_y,
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
                validate_tensor_generator(
                    dataset,
                    num_attrs,
                    BATCH_SIZE,
                    shape_x=input_shape,
                    shape_y=(NUM_OF_CLASSES,) if sparse_y else (),
                    sparse_x=False,
                    sparse_y=sparse_y,
                )

                train_loader = torch.utils.data.DataLoader(
                    dataset, batch_size=None, num_workers=workers
                )
                unique_x_tensors = []
                unique_y_tensors = []
                for batchindx, data in enumerate(train_loader):
                    for attr in range(num_attrs):
                        # Keep unique X tensors
                        x_tensor = data[attr]
                        if not any(torch.equal(x_tensor, t) for t in unique_x_tensors):
                            unique_x_tensors.append(x_tensor)
                        # Keep unique Y tensors
                        y_tensor = data[attr + num_attrs]
                        if sparse_y:
                            y_tensor = y_tensor.to_dense()
                        if not any(torch.equal(y_tensor, t) for t in unique_y_tensors):
                            unique_y_tensors.append(y_tensor)
                    assert len(unique_x_tensors) - 1 == batchindx
                    assert len(unique_y_tensors) - 1 == batchindx

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
