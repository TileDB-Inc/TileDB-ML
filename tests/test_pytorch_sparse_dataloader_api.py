"""Tests for TileDB integration with Pytorch sparse Data API."""

import numpy as np
import pytest
import torch

import tiledb
from tiledb.ml.readers.pytorch import PyTorchTileDBDataset

from .utils import create_sparse_array_one_hot_2d, ingest_in_tiledb

# Test parameters
NUM_OF_CLASSES = 5
BATCH_SIZE = 20
ROWS = 100

# TODO: Multiple workers require tiledb.SparseArray to be pickled hence serializable as well
# @pytest.mark.parametrize("workers", [0, 1, 2])


@pytest.mark.parametrize("input_shape", [(10,)])
@pytest.mark.parametrize("num_attrs", [1])
@pytest.mark.parametrize("batch_shuffle", [False, True])
@pytest.mark.parametrize("buffer_size", [50, None])
class TestTileDBSparsePyTorchDataloaderAPI:
    def test_sparse_data_api_with_sparse_data_sparse_label(
        self, tmpdir, input_shape, num_attrs, batch_shuffle, buffer_size
    ):
        uri_x, uri_y = ingest_in_tiledb(
            tmpdir,
            data_x=create_sparse_array_one_hot_2d(ROWS, input_shape[0]),
            data_y=create_sparse_array_one_hot_2d(ROWS, NUM_OF_CLASSES),
            sparse_x=True,
            sparse_y=True,
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
                    x_attrs=attrs if pass_attrs else [],
                    y_attrs=attrs if pass_attrs else [],
                )
                assert isinstance(dataset, torch.utils.data.IterableDataset)

    def test_sparse_data_api_with_sparse_data_dense_label(
        self, tmpdir, input_shape, num_attrs, batch_shuffle, buffer_size
    ):
        uri_x, uri_y = ingest_in_tiledb(
            tmpdir,
            data_x=create_sparse_array_one_hot_2d(ROWS, input_shape[0]),
            data_y=np.random.randint(low=0, high=NUM_OF_CLASSES, size=ROWS),
            sparse_x=True,
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
                    x_attrs=attrs if pass_attrs else [],
                    y_attrs=attrs if pass_attrs else [],
                )
                assert isinstance(dataset, torch.utils.data.IterableDataset)

    def test_sparse_data_api_with_sparse_data_diff_number_of_x_y_rows(
        self, tmpdir, input_shape, num_attrs, batch_shuffle, buffer_size
    ):
        uri_x, uri_y = ingest_in_tiledb(
            tmpdir,
            # Add one extra row on X
            data_x=create_sparse_array_one_hot_2d(ROWS + 1, input_shape[0]),
            data_y=create_sparse_array_one_hot_2d(ROWS, NUM_OF_CLASSES),
            sparse_x=True,
            sparse_y=True,
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
                        x_attrs=attrs if pass_attrs else [],
                        y_attrs=attrs if pass_attrs else [],
                    )

    def test_sparse_data_api_with_diff_number_of_batch_x_y_rows_empty_record_except(
        self, tmpdir, input_shape, num_attrs, batch_shuffle, buffer_size
    ):
        spoiled_data = create_sparse_array_one_hot_2d(ROWS, input_shape[0])
        spoiled_data[np.nonzero(spoiled_data[0])] = 0
        uri_x, uri_y = ingest_in_tiledb(
            tmpdir,
            data_x=spoiled_data,
            data_y=create_sparse_array_one_hot_2d(ROWS, NUM_OF_CLASSES),
            sparse_x=True,
            sparse_y=True,
            batch_size=BATCH_SIZE,
            num_attrs=num_attrs,
        )
        attrs = [f"features_{attr}" for attr in range(num_attrs)]
        with tiledb.open(uri_x) as x, tiledb.open(uri_y) as y:
            with pytest.raises(Exception):
                for pass_attrs in True, False:
                    dataset = PyTorchTileDBDataset(
                        x_array=x,
                        y_array=y,
                        batch_size=BATCH_SIZE,
                        buffer_size=buffer_size,
                        batch_shuffle=batch_shuffle,
                        x_attrs=attrs if pass_attrs else [],
                        y_attrs=attrs if pass_attrs else [],
                    )
                    # Exhaust iterator
                    for _ in dataset:
                        pass

    def test_sparse_sparse_label_data(
        self, tmpdir, input_shape, num_attrs, batch_shuffle, buffer_size
    ):
        uri_x, uri_y = ingest_in_tiledb(
            tmpdir,
            data_x=create_sparse_array_one_hot_2d(ROWS, input_shape[0]),
            data_y=create_sparse_array_one_hot_2d(ROWS, NUM_OF_CLASSES),
            sparse_x=True,
            sparse_y=True,
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
                    x_attrs=attrs if pass_attrs else [],
                    y_attrs=attrs if pass_attrs else [],
                )
                generated_data = next(iter(dataset))
                for attr in range(num_attrs):
                    assert generated_data[attr].layout == torch.sparse_coo
                    assert generated_data[attr + num_attrs].layout == torch.sparse_coo
                    assert generated_data[attr].size() == (BATCH_SIZE, *input_shape)
                    assert generated_data[attr + num_attrs].size() == (
                        BATCH_SIZE,
                        NUM_OF_CLASSES,
                    )

    def test_buffer_size_geq_batch_size_exception(
        self, tmpdir, input_shape, num_attrs, batch_shuffle, buffer_size
    ):
        uri_x, uri_y = ingest_in_tiledb(
            tmpdir,
            data_x=create_sparse_array_one_hot_2d(ROWS, input_shape[0]),
            data_y=create_sparse_array_one_hot_2d(ROWS, NUM_OF_CLASSES),
            sparse_x=True,
            sparse_y=True,
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
                    # Set the buffer_size less than the batch_size
                    buffer_size=BATCH_SIZE - 1,
                    batch_shuffle=batch_shuffle,
                    x_attrs=attrs if pass_attrs else [],
                    y_attrs=attrs if pass_attrs else [],
                )
                with pytest.raises(Exception) as excinfo:
                    next(iter(dataset))
                assert str(excinfo.value).startswith(
                    "Buffer size should be greater or equal to batch size"
                )
