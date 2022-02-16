"""Tests for TileDB integration with Pytorch sparse Data API."""

import numpy as np
import pytest
import torch

import tiledb
from tiledb.ml.readers.pytorch import PyTorchTileDBDataset

from .utils import create_rand_labels, ingest_in_tiledb, validate_tensor_generator

# Test parameters
NUM_OF_FEATURES = 10
NUM_OF_CLASSES = 5
BATCH_SIZE = 20
ROWS = 100

# TODO: Multiple workers require tiledb.SparseArray to be pickled hence serializable as well
# @pytest.mark.parametrize("workers", [0, 1, 2])


@pytest.mark.parametrize("num_attrs", [1])
@pytest.mark.parametrize("batch_shuffle", [False, True])
@pytest.mark.parametrize("buffer_size", [50, None])
class TestPyTorchTileDBDatasetSparse:
    @pytest.mark.parametrize("sparse_y", [True, False])
    def test_sparse_x(self, tmpdir, num_attrs, batch_shuffle, buffer_size, sparse_y):
        uri_x, uri_y = ingest_in_tiledb(
            tmpdir,
            data_x=create_rand_labels(ROWS, NUM_OF_FEATURES, one_hot=True),
            data_y=create_rand_labels(ROWS, NUM_OF_CLASSES, one_hot=sparse_y),
            sparse_x=True,
            sparse_y=sparse_y,
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
                validate_tensor_generator(
                    dataset,
                    num_attrs,
                    BATCH_SIZE,
                    shape_x=(NUM_OF_FEATURES,),
                    shape_y=(NUM_OF_CLASSES,) if sparse_y else (),
                    sparse_x=True,
                    sparse_y=sparse_y,
                )

    def test_unequal_num_rows(self, tmpdir, num_attrs, batch_shuffle, buffer_size):
        uri_x, uri_y = ingest_in_tiledb(
            tmpdir,
            # Add one extra row on X
            data_x=create_rand_labels(ROWS + 1, NUM_OF_FEATURES, one_hot=True),
            data_y=create_rand_labels(ROWS, NUM_OF_CLASSES, one_hot=True),
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

    def test_unequal_num_rows_in_batch(
        self, tmpdir, num_attrs, batch_shuffle, buffer_size
    ):
        spoiled_data = create_rand_labels(ROWS, NUM_OF_FEATURES, one_hot=True)
        spoiled_data[np.nonzero(spoiled_data[0])] = 0
        uri_x, uri_y = ingest_in_tiledb(
            tmpdir,
            data_x=spoiled_data,
            data_y=create_rand_labels(ROWS, NUM_OF_CLASSES, one_hot=True),
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
                with pytest.raises(Exception):
                    for _ in dataset:
                        pass
