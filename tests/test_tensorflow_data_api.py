"""Tests for TileDB integration with Tensorflow Data API."""

import os

import numpy as np
import pytest
import tensorflow as tf

import tiledb
from tiledb.ml.readers._batch_utils import tensor_generator
from tiledb.ml.readers.tensorflow import (
    TensorflowDenseBatch,
    TensorflowSparseBatch,
    TensorflowTileDBDataset,
)

from .utils import (
    ingest_in_tiledb,
    parametrize_for_dataset,
    rand_array,
    validate_tensor_generator,
)

# Suppress all Tensorflow messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Test parameters
BATCH_SIZE = 32
ROWS = 1000


class TestTensorflowTileDBDataset:
    @parametrize_for_dataset()
    def test_generator(
        self,
        tmpdir,
        sparse_x,
        sparse_y,
        input_shape,
        output_shape,
        num_attrs,
        pass_attrs,
        buffer_size,
        batch_shuffle,
        within_batch_shuffle,
    ):
        data_x = rand_array(ROWS, *input_shape, sparse=sparse_x)
        data_y = rand_array(ROWS, *output_shape, sparse=sparse_y)
        uri_x, uri_y = ingest_in_tiledb(
            tmpdir,
            data_x=data_x,
            data_y=data_y,
            sparse_x=sparse_x,
            sparse_y=sparse_y,
            batch_size=BATCH_SIZE,
            num_attrs=num_attrs,
        )
        attrs = [f"features_{attr}" for attr in range(num_attrs)]
        with tiledb.open(uri_x) as x, tiledb.open(uri_y) as y:
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
            dataset = TensorflowTileDBDataset(**kwargs)
            assert isinstance(dataset, tf.data.Dataset)
            # Test the generator twice: once with the public api (TensorflowTileDBDataset)
            # and once with calling tensor_generator directly. Although the former calls
            # the latter internally, it is not reported as covered by the coverage report
            # due to https://github.com/tensorflow/tensorflow/issues/33759
            generators = [
                dataset,
                tensor_generator(
                    dense_batch_cls=TensorflowDenseBatch,
                    sparse_batch_cls=TensorflowSparseBatch,
                    **dict(kwargs, buffer_size=buffer_size or BATCH_SIZE),
                ),
            ]
            for generator in generators:
                validate_tensor_generator(
                    generator,
                    sparse_x=sparse_x,
                    sparse_y=sparse_y,
                    shape_x=input_shape,
                    shape_y=output_shape,
                    batch_size=BATCH_SIZE,
                    num_attrs=num_attrs,
                )

    @parametrize_for_dataset(buffer_size=[BATCH_SIZE - 1])
    def test_buffer_size_smaller_than_batch_size(
        self,
        tmpdir,
        sparse_x,
        sparse_y,
        input_shape,
        output_shape,
        num_attrs,
        pass_attrs,
        buffer_size,
        batch_shuffle,
        within_batch_shuffle,
    ):
        data_x = rand_array(ROWS, *input_shape, sparse=sparse_x)
        data_y = rand_array(ROWS, *output_shape, sparse=sparse_y)
        uri_x, uri_y = ingest_in_tiledb(
            tmpdir,
            data_x=data_x,
            data_y=data_y,
            sparse_x=sparse_x,
            sparse_y=sparse_y,
            batch_size=BATCH_SIZE,
            num_attrs=num_attrs,
        )
        attrs = [f"features_{attr}" for attr in range(num_attrs)]
        with tiledb.open(uri_x) as x, tiledb.open(uri_y) as y:
            with pytest.raises(ValueError):
                TensorflowTileDBDataset(
                    x_array=x,
                    y_array=y,
                    batch_size=BATCH_SIZE,
                    buffer_size=buffer_size,
                    batch_shuffle=batch_shuffle,
                    within_batch_shuffle=within_batch_shuffle,
                    x_attrs=attrs if pass_attrs else [],
                    y_attrs=attrs if pass_attrs else [],
                )

    @parametrize_for_dataset()
    def test_unequal_num_rows(
        self,
        tmpdir,
        sparse_x,
        sparse_y,
        input_shape,
        output_shape,
        num_attrs,
        pass_attrs,
        buffer_size,
        batch_shuffle,
        within_batch_shuffle,
    ):
        # Add one extra row on X
        data_x = rand_array(ROWS + 1, *input_shape, sparse=sparse_x)
        data_y = rand_array(ROWS, *output_shape, sparse=sparse_y)
        uri_x, uri_y = ingest_in_tiledb(
            tmpdir,
            data_x=data_x,
            data_y=data_y,
            sparse_x=sparse_x,
            sparse_y=sparse_y,
            batch_size=BATCH_SIZE,
            num_attrs=num_attrs,
        )
        attrs = [f"features_{attr}" for attr in range(num_attrs)]
        with tiledb.open(uri_x) as x, tiledb.open(uri_y) as y:
            with pytest.raises(ValueError):
                TensorflowTileDBDataset(
                    x_array=x,
                    y_array=y,
                    batch_size=BATCH_SIZE,
                    buffer_size=buffer_size,
                    batch_shuffle=batch_shuffle,
                    within_batch_shuffle=within_batch_shuffle,
                    x_attrs=attrs if pass_attrs else [],
                    y_attrs=attrs if pass_attrs else [],
                )

    @parametrize_for_dataset(sparse_x=[True])
    def test_sparse_x_unequal_num_rows_in_batch(
        self,
        tmpdir,
        sparse_x,
        sparse_y,
        input_shape,
        output_shape,
        num_attrs,
        pass_attrs,
        buffer_size,
        batch_shuffle,
        within_batch_shuffle,
    ):
        data_x = rand_array(ROWS, *input_shape, sparse=sparse_x)
        data_x[np.nonzero(data_x[0])] = 0
        data_y = rand_array(ROWS, *output_shape, sparse=sparse_y)
        uri_x, uri_y = ingest_in_tiledb(
            tmpdir,
            data_x=data_x,
            data_y=data_y,
            sparse_x=sparse_x,
            sparse_y=sparse_y,
            batch_size=BATCH_SIZE,
            num_attrs=num_attrs,
        )
        attrs = [f"features_{attr}" for attr in range(num_attrs)]
        with tiledb.open(uri_x) as x, tiledb.open(uri_y) as y:
            dataset = TensorflowTileDBDataset(
                x_array=x,
                y_array=y,
                batch_size=BATCH_SIZE,
                buffer_size=buffer_size,
                batch_shuffle=batch_shuffle,
                within_batch_shuffle=within_batch_shuffle,
                x_attrs=attrs if pass_attrs else [],
                y_attrs=attrs if pass_attrs else [],
            )
            with pytest.raises(Exception):
                for _ in dataset:
                    pass
