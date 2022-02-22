"""Tests for TileDB integration with Tensorflow Data API."""

import os

import numpy as np
import pytest
import tensorflow as tf

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


@pytest.mark.parametrize("num_rows", [107])
class TestTensorflowTileDBDataset:
    @parametrize_for_dataset()
    def test_generator(
        self,
        tmpdir,
        num_rows,
        x_sparse,
        y_sparse,
        x_shape,
        y_shape,
        num_attrs,
        pass_attrs,
        batch_size,
        buffer_size,
        batch_shuffle,
        within_batch_shuffle,
    ):
        with ingest_in_tiledb(
            tmpdir,
            x_data=rand_array(num_rows, *x_shape, sparse=x_sparse),
            y_data=rand_array(num_rows, *y_shape, sparse=y_sparse),
            x_sparse=x_sparse,
            y_sparse=y_sparse,
            batch_size=batch_size,
            num_attrs=num_attrs,
            pass_attrs=pass_attrs,
            buffer_size=buffer_size,
            batch_shuffle=batch_shuffle,
            within_batch_shuffle=within_batch_shuffle,
        ) as dataset_kwargs:
            dataset = TensorflowTileDBDataset(**dataset_kwargs)
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
                    **dict(dataset_kwargs, buffer_size=buffer_size or batch_size),
                ),
            ]
            for generator in generators:
                validate_tensor_generator(
                    generator,
                    x_sparse=x_sparse,
                    y_sparse=y_sparse,
                    x_shape=x_shape,
                    y_shape=y_shape,
                    batch_size=batch_size,
                    num_attrs=num_attrs,
                )

    @parametrize_for_dataset(batch_size=[32], buffer_size=[31])
    def test_buffer_size_smaller_than_batch_size(
        self,
        tmpdir,
        num_rows,
        x_sparse,
        y_sparse,
        x_shape,
        y_shape,
        num_attrs,
        pass_attrs,
        batch_size,
        buffer_size,
        batch_shuffle,
        within_batch_shuffle,
    ):
        with ingest_in_tiledb(
            tmpdir,
            x_data=rand_array(num_rows, *x_shape, sparse=x_sparse),
            y_data=rand_array(num_rows, *y_shape, sparse=y_sparse),
            x_sparse=x_sparse,
            y_sparse=y_sparse,
            batch_size=batch_size,
            num_attrs=num_attrs,
            pass_attrs=pass_attrs,
            buffer_size=buffer_size,
            batch_shuffle=batch_shuffle,
            within_batch_shuffle=within_batch_shuffle,
        ) as dataset_kwargs:
            with pytest.raises(ValueError) as ex:
                TensorflowTileDBDataset(**dataset_kwargs)
            assert "buffer_size must be >= batch_size" in str(ex.value)

    @parametrize_for_dataset()
    def test_unequal_num_rows(
        self,
        tmpdir,
        num_rows,
        x_sparse,
        y_sparse,
        x_shape,
        y_shape,
        num_attrs,
        pass_attrs,
        batch_size,
        buffer_size,
        batch_shuffle,
        within_batch_shuffle,
    ):
        with ingest_in_tiledb(
            tmpdir,
            # Add one extra row on X
            x_data=rand_array(num_rows + 1, *x_shape, sparse=x_sparse),
            y_data=rand_array(num_rows, *y_shape, sparse=y_sparse),
            x_sparse=x_sparse,
            y_sparse=y_sparse,
            batch_size=batch_size,
            num_attrs=num_attrs,
            pass_attrs=pass_attrs,
            buffer_size=buffer_size,
            batch_shuffle=batch_shuffle,
            within_batch_shuffle=within_batch_shuffle,
        ) as dataset_kwargs:
            with pytest.raises(ValueError) as ex:
                TensorflowTileDBDataset(**dataset_kwargs)
            assert "X and Y arrays must have the same number of rows" in str(ex.value)

    @parametrize_for_dataset(x_sparse=[True])
    def test_x_sparse_unequal_num_rows_in_batch(
        self,
        tmpdir,
        num_rows,
        x_sparse,
        y_sparse,
        x_shape,
        y_shape,
        num_attrs,
        pass_attrs,
        batch_size,
        buffer_size,
        batch_shuffle,
        within_batch_shuffle,
    ):
        x_data = rand_array(num_rows, *x_shape, sparse=x_sparse)
        x_data[np.random.randint(len(x_data))] = 0
        with ingest_in_tiledb(
            tmpdir,
            x_data=x_data,
            y_data=rand_array(num_rows, *y_shape, sparse=y_sparse),
            x_sparse=x_sparse,
            y_sparse=y_sparse,
            batch_size=batch_size,
            num_attrs=num_attrs,
            pass_attrs=pass_attrs,
            buffer_size=buffer_size,
            batch_shuffle=batch_shuffle,
            within_batch_shuffle=within_batch_shuffle,
        ) as dataset_kwargs:
            dataset = TensorflowTileDBDataset(**dataset_kwargs)
            with pytest.raises(tf.errors.InvalidArgumentError) as ex:
                for _ in dataset:
                    pass
            assert "x and y batches should have the same length" in str(ex.value)
