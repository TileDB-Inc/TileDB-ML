"""Tests for TileDB integration with Tensorflow Data API."""

import os

import numpy as np
import pytest
import tensorflow as tf

from tiledb.ml.readers._batch_utils import tensor_generator
from tiledb.ml.readers.tensorflow import (
    TensorflowSparseTileDBTensorGenerator,
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
    def test_dataset(
        self,
        tmpdir,
        num_rows,
        x_sparse,
        y_sparse,
        x_shape,
        y_shape,
        num_attrs,
        pass_attrs,
        buffer_bytes,
        batch_size,
        shuffle_buffer_size,
    ):
        with ingest_in_tiledb(
            tmpdir,
            x_data=rand_array(num_rows, *x_shape, sparse=x_sparse),
            y_data=rand_array(num_rows, *y_shape, sparse=y_sparse),
            x_sparse=x_sparse,
            y_sparse=y_sparse,
            num_attrs=num_attrs,
            pass_attrs=pass_attrs,
        ) as kwargs:
            dataset = TensorflowTileDBDataset(
                buffer_bytes=buffer_bytes,
                batch_size=batch_size,
                shuffle_buffer_size=shuffle_buffer_size,
                **kwargs,
            )
            assert isinstance(dataset, tf.data.Dataset)
            validate_tensor_generator(
                dataset, num_attrs, x_sparse, y_sparse, x_shape, y_shape, batch_size
            )

            # Although TensorflowTileDBDataset calls tensor_generator internally, due to
            # https://github.com/tensorflow/tensorflow/issues/33759 it is not reported as
            # covered so test it explicitly.
            generator = tensor_generator(
                buffer_bytes=buffer_bytes,
                sparse_tensor_generator_cls=TensorflowSparseTileDBTensorGenerator,
                **kwargs,
            )
            # tensor_generator does not take batch_size parameter, so pass batch_size=num_rows
            validate_tensor_generator(
                generator, num_attrs, x_sparse, y_sparse, x_shape, y_shape, num_rows
            )

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
        buffer_bytes,
        batch_size,
        shuffle_buffer_size,
    ):
        with ingest_in_tiledb(
            tmpdir,
            # Add one extra row on X
            x_data=rand_array(num_rows + 1, *x_shape, sparse=x_sparse),
            y_data=rand_array(num_rows, *y_shape, sparse=y_sparse),
            x_sparse=x_sparse,
            y_sparse=y_sparse,
            num_attrs=num_attrs,
            pass_attrs=pass_attrs,
        ) as kwargs:
            with pytest.raises(ValueError) as ex:
                TensorflowTileDBDataset(
                    buffer_bytes=buffer_bytes,
                    batch_size=batch_size,
                    shuffle_buffer_size=shuffle_buffer_size,
                    **kwargs,
                )
            assert "X and Y arrays must have the same number of rows" in str(ex.value)

    @parametrize_for_dataset(x_sparse=[True], shuffle_buffer_size=[0])
    def test_sparse_read_order(
        self,
        tmpdir,
        num_rows,
        x_sparse,
        y_sparse,
        x_shape,
        y_shape,
        num_attrs,
        pass_attrs,
        buffer_bytes,
        batch_size,
        shuffle_buffer_size,
    ):
        x_data = rand_array(num_rows, *x_shape, sparse=x_sparse)
        with ingest_in_tiledb(
            tmpdir,
            x_data=x_data,
            y_data=rand_array(num_rows, *y_shape, sparse=y_sparse),
            x_sparse=x_sparse,
            y_sparse=y_sparse,
            num_attrs=num_attrs,
            pass_attrs=pass_attrs,
        ) as kwargs:
            dataset = TensorflowTileDBDataset(
                buffer_bytes=buffer_bytes,
                batch_size=batch_size,
                shuffle_buffer_size=shuffle_buffer_size,
                **kwargs,
            )
            generated_x_data = np.concatenate(
                [
                    tf.sparse.to_dense(tf.sparse.reorder(tensors[0]))
                    for tensors in dataset
                ]
            )
            np.testing.assert_array_almost_equal(generated_x_data, x_data)
