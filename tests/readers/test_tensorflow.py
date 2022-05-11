"""Tests for TileDB integration with Tensorflow Data API."""

import numpy as np
import pytest
import tensorflow as tf

from tiledb.ml.readers.tensorflow import TensorflowTileDBDataset

from .utils import ingest_in_tiledb, parametrize_for_dataset, validate_tensor_generator


class TestTensorflowTileDBDataset:
    @parametrize_for_dataset()
    def test_dataset(
        self,
        tmpdir,
        x_shape,
        y_shape,
        x_sparse,
        y_sparse,
        num_attrs,
        pass_attrs,
        buffer_bytes,
        batch_size,
        shuffle_buffer_size,
        num_workers,
    ):
        with ingest_in_tiledb(
            tmpdir, x_shape, x_sparse, num_attrs, pass_attrs
        ) as x_kwargs, ingest_in_tiledb(
            tmpdir, y_shape, y_sparse, num_attrs, pass_attrs
        ) as y_kwargs:
            dataset = TensorflowTileDBDataset(
                x_array=x_kwargs["array"],
                y_array=y_kwargs["array"],
                x_attrs=x_kwargs["attrs"],
                y_attrs=y_kwargs["attrs"],
                buffer_bytes=buffer_bytes,
                batch_size=batch_size,
                shuffle_buffer_size=shuffle_buffer_size,
                num_workers=num_workers,
            )
            assert isinstance(dataset, tf.data.Dataset)
            validate_tensor_generator(
                dataset, num_attrs, x_sparse, y_sparse, x_shape, y_shape, batch_size
            )

    @parametrize_for_dataset(
        # Add one extra row on X
        x_shape=((108, 10), (108, 10, 3)),
        y_shape=((107, 5), (107, 5, 2)),
    )
    def test_unequal_num_rows(
        self,
        tmpdir,
        x_shape,
        y_shape,
        x_sparse,
        y_sparse,
        num_attrs,
        pass_attrs,
        buffer_bytes,
        batch_size,
        shuffle_buffer_size,
        num_workers,
    ):
        with ingest_in_tiledb(
            tmpdir, x_shape, x_sparse, num_attrs, pass_attrs
        ) as x_kwargs, ingest_in_tiledb(
            tmpdir, y_shape, y_sparse, num_attrs, pass_attrs
        ) as y_kwargs:
            with pytest.raises(ValueError) as ex:
                TensorflowTileDBDataset(
                    x_array=x_kwargs["array"],
                    y_array=y_kwargs["array"],
                    x_attrs=x_kwargs["attrs"],
                    y_attrs=y_kwargs["attrs"],
                    buffer_bytes=buffer_bytes,
                    batch_size=batch_size,
                    shuffle_buffer_size=shuffle_buffer_size,
                    num_workers=num_workers,
                )
            assert "X and Y arrays must have the same number of rows" in str(ex.value)

    @parametrize_for_dataset(x_sparse=[True], shuffle_buffer_size=[0], num_workers=[0])
    def test_sparse_read_order(
        self,
        tmpdir,
        x_shape,
        y_shape,
        x_sparse,
        y_sparse,
        num_attrs,
        pass_attrs,
        buffer_bytes,
        batch_size,
        shuffle_buffer_size,
        num_workers,
    ):
        with ingest_in_tiledb(
            tmpdir, x_shape, x_sparse, num_attrs, pass_attrs
        ) as x_kwargs, ingest_in_tiledb(
            tmpdir, y_shape, y_sparse, num_attrs, pass_attrs
        ) as y_kwargs:
            dataset = TensorflowTileDBDataset(
                x_array=x_kwargs["array"],
                y_array=y_kwargs["array"],
                x_attrs=x_kwargs["attrs"],
                y_attrs=y_kwargs["attrs"],
                buffer_bytes=buffer_bytes,
                batch_size=batch_size,
                shuffle_buffer_size=shuffle_buffer_size,
                num_workers=num_workers,
            )
            generated_x_data = np.concatenate(
                [
                    tf.sparse.to_dense(
                        tf.sparse.reorder(x_tensors if num_attrs == 1 else x_tensors[0])
                    )
                    for x_tensors, y_tensors in dataset
                ]
            )
            np.testing.assert_array_almost_equal(generated_x_data, x_kwargs["data"])
