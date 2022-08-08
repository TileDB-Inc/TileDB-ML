"""Tests for TileDB integration with Tensorflow Data API."""

import numpy as np
import pytest
import tensorflow as tf

from tiledb.ml.readers.tensorflow import TensorflowTileDBDataset

from .utils import ingest_in_tiledb, parametrize_for_dataset, validate_tensor_generator


def dataset_batching_shuffling(dataset: tf.data.Dataset, batch_size: int, shuffle_buffer_size: int) -> tf.data.Dataset:
    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(shuffle_buffer_size)
    return dataset.batch(batch_size)


class TestTensorflowTileDBDataset:
    @parametrize_for_dataset()
    def test_dataset(
        self, tmpdir, x_spec, y_spec, batch_size, shuffle_buffer_size, num_workers
    ):
        with ingest_in_tiledb(tmpdir, x_spec) as (x_params, x_data):
            with ingest_in_tiledb(tmpdir, y_spec) as (y_params, y_data):
                dataset = TensorflowTileDBDataset(
                    x_params,
                    y_params,
                    num_workers=num_workers,
                )
                dataset = dataset_batching_shuffling(
                    dataset=dataset,
                    batch_size=batch_size,
                    shuffle_buffer_size=shuffle_buffer_size,
                )
                assert isinstance(dataset, tf.data.Dataset)
                validate_tensor_generator(
                    dataset, x_spec, y_spec, batch_size, supports_csr=False
                )

    @parametrize_for_dataset(
        # Add one extra key on X
        x_shape=((108, 10), (108, 10, 3)),
        y_shape=((107, 5), (107, 5, 2)),
    )
    def test_unequal_num_keys(
        self, tmpdir, x_spec, y_spec, batch_size, shuffle_buffer_size, num_workers
    ):
        with ingest_in_tiledb(tmpdir, x_spec) as (x_params, x_data):
            with ingest_in_tiledb(tmpdir, y_spec) as (y_params, y_data):
                with pytest.raises(ValueError) as ex:
                    TensorflowTileDBDataset(
                        x_params,
                        y_params,
                        num_workers=num_workers,
                    )
                assert "All arrays must have the same key range" in str(ex.value)

    @parametrize_for_dataset(num_fields=[0], shuffle_buffer_size=[0], num_workers=[0])
    def test_dataset_order(
        self, tmpdir, x_spec, y_spec, batch_size, shuffle_buffer_size, num_workers
    ):
        """Test we can read the data in the same order as written.

        The order is guaranteed only for sequential processing (num_workers=0) and
        no shuffling (shuffle_buffer_size=0).
        """
        with ingest_in_tiledb(tmpdir, x_spec) as (x_params, x_data):
            with ingest_in_tiledb(tmpdir, y_spec) as (y_params, y_data):
                dataset = TensorflowTileDBDataset(
                    x_params,
                    y_params,
                    num_workers=num_workers,
                )
                dataset = dataset_batching_shuffling(
                    dataset=dataset,
                    batch_size=batch_size,
                    shuffle_buffer_size=shuffle_buffer_size,
                )
                # since num_fields is 0, fields are all the array attributes of each array
                # the first item of each batch corresponds to the first attribute (="data")
                x_batch_tensors, y_batch_tensors = [], []
                for x_tensors, y_tensors in dataset:
                    x_batch_tensors.append(x_tensors[0])
                    y_batch_tensors.append(y_tensors[0])
                assert_tensors_almost_equal_array(x_batch_tensors, x_data)
                assert_tensors_almost_equal_array(y_batch_tensors, y_data)


def assert_tensors_almost_equal_array(batch_tensors, array):
    if isinstance(batch_tensors[0], tf.RaggedTensor):
        # compare each ragged tensor row with the non-zero values of the respective array row
        tensors = [tensor for batch_tensor in batch_tensors for tensor in batch_tensor]
        assert len(tensors) == len(array)
        for tensor_row, row in zip(tensors, array):
            np.testing.assert_array_almost_equal(tensor_row, row[np.nonzero(row)])
    else:
        if isinstance(batch_tensors[0], tf.SparseTensor):
            batch_tensors = list(map(tf.sparse.to_dense, batch_tensors))
        np.testing.assert_array_almost_equal(np.concatenate(batch_tensors), array)
