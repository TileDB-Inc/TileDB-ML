"""Tests for TileDB integration with Tensorflow Data API."""

import pytest
import tensorflow as tf

from tiledb.ml.readers.tensorflow import TensorflowTileDBDataset

from .utils import (
    assert_tensors_almost_equal_array,
    ingest_in_tiledb,
    parametrize_for_dataset,
    validate_tensor_generator,
)


def dataset_batching_shuffling(dataset, batch_size, shuffle_buffer_size):
    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(shuffle_buffer_size)
    if batch_size is not None:
        dataset = dataset.batch(batch_size)
    return dataset


class TestTensorflowTileDBDataset:
    @parametrize_for_dataset()
    def test_dataset(
        self, tmpdir, x_spec, y_spec, batch_size, shuffle_buffer_size, num_workers
    ):
        with ingest_in_tiledb(tmpdir, x_spec) as (x_params, x_data):
            x_schema = x_params.tensor_schema
            with ingest_in_tiledb(tmpdir, y_spec) as (y_params, y_data):
                y_schema = y_params.tensor_schema
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
                    dataset, x_schema, y_schema, x_spec, y_spec, batch_size
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
                    TensorflowTileDBDataset(x_params, y_params, num_workers=num_workers)
                assert "All arrays must have the same key range" in str(ex.value)

    @parametrize_for_dataset(
        num_fields=[0],
        shuffle_buffer_size=[0],
        num_workers=[0],
    )
    def test_dataset_order(
        self, tmpdir, x_spec, y_spec, batch_size, shuffle_buffer_size, num_workers
    ):
        """Test we can read the data in the same order as written.

        The order is guaranteed only for sequential processing (num_workers=0) and
        no shuffling (shuffle_buffer_size=0).
        """
        to_dense = tf.sparse.to_dense
        with ingest_in_tiledb(tmpdir, x_spec) as (x_params, x_data):
            x_tensor_kind = x_params.tensor_schema.kind
            with ingest_in_tiledb(tmpdir, y_spec) as (y_params, y_data):
                y_tensor_kind = y_params.tensor_schema.kind
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
                x_batches, y_batches = [], []
                for x_tensors, y_tensors in dataset:
                    x_batches.append(x_tensors[0])
                    y_batches.append(y_tensors[0])
                assert_tensors_almost_equal_array(
                    x_batches, x_data, x_tensor_kind, batch_size, to_dense
                )
                assert_tensors_almost_equal_array(
                    y_batches, y_data, y_tensor_kind, batch_size, to_dense
                )
