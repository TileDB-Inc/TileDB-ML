"""Tests for TileDB integration with Tensorflow Data API."""

import numpy as np
import pytest
import tensorflow as tf

from tiledb.ml.readers.tensorflow import TensorflowTileDBDataset, TensorKind

from .utils import (
    ArraySpec,
    assert_tensors_almost_equal_array,
    ingest_in_tiledb,
    validate_tensor_generator,
)


def dataset_batching_shuffling(dataset, batch_size=None, shuffle_buffer_size=0):
    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(shuffle_buffer_size)
    if batch_size is not None:
        dataset = dataset.batch(batch_size)
    return dataset


class TestTensorflowTileDBDataset:
    @pytest.mark.parametrize("spec", ArraySpec.combinations())
    @pytest.mark.parametrize("batch_size", [8, None])
    @pytest.mark.parametrize("shuffle_buffer_size", [0, 16])
    @pytest.mark.parametrize("num_workers", [0, 2])
    def test_dataset(self, tmpdir, spec, batch_size, shuffle_buffer_size, num_workers):
        def test(*all_array_params):
            dataset = dataset_batching_shuffling(
                TensorflowTileDBDataset(*all_array_params, num_workers=num_workers),
                batch_size=batch_size,
                shuffle_buffer_size=shuffle_buffer_size,
            )
            assert isinstance(dataset, tf.data.Dataset)
            validate_tensor_generator(
                dataset,
                [params.tensor_schema for params in all_array_params],
                [spec] * len(all_array_params),
                batch_size,
            )

        with ingest_in_tiledb(tmpdir, spec) as (x_params, x_data):
            # load tensors from single array
            test(x_params)
            with ingest_in_tiledb(tmpdir, spec) as (y_params, y_data):
                # load tensors from two arrays
                test(x_params, y_params)

    @pytest.mark.parametrize("spec", ArraySpec.combinations())
    def test_unequal_key_ranges(self, tmpdir, spec):
        y_spec = ArraySpec(
            # Add one extra key on Y
            shape=(spec.shape[0] + 1, *spec.shape[1:]),
            sparse=spec.sparse,
            key_dim=spec.key_dim,
            key_dim_dtype=spec.key_dim_dtype,
            non_key_dim_dtype=spec.non_key_dim_dtype,
            num_fields=spec.num_fields,
            tensor_kind=spec.tensor_kind,
        )
        with ingest_in_tiledb(tmpdir, spec) as (x_params, x_data):
            with ingest_in_tiledb(tmpdir, y_spec) as (y_params, y_data):
                with pytest.raises(ValueError) as ex:
                    TensorflowTileDBDataset(x_params, y_params)
                assert "All arrays must have the same key range" in str(ex.value)

    @pytest.mark.parametrize("spec", ArraySpec.combinations(num_fields=[0]))
    @pytest.mark.parametrize("batch_size", [8, None])
    def test_order(self, tmpdir, spec, batch_size):
        """Test we can read the data in the same order as written.

        The order is guaranteed only for sequential processing (num_workers=0) and
        no shuffling (shuffle_buffer_size=0).
        """
        to_dense = tf.sparse.to_dense
        with ingest_in_tiledb(tmpdir, spec) as (params, data):
            dataset = TensorflowTileDBDataset(params)
            dataset = dataset_batching_shuffling(dataset=dataset, batch_size=batch_size)
            # since num_fields is 0, fields are all the array attributes of each array
            # the first item of each batch corresponds to the first attribute (="data")
            batches = [tensors[0] for tensors in dataset]
            assert_tensors_almost_equal_array(
                batches, data, params.tensor_schema.kind, batch_size, to_dense
            )

    @pytest.mark.parametrize(
        "spec",
        ArraySpec.combinations(
            sparse=[True],
            non_key_dim_dtype=[np.dtype(np.int32)],
            tensor_kind=[TensorKind.SPARSE_CSR],
        ),
    )
    def test_csr(self, tmpdir, spec):
        with ingest_in_tiledb(tmpdir, spec) as (params, data):
            with pytest.raises(NotImplementedError) as ex:
                TensorflowTileDBDataset(params)
            assert "TensorKind.SPARSE_CSR tensors not supported" in str(ex.value)
