"""Tests for TileDB integration with PyTorch Data API."""

import numpy as np
import pytest
import torch

from tiledb.ml.readers.pytorch import PyTorchTileDBDataLoader

from .utils import ingest_in_tiledb, parametrize_for_dataset, validate_tensor_generator

if hasattr(torch, "nested_tensor"):
    non_key_dim_dtype = (np.dtype(np.int32), np.dtype(np.float32))
else:
    non_key_dim_dtype = (np.dtype(np.int32),)


class TestPyTorchTileDBDataLoader:
    @parametrize_for_dataset(non_key_dim_dtype=non_key_dim_dtype)
    def test_dataloader(
        self, tmpdir, x_spec, y_spec, batch_size, shuffle_buffer_size, num_workers
    ):
        with ingest_in_tiledb(tmpdir, x_spec) as (x_params, x_data):
            with ingest_in_tiledb(tmpdir, y_spec) as (y_params, y_data):
                try:
                    dataloader = PyTorchTileDBDataLoader(
                        x_params,
                        y_params,
                        shuffle_buffer_size=shuffle_buffer_size,
                        batch_size=batch_size,
                        num_workers=num_workers,
                    )
                except NotImplementedError:
                    assert num_workers and (x_spec.sparse or y_spec.sparse)
                else:
                    assert isinstance(dataloader, torch.utils.data.DataLoader)
                    validate_tensor_generator(
                        dataloader, x_spec, y_spec, batch_size, supports_csr=True
                    )

    @parametrize_for_dataset(
        non_key_dim_dtype=non_key_dim_dtype,
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
                    PyTorchTileDBDataLoader(
                        x_params,
                        y_params,
                        shuffle_buffer_size=shuffle_buffer_size,
                        batch_size=batch_size,
                        num_workers=num_workers,
                    )
                assert "All arrays must have the same key range" in str(ex.value)

    @parametrize_for_dataset(
        non_key_dim_dtype=non_key_dim_dtype,
        num_fields=[0],
        shuffle_buffer_size=[0],
        num_workers=[0],
    )
    def test_dataloader_order(
        self, tmpdir, x_spec, y_spec, batch_size, shuffle_buffer_size, num_workers
    ):
        """Test we can read the data in the same order as written.

        The order is guaranteed only for sequential processing (num_workers=0) and
        no shuffling (shuffle_buffer_size=0).
        """
        with ingest_in_tiledb(tmpdir, x_spec) as (x_params, x_data):
            with ingest_in_tiledb(tmpdir, y_spec) as (y_params, y_data):
                dataloader = PyTorchTileDBDataLoader(
                    x_params,
                    y_params,
                    shuffle_buffer_size=shuffle_buffer_size,
                    batch_size=batch_size,
                    num_workers=num_workers,
                )
                # since num_fields is 0, fields are all the array attributes of each array
                # the first item of each batch corresponds to the first attribute (="data")
                x_batch_tensors, y_batch_tensors = [], []
                for x_tensors, y_tensors in dataloader:
                    x_batch_tensors.append(x_tensors[0])
                    y_batch_tensors.append(y_tensors[0])
                assert_tensors_almost_equal_array(x_batch_tensors, x_data)
                assert_tensors_almost_equal_array(y_batch_tensors, y_data)


def assert_tensors_almost_equal_array(batch_tensors, array):
    if getattr(batch_tensors[0], "is_nested", False):
        # compare each ragged tensor row with the non-zero values of the respective array row
        tensors = [tensor for batch_tensor in batch_tensors for tensor in batch_tensor]
        assert len(tensors) == len(array)
        for tensor_row, row in zip(tensors, array):
            np.testing.assert_array_almost_equal(tensor_row, row[np.nonzero(row)])
    else:
        if batch_tensors[0].layout in (torch.sparse_coo, torch.sparse_csr):
            batch_tensors = [batch_tensor.to_dense() for batch_tensor in batch_tensors]
        np.testing.assert_array_almost_equal(np.concatenate(batch_tensors), array)
