"""Tests for TileDB integration with PyTorch Data API."""
from operator import methodcaller

import numpy as np
import pytest
import torch

from tiledb.ml.readers.pytorch import PyTorchTileDBDataLoader, TensorKind

from .utils import (
    ArraySpec,
    assert_tensors_almost_equal_array,
    ingest_in_tiledb,
    parametrize_for_dataset,
    validate_tensor_generator,
)


class TestPyTorchTileDBDataLoader:
    @parametrize_for_dataset()
    def test_dataloader(
        self, tmpdir, spec, batch_size, shuffle_buffer_size, num_workers
    ):
        with ingest_in_tiledb(tmpdir, spec) as (params, data):
            try:
                dataloader = PyTorchTileDBDataLoader(
                    params,
                    shuffle_buffer_size=shuffle_buffer_size,
                    batch_size=batch_size,
                    num_workers=num_workers,
                )
            except NotImplementedError:
                assert num_workers and spec.sparse
            else:
                assert isinstance(dataloader, torch.utils.data.DataLoader)
                validate_tensor_generator(
                    dataloader, params.tensor_schema, spec, batch_size
                )
                # ensure the dataloader can be iterated again
                n1 = sum(1 for _ in dataloader)
                assert n1 != 0
                n2 = sum(1 for _ in dataloader)
                assert n1 == n2

    @parametrize_for_dataset(
        sparse=(True,),
        non_key_dim_dtype=(np.dtype(np.int32),),
        num_workers=[0],
    )
    def test_csr(self, tmpdir, spec, batch_size, shuffle_buffer_size, num_workers):
        tensor_kind = TensorKind.SPARSE_CSR
        with ingest_in_tiledb(tmpdir, spec, tensor_kind) as (params, data):
            try:
                dataloader = PyTorchTileDBDataLoader(
                    params,
                    shuffle_buffer_size=shuffle_buffer_size,
                    batch_size=batch_size,
                    num_workers=num_workers,
                )
            except ValueError as ex:
                assert str(ex) == "Cannot generate CSR tensors for 3D array"
                ndim = len(spec.shape)
                assert ndim > 3 or ndim == 3 and batch_size is not None
            else:
                assert isinstance(dataloader, torch.utils.data.DataLoader)
                validate_tensor_generator(
                    dataloader, params.tensor_schema, spec, batch_size
                )
                # ensure the dataloader can be iterated again
                n1 = sum(1 for _ in dataloader)
                assert n1 != 0
                n2 = sum(1 for _ in dataloader)
                assert n1 == n2

    @parametrize_for_dataset()
    def test_multiple_arrays(
        self, tmpdir, spec, batch_size, shuffle_buffer_size, num_workers
    ):
        with ingest_in_tiledb(tmpdir, spec) as (x_params, x_data):
            with ingest_in_tiledb(tmpdir, spec) as (y_params, y_data):
                try:
                    dataloader = PyTorchTileDBDataLoader(
                        x_params,
                        y_params,
                        shuffle_buffer_size=shuffle_buffer_size,
                        batch_size=batch_size,
                        num_workers=num_workers,
                    )
                except NotImplementedError:
                    assert num_workers and spec.sparse
                else:
                    assert isinstance(dataloader, torch.utils.data.DataLoader)
                    validate_tensor_generator(
                        dataloader,
                        [x_params.tensor_schema, y_params.tensor_schema],
                        [spec, spec],
                        batch_size,
                    )
                    # ensure the dataloader can be iterated again
                    n1 = sum(1 for _ in dataloader)
                    assert n1 != 0
                    n2 = sum(1 for _ in dataloader)
                    assert n1 == n2

    @parametrize_for_dataset()
    def test_multiple_arrays_unequal_key_ranges(
        self, tmpdir, spec, batch_size, shuffle_buffer_size, num_workers
    ):
        y_spec = ArraySpec(
            # Add one extra key on Y
            shape=(spec.shape[0] + 1, *spec.shape[1:]),
            sparse=spec.sparse,
            key_dim=spec.key_dim,
            key_dim_dtype=spec.key_dim_dtype,
            non_key_dim_dtype=spec.non_key_dim_dtype,
            num_fields=spec.num_fields,
        )
        with ingest_in_tiledb(tmpdir, spec) as (x_params, x_data):
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
        num_fields=[0],
        shuffle_buffer_size=[0],
        num_workers=[0],
    )
    def test_order(self, tmpdir, spec, batch_size, shuffle_buffer_size, num_workers):
        """Test we can read the data in the same order as written.

        The order is guaranteed only for sequential processing (num_workers=0) and
        no shuffling (shuffle_buffer_size=0).
        """
        to_dense = methodcaller("to_dense")
        with ingest_in_tiledb(tmpdir, spec) as (params, data):
            dataloader = PyTorchTileDBDataLoader(
                params,
                shuffle_buffer_size=shuffle_buffer_size,
                batch_size=batch_size,
                num_workers=num_workers,
            )
            # since num_fields is 0, fields are all the array attributes of each array
            # the first item of each batch corresponds to the first attribute (="data")
            batches = [tensors[0] for tensors in dataloader]
            assert_tensors_almost_equal_array(
                batches, data, params.tensor_schema.kind, batch_size, to_dense
            )
