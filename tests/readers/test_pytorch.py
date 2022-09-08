"""Tests for TileDB integration with PyTorch Data API."""
from operator import methodcaller

import numpy as np
import pytest
import torch
import torchdata

from tiledb.ml.readers.pytorch import PyTorchTileDBDataLoader, TensorKind

from .utils import (
    ArraySpec,
    assert_tensors_almost_equal_array,
    ingest_in_tiledb,
    validate_tensor_generator,
)

non_key_dim_dtype = [np.dtype(np.int32)]
if hasattr(torch, "nested"):
    non_key_dim_dtype.append(np.dtype(np.float32))


class TestPyTorchTileDBDataLoader:
    @pytest.mark.parametrize(
        "spec", ArraySpec.combinations(non_key_dim_dtype=non_key_dim_dtype)
    )
    @pytest.mark.parametrize("batch_size", [8, None])
    @pytest.mark.parametrize("shuffle_buffer_size", [0, 16])
    @pytest.mark.parametrize("num_workers", [0, 2])
    def test_dataloader(
        self, tmpdir, spec, batch_size, shuffle_buffer_size, num_workers
    ):
        def test(*all_array_params):
            try:
                dataloader = PyTorchTileDBDataLoader(
                    *all_array_params,
                    shuffle_buffer_size=shuffle_buffer_size,
                    batch_size=batch_size,
                    num_workers=num_workers,
                )
            except NotImplementedError:
                assert num_workers and (
                    torchdata.__version__ < "0.4"
                    or spec.tensor_kind is not TensorKind.DENSE
                )
            else:
                assert isinstance(dataloader, torch.utils.data.DataLoader)
                validate_tensor_generator(
                    dataloader,
                    [params.tensor_schema for params in all_array_params],
                    [spec] * len(all_array_params),
                    batch_size,
                )
                # ensure the dataloader can be iterated again
                n1 = sum(1 for _ in dataloader)
                assert n1 != 0
                n2 = sum(1 for _ in dataloader)
                assert n1 == n2

        with ingest_in_tiledb(tmpdir, spec) as (x_params, x_data):
            # load tensors from single array
            test(x_params)
            with ingest_in_tiledb(tmpdir, spec) as (y_params, y_data):
                # load tensors from two arrays
                test(x_params, y_params)

    @pytest.mark.parametrize(
        "spec",
        ArraySpec.combinations(
            sparse=[True],
            non_key_dim_dtype=[np.dtype(np.int32)],
            tensor_kind=[TensorKind.SPARSE_CSR],
        ),
    )
    @pytest.mark.parametrize("batch_size", [8, None])
    @pytest.mark.parametrize("shuffle_buffer_size", [0, 16])
    def test_csr(self, tmpdir, spec, batch_size, shuffle_buffer_size):
        with ingest_in_tiledb(tmpdir, spec) as (params, data):
            try:
                dataloader = PyTorchTileDBDataLoader(
                    params,
                    shuffle_buffer_size=shuffle_buffer_size,
                    batch_size=batch_size,
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

    @pytest.mark.parametrize(
        "spec", ArraySpec.combinations(non_key_dim_dtype=non_key_dim_dtype)
    )
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
                    PyTorchTileDBDataLoader(x_params, y_params)
                assert "All arrays must have the same key range" in str(ex.value)

    @pytest.mark.parametrize(
        "spec",
        ArraySpec.combinations(non_key_dim_dtype=non_key_dim_dtype, num_fields=[0]),
    )
    @pytest.mark.parametrize("batch_size", [8, None])
    def test_order(self, tmpdir, spec, batch_size):
        """Test we can read the data in the same order as written.

        The order is guaranteed only for sequential processing (num_workers=0) and
        no shuffling (shuffle_buffer_size=0).
        """
        to_dense = methodcaller("to_dense")
        with ingest_in_tiledb(tmpdir, spec) as (params, data):
            dataloader = PyTorchTileDBDataLoader(params, batch_size=batch_size)
            # since num_fields is 0, fields are all the array attributes of each array
            # the first item of each batch corresponds to the first attribute (="data")
            batches = [tensors[0] for tensors in dataloader]
            assert_tensors_almost_equal_array(
                batches, data, params.tensor_schema.kind, batch_size, to_dense
            )
