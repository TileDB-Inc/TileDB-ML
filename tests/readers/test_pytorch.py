"""Tests for TileDB integration with PyTorch Data API."""

import numpy as np
import pytest
import torch

from tiledb.ml.readers.pytorch import PyTorchTileDBDataLoader

from .utils import ingest_in_tiledb, parametrize_for_dataset, validate_tensor_generator


class TestPyTorchTileDBDataLoader:
    @parametrize_for_dataset()
    def test_dataloader(
        self,
        tmpdir,
        x_shape,
        y_shape,
        x_sparse,
        y_sparse,
        x_key_dim,
        y_key_dim,
        num_fields,
        buffer_bytes,
        batch_size,
        shuffle_buffer_size,
        num_workers,
    ):
        if num_workers and (x_sparse or y_sparse):
            pytest.skip("multiple workers not supported with sparse arrays")

        with ingest_in_tiledb(
            tmpdir, x_shape, x_sparse, x_key_dim, num_fields
        ) as x_kwargs, ingest_in_tiledb(
            tmpdir, y_shape, y_sparse, y_key_dim, num_fields
        ) as y_kwargs:
            dataloader = PyTorchTileDBDataLoader(
                x_array=x_kwargs["array"],
                y_array=y_kwargs["array"],
                x_attrs=x_kwargs["fields"],
                y_attrs=y_kwargs["fields"],
                x_key_dim=x_kwargs["key_dim"],
                y_key_dim=y_kwargs["key_dim"],
                buffer_bytes=buffer_bytes,
                batch_size=batch_size,
                shuffle_buffer_size=shuffle_buffer_size,
                num_workers=num_workers,
            )
            assert isinstance(dataloader, torch.utils.data.DataLoader)
            validate_tensor_generator(
                dataloader, num_fields, x_sparse, y_sparse, x_shape, y_shape, batch_size
            )

    @parametrize_for_dataset(
        # Add one extra key on X
        x_shape=((108, 10), (108, 10, 3)),
        y_shape=((107, 5), (107, 5, 2)),
    )
    def test_unequal_num_keys(
        self,
        tmpdir,
        x_shape,
        y_shape,
        x_sparse,
        y_sparse,
        x_key_dim,
        y_key_dim,
        num_fields,
        buffer_bytes,
        batch_size,
        shuffle_buffer_size,
        num_workers,
    ):
        with ingest_in_tiledb(
            tmpdir, x_shape, x_sparse, x_key_dim, num_fields
        ) as x_kwargs, ingest_in_tiledb(
            tmpdir, y_shape, y_sparse, y_key_dim, num_fields
        ) as y_kwargs:
            with pytest.raises(ValueError) as ex:
                PyTorchTileDBDataLoader(
                    x_array=x_kwargs["array"],
                    y_array=y_kwargs["array"],
                    x_attrs=x_kwargs["fields"],
                    y_attrs=y_kwargs["fields"],
                    x_key_dim=x_kwargs["key_dim"],
                    y_key_dim=y_kwargs["key_dim"],
                    buffer_bytes=buffer_bytes,
                    batch_size=batch_size,
                    shuffle_buffer_size=shuffle_buffer_size,
                    num_workers=num_workers,
                )
            assert "X and Y arrays have different key domain" in str(ex.value)

    @parametrize_for_dataset(
        x_sparse=[True], num_fields=[0], shuffle_buffer_size=[0], num_workers=[0]
    )
    @pytest.mark.parametrize("csr", [True, False])
    def test_sparse_read_order(
        self,
        tmpdir,
        x_shape,
        y_shape,
        x_sparse,
        y_sparse,
        x_key_dim,
        y_key_dim,
        num_fields,
        buffer_bytes,
        batch_size,
        shuffle_buffer_size,
        num_workers,
        csr,
    ):
        with ingest_in_tiledb(
            tmpdir, x_shape, x_sparse, x_key_dim, num_fields
        ) as x_kwargs, ingest_in_tiledb(
            tmpdir, y_shape, y_sparse, y_key_dim, num_fields
        ) as y_kwargs:
            dataloader = PyTorchTileDBDataLoader(
                x_array=x_kwargs["array"],
                y_array=y_kwargs["array"],
                x_attrs=x_kwargs["fields"],
                y_attrs=y_kwargs["fields"],
                x_key_dim=x_kwargs["key_dim"],
                y_key_dim=y_kwargs["key_dim"],
                buffer_bytes=buffer_bytes,
                batch_size=batch_size,
                shuffle_buffer_size=shuffle_buffer_size,
                num_workers=num_workers,
                csr=csr,
            )
            generated_x_data = np.concatenate(
                [
                    (x if num_fields == 1 else x[0]).to_dense().numpy()
                    for x, y in dataloader
                ]
            )
            np.testing.assert_array_almost_equal(generated_x_data, x_kwargs["data"])
