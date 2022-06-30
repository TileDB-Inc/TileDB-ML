"""Tests for TileDB integration with PyTorch Data API."""

import numpy as np
import pytest
import torch

from tiledb.ml.readers.pytorch import PyTorchTileDBDataLoader
from tiledb.ml.readers.types import ArrayParams

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
        key_dim_dtype,
        x_key_dim,
        y_key_dim,
        num_fields,
        batch_size,
        shuffle_buffer_size,
        num_workers,
    ):
        if num_workers and (x_sparse or y_sparse):
            pytest.skip("multiple workers not supported with sparse arrays")

        with ingest_in_tiledb(
            tmpdir, x_shape, x_sparse, key_dim_dtype, x_key_dim, num_fields
        ) as x_kwargs, ingest_in_tiledb(
            tmpdir, y_shape, y_sparse, key_dim_dtype, y_key_dim, num_fields
        ) as y_kwargs:
            dataloader = PyTorchTileDBDataLoader(
                ArrayParams(x_kwargs["array"], x_kwargs["key_dim"], x_kwargs["fields"]),
                ArrayParams(y_kwargs["array"], y_kwargs["key_dim"], y_kwargs["fields"]),
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
        key_dim_dtype,
        x_key_dim,
        y_key_dim,
        num_fields,
        batch_size,
        shuffle_buffer_size,
        num_workers,
    ):
        with ingest_in_tiledb(
            tmpdir, x_shape, x_sparse, key_dim_dtype, x_key_dim, num_fields
        ) as x_kwargs, ingest_in_tiledb(
            tmpdir, y_shape, y_sparse, key_dim_dtype, y_key_dim, num_fields
        ) as y_kwargs:
            with pytest.raises(ValueError) as ex:
                PyTorchTileDBDataLoader(
                    ArrayParams(
                        x_kwargs["array"], x_kwargs["key_dim"], x_kwargs["fields"]
                    ),
                    ArrayParams(
                        y_kwargs["array"], y_kwargs["key_dim"], y_kwargs["fields"]
                    ),
                    batch_size=batch_size,
                    shuffle_buffer_size=shuffle_buffer_size,
                    num_workers=num_workers,
                )
            assert "All arrays must have the same key range" in str(ex.value)

    @parametrize_for_dataset(num_fields=[0], shuffle_buffer_size=[0], num_workers=[0])
    @pytest.mark.parametrize("csr", [True, False])
    def test_dataloader_order(
        self,
        tmpdir,
        x_shape,
        y_shape,
        x_sparse,
        y_sparse,
        key_dim_dtype,
        x_key_dim,
        y_key_dim,
        num_fields,
        batch_size,
        shuffle_buffer_size,
        num_workers,
        csr,
    ):
        """Test we can read the data in the same order as written.

        The order is guaranteed only for sequential processing (num_workers=0) and
        no shuffling (shuffle_buffer_size=0).
        """
        with ingest_in_tiledb(
            tmpdir, x_shape, x_sparse, key_dim_dtype, x_key_dim, num_fields
        ) as x_kwargs, ingest_in_tiledb(
            tmpdir, y_shape, y_sparse, key_dim_dtype, y_key_dim, num_fields
        ) as y_kwargs:
            dataloader = PyTorchTileDBDataLoader(
                ArrayParams(x_kwargs["array"], x_kwargs["key_dim"], x_kwargs["fields"]),
                ArrayParams(y_kwargs["array"], y_kwargs["key_dim"], y_kwargs["fields"]),
                batch_size=batch_size,
                shuffle_buffer_size=shuffle_buffer_size,
                num_workers=num_workers,
                csr=csr,
            )
            # since num_fields is 0, fields are all the array attributes of each array
            # the first item of each batch corresponds to the first attribute (="data")
            x_data_batches, y_data_batches = [], []
            for x_tensors, y_tensors in dataloader:
                x_data_batch = x_tensors[0]
                if x_sparse:
                    x_data_batch = x_data_batch.to_dense()
                x_data_batches.append(x_data_batch)

                y_data_batch = y_tensors[0]
                if y_sparse:
                    y_data_batch = y_data_batch.to_dense()
                y_data_batches.append(y_data_batch)

            np.testing.assert_array_almost_equal(
                np.concatenate(x_data_batches), x_kwargs["data"]
            )
            np.testing.assert_array_almost_equal(
                np.concatenate(y_data_batches), y_kwargs["data"]
            )
