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
        num_attrs,
        pass_attrs,
        buffer_bytes,
        batch_size,
        shuffle_buffer_size,
        num_workers,
    ):
        if num_workers and (x_sparse or y_sparse):
            pytest.skip("multiple workers not supported with sparse arrays")

        with ingest_in_tiledb(
            tmpdir, x_shape, x_sparse, x_key_dim, num_attrs, pass_attrs
        ) as x_kwargs, ingest_in_tiledb(
            tmpdir, y_shape, y_sparse, y_key_dim, num_attrs, pass_attrs
        ) as y_kwargs:
            dataloader = PyTorchTileDBDataLoader(
                x_array=x_kwargs["array"],
                y_array=y_kwargs["array"],
                x_attrs=x_kwargs["attrs"],
                y_attrs=y_kwargs["attrs"],
                x_key_dim=x_kwargs["key_dim"],
                y_key_dim=y_kwargs["key_dim"],
                buffer_bytes=buffer_bytes,
                batch_size=batch_size,
                shuffle_buffer_size=shuffle_buffer_size,
                num_workers=num_workers,
            )
            assert isinstance(dataloader, torch.utils.data.DataLoader)
            validate_tensor_generator(
                dataloader, num_attrs, x_sparse, y_sparse, x_shape, y_shape, batch_size
            )

            unique_x_tensors = []
            unique_y_tensors = []
            for i, (x_tensors, y_tensors) in enumerate(dataloader):
                # Keep unique X tensors
                for x_tensor in x_tensors if num_attrs > 1 else [x_tensors]:
                    if x_sparse:
                        x_tensor = x_tensor.to_dense()
                    if not any(torch.equal(x_tensor, t) for t in unique_x_tensors):
                        unique_x_tensors.append(x_tensor)

                # Keep unique Y tensors
                for y_tensor in y_tensors if num_attrs > 1 else [y_tensors]:
                    if y_sparse:
                        y_tensor = y_tensor.to_dense()
                    if not any(torch.equal(y_tensor, t) for t in unique_y_tensors):
                        unique_y_tensors.append(y_tensor)

                assert len(unique_x_tensors) - 1 == i
                assert len(unique_y_tensors) - 1 == i

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
        num_attrs,
        pass_attrs,
        buffer_bytes,
        batch_size,
        shuffle_buffer_size,
        num_workers,
    ):
        with ingest_in_tiledb(
            tmpdir, x_shape, x_sparse, x_key_dim, num_attrs, pass_attrs
        ) as x_kwargs, ingest_in_tiledb(
            tmpdir, y_shape, y_sparse, y_key_dim, num_attrs, pass_attrs
        ) as y_kwargs:
            with pytest.raises(ValueError) as ex:
                PyTorchTileDBDataLoader(
                    x_array=x_kwargs["array"],
                    y_array=y_kwargs["array"],
                    x_attrs=x_kwargs["attrs"],
                    y_attrs=y_kwargs["attrs"],
                    x_key_dim=x_kwargs["key_dim"],
                    y_key_dim=y_kwargs["key_dim"],
                    buffer_bytes=buffer_bytes,
                    batch_size=batch_size,
                    shuffle_buffer_size=shuffle_buffer_size,
                    num_workers=num_workers,
                )
            assert "X and Y arrays have different key domain" in str(ex.value)

    @parametrize_for_dataset(x_sparse=[True], shuffle_buffer_size=[0], num_workers=[0])
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
        num_attrs,
        pass_attrs,
        buffer_bytes,
        batch_size,
        shuffle_buffer_size,
        num_workers,
        csr,
    ):
        with ingest_in_tiledb(
            tmpdir, x_shape, x_sparse, x_key_dim, num_attrs, pass_attrs
        ) as x_kwargs, ingest_in_tiledb(
            tmpdir, y_shape, y_sparse, y_key_dim, num_attrs, pass_attrs
        ) as y_kwargs:
            dataloader = PyTorchTileDBDataLoader(
                x_array=x_kwargs["array"],
                y_array=y_kwargs["array"],
                x_attrs=x_kwargs["attrs"],
                y_attrs=y_kwargs["attrs"],
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
                    (x if num_attrs == 1 else x[0]).to_dense().numpy()
                    for x, y in dataloader
                ]
            )
            np.testing.assert_array_almost_equal(generated_x_data, x_kwargs["data"])
