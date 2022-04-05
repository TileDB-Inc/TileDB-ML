"""Tests for TileDB integration with PyTorch Data API."""

import numpy as np
import pytest
import torch

from tiledb.ml.readers.pytorch import PyTorchTileDBDataLoader, PyTorchTileDBDataset

from .utils import (
    ingest_in_tiledb,
    parametrize_for_dataset,
    rand_array,
    validate_tensor_generator,
)


@pytest.mark.parametrize("num_rows", [107])
class TestPyTorchTileDBDataset:
    @parametrize_for_dataset()
    def test_dataset(
        self,
        tmpdir,
        num_rows,
        x_sparse,
        y_sparse,
        x_shape,
        y_shape,
        num_attrs,
        pass_attrs,
        buffer_bytes,
        batch_size,
        shuffle_buffer_size,
    ):
        with ingest_in_tiledb(
            tmpdir,
            x_data=rand_array(num_rows, *x_shape, sparse=x_sparse),
            y_data=rand_array(num_rows, *y_shape, sparse=y_sparse),
            x_sparse=x_sparse,
            y_sparse=y_sparse,
            num_attrs=num_attrs,
            pass_attrs=pass_attrs,
        ) as kwargs:
            dataset = PyTorchTileDBDataset(buffer_bytes=buffer_bytes, **kwargs)
            assert isinstance(dataset, torch.utils.data.IterableDataset)
            validate_tensor_generator(
                dataset, num_attrs, x_sparse, y_sparse, x_shape, y_shape
            )

    @parametrize_for_dataset()
    @pytest.mark.parametrize("num_workers", [0, 2])
    def test_dataloader(
        self,
        tmpdir,
        num_rows,
        num_workers,
        x_sparse,
        y_sparse,
        x_shape,
        y_shape,
        num_attrs,
        pass_attrs,
        buffer_bytes,
        batch_size,
        shuffle_buffer_size,
    ):
        if num_workers and (x_sparse or y_sparse):
            pytest.skip("multiple workers not supported with sparse arrays")

        with ingest_in_tiledb(
            tmpdir,
            x_data=rand_array(num_rows, *x_shape, sparse=x_sparse),
            y_data=rand_array(num_rows, *y_shape, sparse=y_sparse),
            x_sparse=x_sparse,
            y_sparse=y_sparse,
            num_attrs=num_attrs,
            pass_attrs=pass_attrs,
        ) as kwargs:
            dataloader = PyTorchTileDBDataLoader(
                num_workers=num_workers,
                buffer_bytes=buffer_bytes,
                batch_size=batch_size,
                shuffle_buffer_size=shuffle_buffer_size,
                **kwargs
            )
            assert isinstance(dataloader, torch.utils.data.DataLoader)
            validate_tensor_generator(
                dataloader, num_attrs, x_sparse, y_sparse, x_shape, y_shape, batch_size
            )

            unique_x_tensors = []
            unique_y_tensors = []
            for batchindx, data in enumerate(dataloader):
                for attr in range(num_attrs):
                    # Keep unique X tensors
                    x_tensor = data[attr]
                    if x_sparse:
                        x_tensor = x_tensor.to_dense()
                    if not any(torch.equal(x_tensor, t) for t in unique_x_tensors):
                        unique_x_tensors.append(x_tensor)

                    # Keep unique Y tensors
                    y_tensor = data[attr + num_attrs]
                    if y_sparse:
                        y_tensor = y_tensor.to_dense()
                    if not any(torch.equal(y_tensor, t) for t in unique_y_tensors):
                        unique_y_tensors.append(y_tensor)

                assert len(unique_x_tensors) - 1 == batchindx
                assert len(unique_y_tensors) - 1 == batchindx

    @parametrize_for_dataset()
    @pytest.mark.parametrize("num_workers", [0, 2])
    def test_unequal_num_rows(
        self,
        tmpdir,
        num_rows,
        num_workers,
        x_sparse,
        y_sparse,
        x_shape,
        y_shape,
        num_attrs,
        pass_attrs,
        buffer_bytes,
        batch_size,
        shuffle_buffer_size,
    ):
        with ingest_in_tiledb(
            tmpdir,
            # Add one extra row on X
            x_data=rand_array(num_rows + 1, *x_shape, sparse=x_sparse),
            y_data=rand_array(num_rows, *y_shape, sparse=y_sparse),
            x_sparse=x_sparse,
            y_sparse=y_sparse,
            num_attrs=num_attrs,
            pass_attrs=pass_attrs,
        ) as kwargs:
            with pytest.raises(ValueError) as ex:
                PyTorchTileDBDataLoader(
                    num_workers=num_workers,
                    buffer_bytes=buffer_bytes,
                    batch_size=batch_size,
                    shuffle_buffer_size=shuffle_buffer_size,
                    **kwargs
                )
            assert "X and Y arrays must have the same number of rows" in str(ex.value)

    @parametrize_for_dataset(x_sparse=[True], shuffle_buffer_size=[0])
    def test_sparse_read_order(
        self,
        tmpdir,
        num_rows,
        x_sparse,
        y_sparse,
        x_shape,
        y_shape,
        num_attrs,
        pass_attrs,
        buffer_bytes,
        batch_size,
        shuffle_buffer_size,
    ):
        x_data = rand_array(num_rows, *x_shape, sparse=x_sparse)
        with ingest_in_tiledb(
            tmpdir,
            x_data=x_data,
            y_data=rand_array(num_rows, *y_shape, sparse=y_sparse),
            x_sparse=x_sparse,
            y_sparse=y_sparse,
            num_attrs=num_attrs,
            pass_attrs=pass_attrs,
        ) as kwargs:
            dataloader = PyTorchTileDBDataLoader(
                buffer_bytes=buffer_bytes,
                batch_size=batch_size,
                shuffle_buffer_size=shuffle_buffer_size,
                **kwargs
            )
            generated_x_data = np.concatenate(
                [tensors[0].to_dense().numpy() for tensors in dataloader]
            )
            np.testing.assert_array_almost_equal(generated_x_data, x_data)
