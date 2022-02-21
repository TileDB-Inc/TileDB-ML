"""Tests for TileDB integration with PyTorch Data API."""

import numpy as np
import pytest
import torch

from tiledb.ml.readers.pytorch import PyTorchTileDBDataset

from .utils import (
    ingest_in_tiledb,
    parametrize_for_dataset,
    rand_array,
    validate_tensor_generator,
)


@pytest.mark.parametrize("num_rows", [107])
class TestPyTorchTileDBDataset:
    @parametrize_for_dataset()
    @pytest.mark.parametrize("num_workers", [0, 2])
    def test_generator(
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
        batch_size,
        buffer_size,
        batch_shuffle,
        within_batch_shuffle,
    ):
        if num_workers and (x_sparse or y_sparse):
            pytest.skip("multiple workers not supported with sparse arrays")

        with ingest_in_tiledb(
            tmpdir,
            x_data=rand_array(num_rows, *x_shape, sparse=x_sparse),
            y_data=rand_array(num_rows, *y_shape, sparse=y_sparse),
            x_sparse=x_sparse,
            y_sparse=y_sparse,
            batch_size=batch_size,
            num_attrs=num_attrs,
            pass_attrs=pass_attrs,
            buffer_size=buffer_size,
            batch_shuffle=batch_shuffle,
            within_batch_shuffle=within_batch_shuffle,
        ) as dataset_kwargs:
            dataset = PyTorchTileDBDataset(**dataset_kwargs)
            assert isinstance(dataset, torch.utils.data.IterableDataset)
            validate_tensor_generator(
                dataset,
                x_sparse=x_sparse,
                y_sparse=y_sparse,
                x_shape=x_shape,
                y_shape=y_shape,
                batch_size=batch_size,
                num_attrs=num_attrs,
            )
            train_loader = torch.utils.data.DataLoader(
                dataset, batch_size=None, num_workers=num_workers
            )
            unique_x_tensors = []
            unique_y_tensors = []
            for batchindx, data in enumerate(train_loader):
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

    @parametrize_for_dataset(batch_size=[32], buffer_size=[31])
    def test_buffer_size_smaller_than_batch_size(
        self,
        tmpdir,
        num_rows,
        x_sparse,
        y_sparse,
        x_shape,
        y_shape,
        num_attrs,
        pass_attrs,
        batch_size,
        buffer_size,
        batch_shuffle,
        within_batch_shuffle,
    ):
        with ingest_in_tiledb(
            tmpdir,
            x_data=rand_array(num_rows, *x_shape, sparse=x_sparse),
            y_data=rand_array(num_rows, *y_shape, sparse=y_sparse),
            x_sparse=x_sparse,
            y_sparse=y_sparse,
            batch_size=batch_size,
            num_attrs=num_attrs,
            pass_attrs=pass_attrs,
            buffer_size=buffer_size,
            batch_shuffle=batch_shuffle,
            within_batch_shuffle=within_batch_shuffle,
        ) as dataset_kwargs:
            with pytest.raises(ValueError) as ex:
                PyTorchTileDBDataset(**dataset_kwargs)
            assert "buffer_size must be >= batch_size" in str(ex.value)

    @parametrize_for_dataset()
    def test_unequal_num_rows(
        self,
        tmpdir,
        num_rows,
        x_sparse,
        y_sparse,
        x_shape,
        y_shape,
        num_attrs,
        pass_attrs,
        batch_size,
        buffer_size,
        batch_shuffle,
        within_batch_shuffle,
    ):
        with ingest_in_tiledb(
            tmpdir,
            # Add one extra row on X
            x_data=rand_array(num_rows + 1, *x_shape, sparse=x_sparse),
            y_data=rand_array(num_rows, *y_shape, sparse=y_sparse),
            x_sparse=x_sparse,
            y_sparse=y_sparse,
            batch_size=batch_size,
            num_attrs=num_attrs,
            pass_attrs=pass_attrs,
            buffer_size=buffer_size,
            batch_shuffle=batch_shuffle,
            within_batch_shuffle=within_batch_shuffle,
        ) as dataset_kwargs:
            with pytest.raises(ValueError) as ex:
                PyTorchTileDBDataset(**dataset_kwargs)
            assert "X and Y arrays must have the same number of rows" in str(ex.value)

    @parametrize_for_dataset(x_sparse=[True])
    def test_x_sparse_unequal_num_rows_in_batch(
        self,
        tmpdir,
        num_rows,
        x_sparse,
        y_sparse,
        x_shape,
        y_shape,
        num_attrs,
        pass_attrs,
        batch_size,
        buffer_size,
        batch_shuffle,
        within_batch_shuffle,
    ):
        x_data = rand_array(num_rows, *x_shape, sparse=x_sparse)
        x_data[np.random.randint(len(x_data))] = 0
        with ingest_in_tiledb(
            tmpdir,
            x_data=x_data,
            y_data=rand_array(num_rows, *y_shape, sparse=y_sparse),
            x_sparse=x_sparse,
            y_sparse=y_sparse,
            batch_size=batch_size,
            num_attrs=num_attrs,
            pass_attrs=pass_attrs,
            buffer_size=buffer_size,
            batch_shuffle=batch_shuffle,
            within_batch_shuffle=within_batch_shuffle,
        ) as dataset_kwargs:
            dataset = PyTorchTileDBDataset(**dataset_kwargs)
            with pytest.raises(ValueError) as ex:
                for _ in dataset:
                    pass
            assert "x and y batches should have the same length" in str(ex.value)
