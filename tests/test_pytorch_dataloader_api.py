"""Tests for TileDB integration with PyTorch Data API."""

import numpy as np
import pytest
import torch

import tiledb
from tiledb.ml.readers.pytorch import PyTorchTileDBDataset

from .utils import (
    ingest_in_tiledb,
    parametrize_for_dataset,
    rand_array,
    validate_tensor_generator,
)

# Test parameters
BATCH_SIZE = 20
ROWS = 100


class TestPyTorchTileDBDataset:
    @parametrize_for_dataset()
    @pytest.mark.parametrize("workers", [0, 2])
    def test_generator(
        self,
        tmpdir,
        sparse_x,
        sparse_y,
        input_shape,
        output_shape,
        num_attrs,
        pass_attrs,
        buffer_size,
        batch_shuffle,
        within_batch_shuffle,
        workers,
    ):
        if workers and (sparse_x or sparse_y):
            pytest.skip("multiple workers not supported with sparse arrays")

        data_x = rand_array(ROWS, *input_shape, sparse=sparse_x)
        data_y = rand_array(ROWS, *output_shape, sparse=sparse_y)
        uri_x, uri_y = ingest_in_tiledb(
            tmpdir,
            data_x=data_x,
            data_y=data_y,
            sparse_x=sparse_x,
            sparse_y=sparse_y,
            batch_size=BATCH_SIZE,
            num_attrs=num_attrs,
        )
        attrs = [f"features_{attr}" for attr in range(num_attrs)]
        with tiledb.open(uri_x) as x, tiledb.open(uri_y) as y:
            dataset = PyTorchTileDBDataset(
                x_array=x,
                y_array=y,
                batch_size=BATCH_SIZE,
                buffer_size=buffer_size,
                batch_shuffle=batch_shuffle,
                within_batch_shuffle=within_batch_shuffle,
                x_attrs=attrs if pass_attrs else [],
                y_attrs=attrs if pass_attrs else [],
            )
            assert isinstance(dataset, torch.utils.data.IterableDataset)
            validate_tensor_generator(
                dataset,
                sparse_x=sparse_x,
                sparse_y=sparse_y,
                shape_x=input_shape,
                shape_y=output_shape,
                batch_size=BATCH_SIZE,
                num_attrs=num_attrs,
            )
            train_loader = torch.utils.data.DataLoader(
                dataset, batch_size=None, num_workers=workers
            )
            unique_x_tensors = []
            unique_y_tensors = []
            for batchindx, data in enumerate(train_loader):
                for attr in range(num_attrs):
                    # Keep unique X tensors
                    x_tensor = data[attr]
                    if sparse_x:
                        x_tensor = x_tensor.to_dense()
                    if not any(torch.equal(x_tensor, t) for t in unique_x_tensors):
                        unique_x_tensors.append(x_tensor)

                    # Keep unique Y tensors
                    y_tensor = data[attr + num_attrs]
                    if sparse_y:
                        y_tensor = y_tensor.to_dense()
                    if not any(torch.equal(y_tensor, t) for t in unique_y_tensors):
                        unique_y_tensors.append(y_tensor)

                assert len(unique_x_tensors) - 1 == batchindx
                assert len(unique_y_tensors) - 1 == batchindx

    @parametrize_for_dataset(buffer_size=[BATCH_SIZE - 1])
    def test_buffer_size_smaller_than_batch_size(
        self,
        tmpdir,
        sparse_x,
        sparse_y,
        input_shape,
        output_shape,
        num_attrs,
        pass_attrs,
        buffer_size,
        batch_shuffle,
        within_batch_shuffle,
    ):
        data_x = rand_array(ROWS, *input_shape, sparse=sparse_x)
        data_y = rand_array(ROWS, *output_shape, sparse=sparse_y)
        uri_x, uri_y = ingest_in_tiledb(
            tmpdir,
            data_x=data_x,
            data_y=data_y,
            sparse_x=sparse_x,
            sparse_y=sparse_y,
            batch_size=BATCH_SIZE,
            num_attrs=num_attrs,
        )
        attrs = [f"features_{attr}" for attr in range(num_attrs)]
        with tiledb.open(uri_x) as x, tiledb.open(uri_y) as y:
            with pytest.raises(ValueError):
                PyTorchTileDBDataset(
                    x_array=x,
                    y_array=y,
                    batch_size=BATCH_SIZE,
                    buffer_size=buffer_size,
                    batch_shuffle=batch_shuffle,
                    within_batch_shuffle=within_batch_shuffle,
                    x_attrs=attrs if pass_attrs else [],
                    y_attrs=attrs if pass_attrs else [],
                )

    @parametrize_for_dataset()
    def test_unequal_num_rows(
        self,
        tmpdir,
        sparse_x,
        sparse_y,
        input_shape,
        output_shape,
        num_attrs,
        pass_attrs,
        buffer_size,
        batch_shuffle,
        within_batch_shuffle,
    ):
        # Add one extra row on X
        data_x = rand_array(ROWS + 1, *input_shape, sparse=sparse_x)
        data_y = rand_array(ROWS, *output_shape, sparse=sparse_y)
        uri_x, uri_y = ingest_in_tiledb(
            tmpdir,
            data_x=data_x,
            data_y=data_y,
            sparse_x=sparse_x,
            sparse_y=sparse_y,
            batch_size=BATCH_SIZE,
            num_attrs=num_attrs,
        )
        attrs = [f"features_{attr}" for attr in range(num_attrs)]
        with tiledb.open(uri_x) as x, tiledb.open(uri_y) as y:
            with pytest.raises(ValueError):
                PyTorchTileDBDataset(
                    x_array=x,
                    y_array=y,
                    batch_size=BATCH_SIZE,
                    buffer_size=buffer_size,
                    batch_shuffle=batch_shuffle,
                    within_batch_shuffle=within_batch_shuffle,
                    x_attrs=attrs if pass_attrs else [],
                    y_attrs=attrs if pass_attrs else [],
                )

    @parametrize_for_dataset(sparse_x=[True])
    def test_sparse_x_unequal_num_rows_in_batch(
        self,
        tmpdir,
        sparse_x,
        sparse_y,
        input_shape,
        output_shape,
        num_attrs,
        pass_attrs,
        buffer_size,
        batch_shuffle,
        within_batch_shuffle,
    ):
        data_x = rand_array(ROWS, *input_shape, sparse=sparse_x)
        data_x[np.nonzero(data_x[0])] = 0
        data_y = rand_array(ROWS, *output_shape, sparse=sparse_y)
        uri_x, uri_y = ingest_in_tiledb(
            tmpdir,
            data_x=data_x,
            data_y=data_y,
            sparse_x=sparse_x,
            sparse_y=sparse_y,
            batch_size=BATCH_SIZE,
            num_attrs=num_attrs,
        )
        attrs = [f"features_{attr}" for attr in range(num_attrs)]
        with tiledb.open(uri_x) as x, tiledb.open(uri_y) as y:
            dataset = PyTorchTileDBDataset(
                x_array=x,
                y_array=y,
                batch_size=BATCH_SIZE,
                buffer_size=buffer_size,
                batch_shuffle=batch_shuffle,
                within_batch_shuffle=within_batch_shuffle,
                x_attrs=attrs if pass_attrs else [],
                y_attrs=attrs if pass_attrs else [],
            )
            with pytest.raises(Exception):
                for _ in dataset:
                    pass
