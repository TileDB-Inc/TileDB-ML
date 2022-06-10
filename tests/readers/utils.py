import itertools as it
import os
import uuid
from contextlib import contextmanager
from typing import Sequence

import numpy as np
import pytest
import scipy.sparse
import sparse
import tensorflow as tf
import torch

import tiledb

NUM_ROWS = 107


def parametrize_for_dataset(
    *,
    x_shape=((NUM_ROWS, 10), (NUM_ROWS, 10, 3)),
    y_shape=((NUM_ROWS, 5), (NUM_ROWS, 5, 2)),
    x_sparse=(True, False),
    y_sparse=(True, False),
    x_key_dim=(0, 1),
    y_key_dim=(0, 1),
    num_fields=(0, 1, 2),
    batch_size=(8,),
    shuffle_buffer_size=(16,),
    num_workers=(0, 2),
):
    argnames = [
        "x_shape",
        "y_shape",
        "x_sparse",
        "y_sparse",
        "x_key_dim",
        "y_key_dim",
        "num_fields",
        "batch_size",
        "shuffle_buffer_size",
        "num_workers",
    ]
    argvalues = it.product(
        x_shape,
        y_shape,
        x_sparse,
        y_sparse,
        x_key_dim,
        y_key_dim,
        num_fields,
        batch_size,
        shuffle_buffer_size,
        num_workers,
    )
    return pytest.mark.parametrize(argnames, argvalues)


@contextmanager
def ingest_in_tiledb(tmpdir, shape, sparse, key_dim, num_fields):
    """Context manager for ingesting data into TileDB."""
    array_uuid = str(uuid.uuid4())
    uri = os.path.join(tmpdir, array_uuid)
    data = original_data = _rand_array(shape, sparse)
    if key_dim > 0:
        data = np.moveaxis(data, 0, key_dim)

    # set the domain to (-n/2, n/2) to test negative domain indexing
    dim_starts = [-(data.shape[dim] // 2) for dim in range(data.ndim)]
    dims = [
        tiledb.Dim(
            name=f"dim_{dim}",
            domain=(dim_start, dim_start + data.shape[dim] - 1),
            tile=np.random.randint(1, data.shape[dim] + 1),
            dtype=np.int32,
        )
        for dim, dim_start in enumerate(dim_starts)
    ]
    attrs = [
        tiledb.Attr(name="data", dtype=np.float32),
        tiledb.Attr(name="idx", dtype=np.int16),
    ]
    schema = tiledb.ArraySchema(domain=tiledb.Domain(*dims), attrs=attrs, sparse=sparse)
    tiledb.Array.create(uri, schema)

    with tiledb.open(uri, "w") as tiledb_array:
        data_idx = np.arange(data.size).reshape(data.shape)
        if sparse:
            nz_idxs = np.nonzero(data)
            dim_idxs = tuple(
                dim_start + idx for idx, dim_start in zip(nz_idxs, dim_starts)
            )
            tiledb_array[dim_idxs] = {"data": data[nz_idxs], "idx": data_idx[nz_idxs]}
        else:
            tiledb_array[:] = {"data": data, "idx": data_idx}

    all_fields = [f.name for f in dims + attrs]
    fields = np.random.choice(all_fields, size=num_fields, replace=False).tolist()
    with tiledb.open(uri) as array:
        yield {
            "data": original_data,
            "array": array,
            "key_dim": key_dim,
            "fields": fields,
        }


def _rand_array(shape: Sequence[int], sparse: bool = False) -> np.ndarray:
    """Create a random array of the given shape.

    :param shape: Shape of the array.
    :param sparse:
      - If false, all values will be in the (0, 1) range.
      - If true, only one value per row will be non-zero, the rest will be 0.
    """
    if not sparse:
        return np.random.random(shape)

    rows, cols = shape[0], np.prod(shape[1:])
    a = np.zeros((rows, cols))
    col_idxs = np.random.choice(cols, size=rows)
    a[np.arange(rows), col_idxs] = np.random.random(rows)
    return a.reshape(shape)


def validate_tensor_generator(
    generator, num_fields, x_sparse, y_sparse, x_shape, y_shape, batch_size=None
):
    for x_tensors, y_tensors in generator:
        for x_tensor in x_tensors if num_fields != 1 else [x_tensors]:
            _validate_tensor(x_tensor, x_sparse, x_shape[1:], batch_size)
        for y_tensor in y_tensors if num_fields != 1 else [y_tensors]:
            _validate_tensor(y_tensor, y_sparse, y_shape[1:], batch_size)


def _validate_tensor(tensor, expected_sparse, expected_row_shape, batch_size=None):
    if batch_size is None and not isinstance(tensor, scipy.sparse.spmatrix):
        row_shape = tensor.shape
    else:
        num_rows, *row_shape = tensor.shape
        if batch_size is None:
            assert isinstance(tensor, scipy.sparse.spmatrix)
            assert num_rows == 1
        else:
            # num_rows may be less than batch_size
            assert num_rows <= batch_size, (num_rows, batch_size)
    assert tuple(row_shape) == expected_row_shape
    assert _is_sparse(tensor) == expected_sparse


def _is_sparse(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.layout in (torch.sparse_coo, torch.sparse_csr)
    if isinstance(tensor, (scipy.sparse.spmatrix, sparse.SparseArray, tf.SparseTensor)):
        return True
    if isinstance(tensor, (np.ndarray, tf.Tensor)):
        return False
    assert False, f"Unknown tensor type: {type(tensor)}"
