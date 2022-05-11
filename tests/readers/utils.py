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
    num_attrs=(1, 2),
    pass_attrs=(True, False),
    buffer_bytes=(1024, None),
    batch_size=(8,),
    shuffle_buffer_size=(0, 16),
    num_workers=(0, 2),
):
    argnames = [
        "x_shape",
        "y_shape",
        "x_sparse",
        "y_sparse",
        "x_key_dim",
        "y_key_dim",
        "num_attrs",
        "pass_attrs",
        "buffer_bytes",
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
        num_attrs,
        pass_attrs,
        buffer_bytes,
        batch_size,
        shuffle_buffer_size,
        num_workers,
    )
    return pytest.mark.parametrize(argnames, argvalues)


@contextmanager
def ingest_in_tiledb(tmpdir, shape, sparse, key_dim, num_attrs, pass_attrs):
    """Context manager for ingesting data into TileDB."""
    array_uuid = str(uuid.uuid4())
    uri = os.path.join(tmpdir, array_uuid)
    attrs = tuple(f"{array_uuid[0]}{i}" for i in range(num_attrs))
    data = original_data = rand_array(shape, sparse)
    if key_dim > 0:
        data = np.moveaxis(data, 0, key_dim)

    dims = [
        tiledb.Dim(
            name=f"dim_{dim}",
            domain=(0, data.shape[dim] - 1),
            tile=np.random.randint(1, data.shape[dim] + 1),
            dtype=np.int32,
        )
        for dim in range(data.ndim)
    ]

    # TileDB schema
    schema = tiledb.ArraySchema(
        domain=tiledb.Domain(*dims),
        sparse=sparse,
        attrs=[tiledb.Attr(name=attr, dtype=np.float32) for attr in attrs],
    )

    # Create the (empty) array on disk.
    tiledb.Array.create(uri, schema)

    # Ingest
    with tiledb.open(uri, "w") as tiledb_array:
        idx = np.nonzero(data) if sparse else slice(None)
        tiledb_array[idx] = {attr: data[idx] for attr in attrs}

    with tiledb.open(uri) as array:
        yield {
            "data": original_data,
            "array": array,
            "key_dim": key_dim,
            "attrs": attrs if pass_attrs else (),
        }


def rand_array(shape: Sequence[int], sparse: bool = False) -> np.ndarray:
    """Create a random array of the given shape.

    :param shape: Shape of the array.
    :param sparse:
      - If false, all values will be in the (0, 1) range.
      - If true, only `shape[0]` values will be in the (0, 1) range, the rest will be 0.
        Note: some rows may be all zeros.
    """
    if sparse:
        a = np.zeros(shape)
        flat_idxs = np.random.choice(a.size, size=len(a), replace=False)
        a.flat[flat_idxs] = np.random.random(len(flat_idxs))
    else:
        a = np.random.random(shape)
    assert a.shape == shape
    return a


def validate_tensor_generator(
    generator, num_attrs, x_sparse, y_sparse, x_shape, y_shape, batch_size=None
):
    for x_tensors, y_tensors in generator:
        for x_tensor in x_tensors if num_attrs > 1 else [x_tensors]:
            _validate_tensor(x_tensor, x_sparse, x_shape[1:], batch_size)
        for y_tensor in y_tensors if num_attrs > 1 else [y_tensors]:
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
