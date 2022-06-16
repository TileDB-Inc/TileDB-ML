import itertools as it
import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Sequence

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
    key_dim_dtype=(np.dtype(np.int32), np.dtype("datetime64[D]"), np.dtype(np.bytes_)),
    x_key_dim=(0, 1),
    y_key_dim=(0, 1),
    num_fields=(0, 1, 2),
    batch_size=(8,),
    shuffle_buffer_size=(16,),
    num_workers=(0, 2),
):
    def is_valid_combination(t):
        _, _, x_sparse_, y_sparse_, key_dim_dtype_, *_ = t
        return bool(
            np.issubdtype(key_dim_dtype_, np.integer) or (x_sparse_ and y_sparse_)
        )

    argnames = [
        "x_shape",
        "y_shape",
        "x_sparse",
        "y_sparse",
        "key_dim_dtype",
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
        key_dim_dtype,
        x_key_dim,
        y_key_dim,
        num_fields,
        batch_size,
        shuffle_buffer_size,
        num_workers,
    )
    return pytest.mark.parametrize(argnames, filter(is_valid_combination, argvalues))


@contextmanager
def ingest_in_tiledb(tmpdir, shape, sparse, key_dim_dtype, key_dim, num_fields):
    """Context manager for ingesting data into TileDB."""
    array_uuid = str(uuid.uuid4())
    uri = os.path.join(tmpdir, array_uuid)
    data = original_data = _rand_array(shape, sparse)
    if key_dim > 0:
        data = np.moveaxis(data, 0, key_dim)
    data_idx = np.arange(data.size).reshape(data.shape)

    transforms = []
    for i in range(data.ndim):
        n = data.shape[i]
        dtype = key_dim_dtype
        if i != key_dim:
            dtype = np.dtype("int32")
            # set the domain to (-n/2, n/2) to test negative domain indexing
            min_value = -(n // 2)
        elif np.issubdtype(key_dim_dtype, np.integer):
            # set the domain to (-n/2, n/2) to test negative domain indexing
            min_value = -(n // 2)
        elif np.issubdtype(key_dim_dtype, np.datetime64):
            min_value = np.datetime64("2022-06-15")
        elif np.issubdtype(key_dim_dtype, np.bytes_):
            min_value = b"a"
        else:
            assert False, key_dim_dtype
        transforms.append(_IndexTransformer(f"dim_{i}", n, min_value, dtype))

    dims = [transform.dim for transform in transforms]
    attrs = [
        tiledb.Attr(name="data", dtype=np.float32),
        tiledb.Attr(name="idx", dtype=np.int16),
    ]
    schema = tiledb.ArraySchema(domain=tiledb.Domain(*dims), attrs=attrs, sparse=sparse)
    tiledb.Array.create(uri, schema)

    with tiledb.open(uri, "w") as tiledb_array:
        if sparse:
            nz_idxs = np.nonzero(data)
            dim_idxs = tuple(
                transform(idx) for transform, idx in zip(transforms, nz_idxs)
            )
            tiledb_array[dim_idxs] = {"data": data[nz_idxs], "idx": data_idx[nz_idxs]}
        else:
            tiledb_array[:] = {"data": data, "idx": data_idx}

    all_fields = [f.name for f in dims + attrs]
    # exclude the key dimension from the fields if it is not an integer
    if not np.issubdtype(key_dim_dtype, np.integer):
        del all_fields[key_dim]
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


@dataclass(frozen=True)
class _IndexTransformer:
    name: str
    size: int
    min_value: Any
    dtype: np.dtype

    @property
    def dim(self):
        return tiledb.Dim(
            name=self.name,
            domain=(self.min_value, self(self.size - 1)),
            tile=np.random.randint(1, self.size + 1),
            dtype=self.dtype,
        )

    def __call__(self, idx):
        if isinstance(self.min_value, bytes):
            transformed_idx = _bytes_to_int(self.min_value) + idx
            if isinstance(transformed_idx, np.ndarray):
                int_to_bytes = np.vectorize(_int_to_bytes)
            else:
                int_to_bytes = _int_to_bytes
            return int_to_bytes(transformed_idx)
        else:
            return self.min_value + idx


def _bytes_to_int(data: bytes) -> int:
    s = 0
    for i, b in enumerate(reversed(data)):
        s += b * 256**i
    return s


def _int_to_bytes(n: int) -> bytes:
    s = bytearray()
    while n > 0:
        n, m = divmod(n, 256)
        s.append(m)
    s.reverse()
    return bytes(s)


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
