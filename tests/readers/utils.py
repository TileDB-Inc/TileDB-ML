import itertools as it
import os
import uuid
from contextlib import contextmanager

import numpy as np
import pytest
import tensorflow as tf
import torch

import tiledb


def parametrize_for_dataset(
    x_sparse=(True, False),
    y_sparse=(True, False),
    x_shape=((10,), (10, 3)),
    y_shape=((5,), (5, 2)),
    num_attrs=(1, 2),
    pass_attrs=(True, False),
    batch_size=(8,),
    buffer_bytes=(1024, None),
    shuffle_buffer_size=(0, 16),
):
    argnames = [
        "x_sparse",
        "y_sparse",
        "x_shape",
        "y_shape",
        "num_attrs",
        "pass_attrs",
        "buffer_bytes",
        "batch_size",
        "shuffle_buffer_size",
    ]
    argvalues = it.product(
        x_sparse,
        y_sparse,
        x_shape,
        y_shape,
        num_attrs,
        pass_attrs,
        buffer_bytes,
        batch_size,
        shuffle_buffer_size,
    )
    return pytest.mark.parametrize(argnames, argvalues)


@contextmanager
def ingest_in_tiledb(
    tmpdir,
    x_data,
    y_data,
    x_sparse,
    y_sparse,
    num_attrs,
    pass_attrs,
):
    """Context manager for ingest data into TileDB.

    Yield the keyword arguments for instantiating a TiledbDataset.
    """
    array_uuid = str(uuid.uuid4())
    x_uri = os.path.join(tmpdir, "x_" + array_uuid)
    y_uri = os.path.join(tmpdir, "y_" + array_uuid)
    _ingest_in_tiledb(x_uri, x_data, x_sparse, num_attrs)
    _ingest_in_tiledb(y_uri, y_data, y_sparse, num_attrs)
    attrs = [f"features_{attr}" for attr in range(num_attrs)] if pass_attrs else []
    with tiledb.open(x_uri) as x_array, tiledb.open(y_uri) as y_array:
        yield dict(x_array=x_array, y_array=y_array, x_attrs=attrs, y_attrs=attrs)


def _ingest_in_tiledb(uri: str, data: np.ndarray, sparse: bool, num_attrs: int) -> None:
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
        attrs=[
            tiledb.Attr(name=f"features_{attr}", dtype=np.float32)
            for attr in range(num_attrs)
        ],
    )

    # Create the (empty) array on disk.
    tiledb.Array.create(uri, schema)

    # Ingest
    with tiledb.open(uri, "w") as tiledb_array:
        idx = np.nonzero(data) if sparse else slice(None)
        tiledb_array[idx] = {f"features_{attr}": data[idx] for attr in range(num_attrs)}


def rand_array(num_rows: int, *row_shape: int, sparse: bool = False) -> np.ndarray:
    """Create a random array of shape (num_rows, *row_shape).

    :param num_rows: Number of rows of the array (i.e. first dimension size).
    :param row_shape: Shape of each row (i.e. remaining dimension sizes).
    :param sparse:
      - If false, all values will be in the (0, 1) range.
      - If true, only `num_rows` values will be in the (0, 1) range, the rest will be 0.
        Note: some rows may be all zeros.
    """
    shape = (num_rows, *row_shape)
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
    for tensors in generator:
        assert len(tensors) == 2 * num_attrs
        # the first num_attrs tensors are the features (x)
        for tensor in tensors[:num_attrs]:
            _validate_tensor(tensor, x_sparse, x_shape, batch_size)
        # the last num_attrs tensors are the labels (y)
        for tensor in tensors[num_attrs:]:
            _validate_tensor(tensor, y_sparse, y_shape, batch_size)


def _validate_tensor(tensor, expected_sparse, expected_shape, batch_size=None):
    if batch_size is None:
        row_shape = tensor.shape
    else:
        num_rows, *row_shape = tensor.shape
        # num_rows may be less than batch_size
        assert num_rows <= batch_size, (num_rows, batch_size)
    assert tuple(row_shape) == expected_shape
    assert _is_sparse_tensor(tensor) == expected_sparse


def _is_sparse_tensor(tensor):
    if isinstance(tensor, np.ndarray):
        return False
    if isinstance(tensor, torch.Tensor):
        return tensor.is_sparse
    if isinstance(tensor, tf.SparseTensor):
        return True
    if isinstance(tensor, tf.Tensor):
        return False
    assert False, f"Unknown tensor type: {type(tensor)}"
