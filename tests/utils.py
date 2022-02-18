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
    sparse_x=(True, False),
    sparse_y=(True, False),
    input_shape=((10,), (10, 3)),
    output_shape=((5,), (5, 2)),
    num_attrs=(1, 2),
    pass_attrs=(True, False),
    buffer_size=(50, None),
    batch_shuffle=(True, False),
    within_batch_shuffle=(True, False),
):
    def is_valid_combination(t):
        sparse_x_, sparse_y_, input_shape_, output_shape_, *_, within_batch_shuffle_ = t
        # within_batch_shuffle not supported with sparse arrays
        if within_batch_shuffle_ and (sparse_x_ or sparse_y_):
            return False
        # sparse not supported with multi-dimensional arrays
        if sparse_x_ and len(input_shape_) > 1 or sparse_y_ and len(output_shape_) > 1:
            return False
        return True

    argnames = [
        "sparse_x",
        "sparse_y",
        "input_shape",
        "output_shape",
        "num_attrs",
        "pass_attrs",
        "buffer_size",
        "batch_shuffle",
        "within_batch_shuffle",
    ]
    argvalues = filter(
        is_valid_combination,
        it.product(
            sparse_x,
            sparse_y,
            input_shape,
            output_shape,
            num_attrs,
            pass_attrs,
            buffer_size,
            batch_shuffle,
            within_batch_shuffle,
        ),
    )
    return pytest.mark.parametrize(argnames, argvalues)


@contextmanager
def ingest_in_tiledb(
    tmpdir,
    data_x,
    data_y,
    sparse_x,
    sparse_y,
    batch_size,
    num_attrs,
    pass_attrs,
    buffer_size,
    batch_shuffle,
    within_batch_shuffle,
):
    """Context manager for ingest data into TileDB.

    Yield the keyword arguments for instantiating a TiledbDataset.
    """
    array_uuid = str(uuid.uuid4())
    uri_x = os.path.join(tmpdir, "x_" + array_uuid)
    uri_y = os.path.join(tmpdir, "y_" + array_uuid)
    _ingest_in_tiledb(uri_x, data_x, sparse_x, batch_size, num_attrs)
    _ingest_in_tiledb(uri_y, data_y, sparse_y, batch_size, num_attrs)
    attrs = [f"features_{attr}" for attr in range(num_attrs)] if pass_attrs else []
    with tiledb.open(uri_x) as x, tiledb.open(uri_y) as y:
        yield dict(
            x_array=x,
            y_array=y,
            batch_size=batch_size,
            buffer_size=buffer_size,
            batch_shuffle=batch_shuffle,
            within_batch_shuffle=within_batch_shuffle,
            x_attrs=attrs,
            y_attrs=attrs,
        )


def _ingest_in_tiledb(
    uri: str, data: np.ndarray, sparse: bool, batch_size: int, num_attrs: int
) -> None:
    dims = [
        tiledb.Dim(
            name=f"dim_{dim}",
            domain=(0, data.shape[dim] - 1),
            tile=data.shape[dim] if dim > 0 else batch_size,
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


def rand_array(num_rows: int, *row_shape: int, sparse=False) -> np.ndarray:
    """Create a random array of shape (num_rows, *row_shape).

    :param num_rows: Number of rows of the array (i.e. first dimension size).
    :param row_shape: Shape of each row (i.e. remaining dimension sizes).
    :param sparse: If True, the array will be sparse: exactly one element per row
        will be non-zero.
    """
    shape = (num_rows, *row_shape)
    if sparse:
        a = np.zeros((num_rows, np.asarray(row_shape).prod()))
        non_zero_coords = np.random.randint(a.shape[1], size=num_rows)
        a[np.arange(num_rows), non_zero_coords] = np.random.rand(num_rows)
        a = a.reshape(shape)
    else:
        a = np.random.random(shape)
    assert a.shape == shape
    return a


def validate_tensor_generator(
    generator, *, sparse_x, sparse_y, shape_x, shape_y, batch_size, num_attrs
):
    for tensors in generator:
        assert len(tensors) == 2 * num_attrs
        # the first num_attrs tensors are the features (x)
        for tensor in tensors[:num_attrs]:
            _validate_tensor(tensor, batch_size, sparse_x, shape_x)
        # the last num_attrs tensors are the labels (y)
        for tensor in tensors[num_attrs:]:
            _validate_tensor(tensor, batch_size, sparse_y, shape_y)


def _validate_tensor(tensor, batch_size, expected_sparse, expected_shape):
    num_rows, *row_shape = tensor.shape
    assert row_shape == list(expected_shape)
    if expected_sparse:
        assert _is_sparse_tensor(tensor)
        # for sparse tensors, num_rows should be equal to batch_size
        assert num_rows == batch_size
    else:
        assert not _is_sparse_tensor(tensor)
        # for dense tensors, num_rows may be less than batch_size
        assert num_rows <= batch_size


def _is_sparse_tensor(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.is_sparse
    if isinstance(tensor, tf.SparseTensor):
        return True
    if isinstance(tensor, tf.Tensor):
        return False
    assert False, f"Unknown tensor type: {type(tensor)}"
