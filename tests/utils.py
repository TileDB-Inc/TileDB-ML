import itertools as it
import os
import uuid

import numpy as np
import pytest
import tensorflow as tf
import torch

import tiledb


def parametrize_for_dataset(
    sparse_x=(True, False),
    sparse_y=(True, False),
    input_shape=((10,), (10, 3)),
    num_attrs=(1, 2),
    pass_attrs=(True, False),
    buffer_size=(50, None),
    batch_shuffle=(True, False),
    within_batch_shuffle=(True, False),
):
    def is_valid_combination(t):
        sparse_x_, sparse_y_, input_shape_, *_, within_batch_shuffle_ = t
        # within_batch_shuffle not supported with sparse arrays
        if within_batch_shuffle_ and (sparse_x_ or sparse_y_):
            return False
        # sparse_x not supported with multi-dimensional arrays
        if sparse_x_ and len(input_shape_) > 1:
            return False
        return True

    argnames = [
        "sparse_x",
        "sparse_y",
        "input_shape",
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
            num_attrs,
            pass_attrs,
            buffer_size,
            batch_shuffle,
            within_batch_shuffle,
        ),
    )
    return pytest.mark.parametrize(argnames, argvalues)


def ingest_in_tiledb(tmpdir, data_x, data_y, sparse_x, sparse_y, batch_size, num_attrs):
    array_uuid = str(uuid.uuid4())
    uri_x = os.path.join(tmpdir, "x_" + array_uuid)
    uri_y = os.path.join(tmpdir, "y_" + array_uuid)
    _ingest_in_tiledb(uri_x, data_x, sparse_x, batch_size, num_attrs)
    _ingest_in_tiledb(uri_y, data_y, sparse_y, batch_size, num_attrs)
    return uri_x, uri_y


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


def create_rand_labels(
    num_rows: int, num_classes: int, one_hot: bool = False
) -> np.ndarray:
    """Create labels for `num_rows` observations with `num_classes` classes.

    :param num_rows: Number of observations to create labels for.
    :param num_classes: Number of possible labels.
    :param one_hot: Whether to create one-hot labels.

    :returns:
    - If one-hot is False, 1-D numpy array of length `num_rows` with labels from 0 to
      `num_classes`
    - If one-hot is True, binary 2-D numpy array of shape `(num_rows, num_classes)`.
    """
    labels = np.random.randint(num_classes, size=num_rows)
    return np.eye(num_classes, dtype=np.uint8)[labels] if one_hot else labels


def validate_tensor_generator(
    generator, num_attrs, batch_size, *, shape_x, shape_y, sparse_x, sparse_y
):
    for tensors in generator:
        assert len(tensors) == 2 * num_attrs
        # the first num_attrs tensors are the features (x)
        for tensor in tensors[:num_attrs]:
            _validate_tensor(tensor, batch_size, shape_x, sparse_x)
        # the last num_attrs tensors are the labels (y)
        for tensor in tensors[num_attrs:]:
            _validate_tensor(tensor, batch_size, shape_y, sparse_y)


def _validate_tensor(tensor, batch_size, shape, sparse):
    tensor_size, *tensor_shape = tensor.shape
    assert _is_sparse_tensor(tensor) == sparse
    # tensor size must be equal to batch_size for sparse but may be smaller for dense
    assert (tensor_size == batch_size) if sparse else (tensor_size <= batch_size)
    assert tuple(tensor_shape) == shape


def _is_sparse_tensor(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.is_sparse
    if isinstance(tensor, tf.SparseTensor):
        return True
    if isinstance(tensor, tf.Tensor):
        return False
    assert False, f"Unknown tensor type: {type(tensor)}"
