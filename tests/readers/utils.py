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
from tiledb.ml.readers.types import ArrayParams, TensorKind


@dataclass(frozen=True)
class ArraySpec:
    sparse: bool
    shape: Sequence[int]
    key_dim: int
    key_dim_dtype: np.dtype
    non_key_dim_dtype: np.dtype
    num_fields: int

    def tensor_kind(self, supports_csr: bool) -> TensorKind:
        if not self.sparse:
            return TensorKind.DENSE
        elif not np.issubdtype(self.non_key_dim_dtype, np.integer):
            return TensorKind.RAGGED
        elif len(self.shape) == 2 and supports_csr:
            return TensorKind.SPARSE_CSR
        else:
            return TensorKind.SPARSE_COO


def parametrize_for_dataset(
    *,
    x_sparse=(True, False),
    y_sparse=(True, False),
    x_shape=((107, 10), (107, 10, 3)),
    y_shape=((107, 5), (107, 5, 2)),
    x_key_dim=(0, 1),
    y_key_dim=(0, 1),
    key_dim_dtype=(np.dtype(np.int32), np.dtype("datetime64[D]"), np.dtype(np.bytes_)),
    non_key_dim_dtype=(np.dtype(np.int32), np.dtype(np.float32)),
    num_fields=(0, 1, 2),
    batch_size=(8,),
    shuffle_buffer_size=(16,),
    num_workers=(0, 2),
):
    argnames = ("x_spec", "y_spec", "batch_size", "shuffle_buffer_size", "num_workers")
    argvalues = []
    for (
        x_sparse_,
        y_sparse_,
        x_shape_,
        y_shape_,
        x_key_dim_,
        y_key_dim_,
        key_dim_dtype_,
        non_key_dim_dtype_,
        num_fields_,
        batch_size_,
        shuffle_buffer_size_,
        num_workers_,
    ) in it.product(
        x_sparse,
        y_sparse,
        x_shape,
        y_shape,
        x_key_dim,
        y_key_dim,
        key_dim_dtype,
        non_key_dim_dtype,
        num_fields,
        batch_size,
        shuffle_buffer_size,
        num_workers,
    ):
        # if x and/or y is dense, all dtypes must be integer
        if not x_sparse_ or not y_sparse_:
            if not np.issubdtype(key_dim_dtype_, np.integer):
                continue
            if not np.issubdtype(non_key_dim_dtype_, np.integer):
                continue

        common_args = (key_dim_dtype_, non_key_dim_dtype_, num_fields_)
        x_spec = ArraySpec(x_sparse_, x_shape_, x_key_dim_, *common_args)
        y_spec = ArraySpec(y_sparse_, y_shape_, y_key_dim_, *common_args)
        argvalues.append(
            (x_spec, y_spec, batch_size_, shuffle_buffer_size_, num_workers_)
        )

    return pytest.mark.parametrize(argnames, argvalues)


@contextmanager
def ingest_in_tiledb(tmpdir, spec: ArraySpec):
    """Context manager for ingesting data into TileDB."""
    array_uuid = str(uuid.uuid4())
    uri = os.path.join(tmpdir, array_uuid)
    data = original_data = _rand_array(spec.shape, spec.sparse)
    if spec.key_dim > 0:
        data = np.moveaxis(data, 0, spec.key_dim)
    data_idx = np.arange(data.size).reshape(data.shape)

    transforms = []
    for i in range(data.ndim):
        n = data.shape[i]
        dtype = spec.key_dim_dtype if i == spec.key_dim else spec.non_key_dim_dtype
        if np.issubdtype(dtype, np.number):
            # set the domain to (-n/2, n/2) to test negative domain indexing
            min_value = -(n // 2)
        elif np.issubdtype(dtype, np.datetime64):
            min_value = np.datetime64("2022-06-15")
        elif np.issubdtype(dtype, np.bytes_):
            min_value = b"a"
        else:
            assert False, dtype
        transforms.append(_IndexTransformer(f"dim_{i}", n, min_value, dtype))

    dims = [transform.dim for transform in transforms]
    attrs = [
        tiledb.Attr(name="data", dtype=np.float32),
        tiledb.Attr(name="idx", dtype=np.int16),
    ]
    schema = tiledb.ArraySchema(
        domain=tiledb.Domain(*dims), attrs=attrs, sparse=spec.sparse
    )
    tiledb.Array.create(uri, schema)

    with tiledb.open(uri, "w") as tiledb_array:
        if spec.sparse:
            nz_idxs = np.nonzero(data)
            dim_idxs = tuple(
                transform(idx) for transform, idx in zip(transforms, nz_idxs)
            )
            tiledb_array[dim_idxs] = {"data": data[nz_idxs], "idx": data_idx[nz_idxs]}
        else:
            tiledb_array[:] = {"data": data, "idx": data_idx}

    all_fields = [f.name for f in dims + attrs]
    # exclude the key dimension from the fields if it is not an integer
    if not np.issubdtype(spec.key_dim_dtype, np.integer):
        del all_fields[spec.key_dim]
    fields = np.random.choice(all_fields, size=spec.num_fields, replace=False).tolist()

    with tiledb.open(uri) as array:
        yield ArrayParams(array, spec.key_dim, fields), original_data


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


def validate_tensor_generator(generator, x_spec, y_spec, batch_size, supports_csr):
    for x_tensors, y_tensors in generator:
        for x_tensor in x_tensors if isinstance(x_tensors, Sequence) else [x_tensors]:
            _validate_tensor(x_tensor, x_spec, batch_size, supports_csr)
        for y_tensor in y_tensors if isinstance(y_tensors, Sequence) else [y_tensors]:
            _validate_tensor(y_tensor, y_spec, batch_size, supports_csr)


def _validate_tensor(tensor, spec, batch_size, supports_csr):
    tensor_kind = _get_tensor_kind(tensor)
    assert tensor_kind is spec.tensor_kind(supports_csr)

    spec_row_shape = spec.shape[1:]
    if tensor_kind is not TensorKind.RAGGED:
        num_rows, *row_shape = tensor.shape
        assert tuple(row_shape) == spec_row_shape
    else:
        # every ragged array row has at most `np.prod(spec_row_shape)` elements,
        # the product of all non-key dimension sizes
        row_lengths = tuple(map(len, tensor))
        assert all(row_length <= np.prod(spec_row_shape) for row_length in row_lengths)
        num_rows = len(row_lengths)

    # num_rows may be less than batch_size
    assert num_rows <= batch_size, (num_rows, batch_size)


def _get_tensor_kind(tensor) -> TensorKind:
    if isinstance(tensor, tf.Tensor):
        return TensorKind.DENSE
    if isinstance(tensor, torch.Tensor):
        if getattr(tensor, "is_nested", False):
            return TensorKind.RAGGED
        return _torch_tensor_layout_to_kind[tensor.layout]
    return _tensor_type_to_kind[type(tensor)]


_tensor_type_to_kind = {
    np.ndarray: TensorKind.DENSE,
    sparse.COO: TensorKind.SPARSE_COO,
    scipy.sparse.coo_matrix: TensorKind.SPARSE_COO,
    scipy.sparse.csr_matrix: TensorKind.SPARSE_CSR,
    tf.SparseTensor: TensorKind.SPARSE_COO,
    tf.RaggedTensor: TensorKind.RAGGED,
}

_torch_tensor_layout_to_kind = {
    torch.strided: TensorKind.DENSE,
    torch.sparse_coo: TensorKind.SPARSE_COO,
    torch.sparse_csr: TensorKind.SPARSE_CSR,
}
