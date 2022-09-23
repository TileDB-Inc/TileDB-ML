import itertools as it
import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
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
    tensor_kind: Optional[TensorKind]

    @staticmethod
    def combinations(
        *,
        sparse=(True, False),
        shape=((107, 10), (107, 10, 3)),
        key_dim=(0, 1),
        key_dim_dtype=(
            np.dtype(np.int32),
            np.dtype("datetime64[D]"),
            np.dtype(np.bytes_),
        ),
        non_key_dim_dtype=(np.dtype(np.int32), np.dtype(np.float32)),
        num_fields=(0, 1),
        tensor_kind=(None, TensorKind.DENSE),
    ):
        for combo in it.product(
            sparse,
            shape,
            key_dim,
            key_dim_dtype,
            non_key_dim_dtype,
            num_fields,
            tensor_kind,
        ):
            spec = ArraySpec(*combo)
            if not spec.sparse:  # dense array
                # all dimension dtypes must be integer
                if not all(
                    np.issubdtype(dtype, np.integer)
                    for dtype in (spec.key_dim_dtype, spec.non_key_dim_dtype)
                ):
                    continue
                # tensor_kind must be DENSE or None (which defaults to DENSE here)
                # don't need to check both DENSE and None, just one of them
                if spec.tensor_kind is not None:
                    continue
            else:  # sparse array
                # for non-RAGGED tensors, non_key_dim_dtype must be integer
                if spec.tensor_kind not in (None, TensorKind.RAGGED):
                    if not np.issubdtype(spec.non_key_dim_dtype, np.integer):
                        continue
            yield spec


@contextmanager
def ingest_in_tiledb(tmpdir, spec):
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
        params = ArrayParams(array, spec.key_dim, fields, tensor_kind=spec.tensor_kind)
        yield params, original_data


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
            tile=np.random.randint(1, self.size),
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


def validate_tensor_generator(generator, schemas, specs, batch_size):
    if not isinstance(schemas, Sequence):
        schemas = [schemas]
    if not isinstance(specs, Sequence):
        specs = [specs]
    for batches in generator:
        if len(schemas) == 1:
            batches = [batches]
        assert len(schemas) == len(specs) == len(batches)
        for schema, spec, batch in zip(schemas, specs, batches):
            if not isinstance(batch, Sequence):
                batch = [batch]
            for tensor in batch:
                _validate_tensor(tensor, schema.kind, spec.shape[1:], batch_size)


def _validate_tensor(tensor, schema_tensor_kind, spec_row_shape, batch_size):
    tensor_kind = get_tensor_kind(tensor)
    if schema_tensor_kind is not TensorKind.RAGGED:
        assert tensor_kind is schema_tensor_kind
        if batch_size is not None:
            # batched tensor: the length of the first dimension is the size of the batch,
            # which may be smaller than the requested batch_size
            num_rows, *row_shape = tensor.shape
            assert num_rows <= batch_size, (num_rows, batch_size)
        elif schema_tensor_kind is TensorKind.SPARSE_CSR and len(spec_row_shape) == 1:
            # unbatched CSR 1D tensor has (1, N) shape
            num_rows, *row_shape = tensor.shape
            assert num_rows == 1, num_rows
        else:
            # unbatched non-CSR tensor: num_rows is implicitly 1
            row_shape = tensor.shape
        assert tuple(row_shape) == spec_row_shape
    else:
        # every ragged tensor row has at most `max_row_len` elements, the product of all
        # non-key dimension sizes
        max_row_len = np.prod(spec_row_shape)
        if batch_size is not None:
            assert tensor_kind is schema_tensor_kind
            row_lens = tuple(map(len, tensor))
            assert all(row_len <= max_row_len for row_len in row_lens)
            num_rows = len(row_lens)
            assert num_rows <= batch_size, (num_rows, batch_size)
        else:
            # in case of no batching, a 1D dense tensor is returned instead of a ragged
            # tensor with a single nested tensor
            assert tensor_kind is TensorKind.DENSE
            assert len(tensor) <= max_row_len


def assert_tensors_almost_equal_array(
    batches, array, schema_tensor_kind, batch_size, to_dense
):
    if schema_tensor_kind is TensorKind.RAGGED:
        tensor_kind = get_tensor_kind(batches[0])
        if tensor_kind is TensorKind.RAGGED:
            tensors = [tensor for batch in batches for tensor in batch]
        else:
            assert tensor_kind is TensorKind.DENSE
            tensors = batches
        # compare each tensor with the non-zero values of the respective array row
        assert len(tensors) == len(array)
        for tensor_row, row in zip(tensors, array):
            np.testing.assert_array_almost_equal(tensor_row, row[np.nonzero(row)])
    else:
        if schema_tensor_kind in (TensorKind.SPARSE_COO, TensorKind.SPARSE_CSR):
            batches = list(map(to_dense, batches))
        if batch_size is not None or schema_tensor_kind is TensorKind.SPARSE_CSR:
            np.testing.assert_array_almost_equal(np.concatenate(batches), array)
        else:
            np.testing.assert_array_almost_equal(np.stack(batches), array)


def get_tensor_kind(tensor) -> TensorKind:
    if isinstance(tensor, tf.Tensor):
        return TensorKind.DENSE
    if isinstance(tensor, tf.SparseTensor):
        return TensorKind.SPARSE_COO
    if isinstance(tensor, tf.RaggedTensor):
        return TensorKind.RAGGED
    if isinstance(tensor, torch.Tensor):
        if getattr(tensor, "is_nested", False):
            return TensorKind.RAGGED
        if tensor.layout is torch.strided:
            return TensorKind.DENSE
        if tensor.layout is torch.sparse_coo:
            return TensorKind.SPARSE_COO
        if tensor.layout is torch.sparse_csr:
            return TensorKind.SPARSE_CSR
    assert False, f"{tensor} is not a tensor"
