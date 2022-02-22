from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent import futures
from typing import Generic, Iterator, Optional, Sequence, Type, TypeVar, Union

import numpy as np
import scipy.sparse as sp

import tiledb

Tensor = TypeVar("Tensor")


class BaseBatch(ABC, Generic[Tensor]):
    """
    Base class used for getting tensors from batches of buffers read from a TileDB array.
    """

    def __init__(
        self, schema: tiledb.ArraySchema, attrs: Sequence[str], batch_size: int
    ):
        self._schema = schema
        self._attrs = attrs
        self._batch_size = batch_size

    @abstractmethod
    def read_buffer(self, array: tiledb.Array, buffer_slice: slice) -> None:
        """Read a slice from a TileDB array into a buffer.

        :param array: TileDB array to read from.
        :param buffer_slice: Slice of the array to read.
        """

    @abstractmethod
    def set_batch_slice(self, batch_slice: slice) -> None:
        """Set the current batch as a slice of the read buffer.

        Must be called after `read_buffer`.

        :param batch_slice: Slice of the buffer to be used as the current batch.
        """

    @abstractmethod
    def iter_tensors(self, perm_idxs: Optional[np.ndarray] = None) -> Iterator[Tensor]:
        """
        Return an iterator of tensors for the current batch, one tensor per attribute
        given in the constructor.

        Must be called after `set_batch_slice`.

        :param perm_idxs: Optional permutation indices for the current batch.
            If given, it must be an array equal to `np.arange(len(self))` after sorting.
        """

    @abstractmethod
    def __len__(self) -> int:
        """Get the size (i.e. number of rows) of the current batch.

        Must be called after `set_batch_slice`.
        """


class BaseDenseBatch(BaseBatch[Tensor]):
    def read_buffer(self, array: tiledb.Array, buffer_slice: slice) -> None:
        self._buffer = array.query(dims=(), attrs=self._attrs)[buffer_slice]

    def set_batch_slice(self, batch_slice: slice) -> None:
        assert hasattr(self, "_buffer"), "read_buffer() not called"
        self._attr_batches = tuple(
            self._buffer[attr][batch_slice] for attr in self._attrs
        )

    def iter_tensors(self, perm_idxs: Optional[np.ndarray] = None) -> Iterator[Tensor]:
        assert hasattr(self, "_attr_batches"), "set_batch_slice() not called"
        if perm_idxs is None:
            attr_batches = iter(self._attr_batches)
        else:
            n = len(self)
            assert (
                len(perm_idxs) == n and (np.sort(perm_idxs) == np.arange(n)).all()
            ), f"Invalid permutation of size {n}: {perm_idxs}"
            attr_batches = (attr_batch[perm_idxs] for attr_batch in self._attr_batches)
        return map(self._tensor_from_numpy, attr_batches)

    def __len__(self) -> int:
        assert hasattr(self, "_attr_batches"), "set_batch_slice() not called"
        return len(self._attr_batches[0])

    @staticmethod
    @abstractmethod
    def _tensor_from_numpy(data: np.ndarray) -> Tensor:
        """Convert a numpy array to a Tensor"""


class BaseSparseBatch(BaseBatch[Tensor]):
    def __init__(
        self, schema: tiledb.ArraySchema, attrs: Sequence[str], batch_size: int
    ):
        if schema.ndim != 2:
            raise NotImplementedError("Sparse batches only supported for 2D arrays")
        super().__init__(schema, attrs, batch_size)
        self._row_dim = schema.domain.dim(0).name
        self._col_dim = schema.domain.dim(1).name
        self._dense_shape = (batch_size, schema.shape[1])
        self._attr_dtypes = tuple(schema.attr(attr).dtype for attr in self._attrs)

    def read_buffer(self, array: tiledb.Array, buffer_slice: slice) -> None:
        buffer = array.query(attrs=self._attrs)[buffer_slice]
        # COO to CSR transformation for batching and row slicing
        row = buffer[self._row_dim]
        col = buffer[self._col_dim]
        # Normalize indices: We want the coords indices to be in the [0, batch_size]
        # range. If we do not normalize the sparse tensor is being created but with a
        # dimension [0, max(coord_index)], which is overkill
        offset = buffer_slice.start
        self._buffer_csrs = tuple(
            sp.csr_matrix((buffer[attr], (row - offset, col))) for attr in self._attrs
        )

    def set_batch_slice(self, batch_slice: slice) -> None:
        assert hasattr(self, "_buffer_csrs"), "read_buffer() not called"
        self._batch_csrs = tuple(
            buffer_csr[batch_slice] for buffer_csr in self._buffer_csrs
        )

    def iter_tensors(self, perm_idxs: Optional[np.ndarray] = None) -> Iterator[Tensor]:
        assert hasattr(self, "_batch_csrs"), "set_batch_slice() not called"
        if perm_idxs is not None:
            raise NotImplementedError(
                "within_batch_shuffle not implemented for sparse arrays"
            )
        for batch_csr, dtype in zip(self._batch_csrs, self._attr_dtypes):
            batch_coo = batch_csr.tocoo()
            data = batch_coo.data
            coords = np.stack((batch_coo.row, batch_coo.col), axis=-1)
            yield self._tensor_from_coo(data, coords, self._dense_shape, dtype)

    def __len__(self) -> int:
        assert hasattr(self, "_batch_csrs"), "set_batch_slice() not called"
        # return number of non-zero rows
        lengths = {
            int((batch_csr.getnnz(axis=1) > 0).sum()) for batch_csr in self._batch_csrs
        }
        assert len(lengths) == 1, f"Multiple different batch lengths: {lengths}"
        return lengths.pop()

    def __bool__(self) -> bool:
        assert hasattr(self, "_batch_csrs"), "set_batch_slice() not called"
        # faster version of __len__() > 0
        lengths = {len(batch_csr.data) for batch_csr in self._batch_csrs}
        assert len(lengths) == 1, f"Multiple different batch lengths: {lengths}"
        return lengths.pop() > 0

    @staticmethod
    @abstractmethod
    def _tensor_from_coo(
        data: np.ndarray,
        coords: np.ndarray,
        dense_shape: Sequence[int],
        dtype: np.dtype,
    ) -> Tensor:
        """Convert a scipy.sparse.coo_matrix to a Tensor"""


DenseTensor = TypeVar("DenseTensor")
SparseTensor = TypeVar("SparseTensor")


def tensor_generator(
    dense_batch_cls: Type[BaseDenseBatch[DenseTensor]],
    sparse_batch_cls: Type[BaseSparseBatch[SparseTensor]],
    x_array: tiledb.Array,
    y_array: tiledb.Array,
    batch_size: int,
    buffer_size: int,
    batch_shuffle: bool = False,
    within_batch_shuffle: bool = False,
    x_attrs: Sequence[str] = (),
    y_attrs: Sequence[str] = (),
    start_offset: int = 0,
    stop_offset: Optional[int] = None,
) -> Iterator[Sequence[Union[DenseTensor, SparseTensor]]]:
    """
    Generator for batches of tensors.

    Each yielded batch is a sequence of N tensors of x_array followed by M tensors
    of y_array, where `N == len(x_attrs)` and `M == len(y_attrs)`.

    :param dense_batch_cls: Type of dense batches.
    :param sparse_batch_cls: Type of sparse batches.
    :param x_array: TileDB array of the features.
    :param y_array: TileDB array of the labels.
    :param batch_size: Size of each batch.
    :param buffer_size: Size of the buffer used to read the data.
    :param batch_shuffle: True for shuffling batches.
    :param within_batch_shuffle: True for shuffling records in each batch.
    :param x_attrs: Attribute names of x_array; defaults to all x_array attributes.
    :param y_attrs: Attribute names of y_array; defaults to all y_array attributes
    :param start_offset: Start row offset; defaults to 0.
    :param stop_offset: Stop row offset; defaults to number of rows.
    """
    if stop_offset is None:
        stop_offset = x_array.shape[0]

    def batch_factory(
        schema: tiledb.ArraySchema, attrs: Sequence[str]
    ) -> Union[BaseDenseBatch[DenseTensor], BaseSparseBatch[SparseTensor]]:
        if not attrs:
            attrs = get_attr_names(schema)
        if schema.sparse:
            return sparse_batch_cls(schema, attrs, batch_size)
        return dense_batch_cls(schema, attrs, batch_size)

    x_batch = batch_factory(x_array.schema, x_attrs)
    y_batch = batch_factory(y_array.schema, y_attrs)
    with futures.ThreadPoolExecutor(max_workers=2) as executor:
        for offset in range(start_offset, stop_offset, buffer_size):
            buffer_slice = slice(offset, offset + buffer_size)
            futures.wait(
                (
                    executor.submit(x_batch.read_buffer, x_array, buffer_slice),
                    executor.submit(y_batch.read_buffer, y_array, buffer_slice),
                )
            )
            # Split the buffer_size into batch_size chunks
            batch_offsets = np.arange(
                0, min(buffer_size, stop_offset - offset), batch_size
            )
            if batch_shuffle:
                np.random.shuffle(batch_offsets)

            for batch_offset in batch_offsets:
                batch_slice = slice(batch_offset, batch_offset + batch_size)
                x_batch.set_batch_slice(batch_slice)
                y_batch.set_batch_slice(batch_slice)
                if len(x_batch) != len(y_batch):
                    raise ValueError(
                        "x and y batches should have the same length: "
                        f"len(x_batch)={len(x_batch)}, len(y_batch)={len(y_batch)}"
                    )
                if x_batch:
                    if within_batch_shuffle:
                        perm_idxs = np.arange(len(x_batch))
                        np.random.shuffle(perm_idxs)
                    else:
                        perm_idxs = None
                    yield (
                        *x_batch.iter_tensors(perm_idxs),
                        *y_batch.iter_tensors(perm_idxs),
                    )


def get_attr_names(schema: tiledb.ArraySchema) -> Sequence[str]:
    return tuple(schema.attr(idx).name for idx in range(schema.nattr))


def get_buffer_size(buffer_size: Optional[int], batch_size: int) -> int:
    if buffer_size is None:
        buffer_size = batch_size
    elif buffer_size < batch_size:
        raise ValueError("buffer_size must be >= batch_size")
    return buffer_size
