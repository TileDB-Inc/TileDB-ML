from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Iterator, Optional, Sequence, Type, TypeVar, Union

import numpy as np
import scipy.sparse as sp

import tiledb

Tensor = TypeVar("Tensor")


class TileDBTensorGenerator(ABC, Generic[Tensor]):
    """Base class for generating tensors read from a TileDB array."""

    def __init__(self, array: tiledb.Array, attrs: Sequence[str]) -> None:
        """
        :param array: TileDB array to read from.
        :param attrs: Attribute names of array to read.
        """
        self._query = array.query(attrs=attrs)

    @abstractmethod
    def read_buffer(self, array_slice: slice) -> None:
        """
        Read an array slice and save it as the current buffer.

        :param array_slice: Requested array slice.
        """

    @abstractmethod
    def iter_tensors(self, buffer_slice: slice) -> Iterator[Tensor]:
        """
        Return an iterator of tensors for the given slice, one tensor per attribute

        Must be called after `read_buffer`.

        :param buffer_slice: Slice of the current buffer to convert to tensors.
        """


class TileDBNumpyGenerator(TileDBTensorGenerator[np.ndarray]):
    def read_buffer(self, array_slice: slice) -> None:
        self._buf_arrays = tuple(self._query[array_slice].values())

    def iter_tensors(self, buffer_slice: slice) -> Iterator[np.ndarray]:
        for buf_array in self._buf_arrays:
            yield buf_array[buffer_slice]


class SparseTileDBTensorGenerator(TileDBTensorGenerator[Tensor]):
    def __init__(self, array: tiledb.Array, attrs: Sequence[str]) -> None:
        schema = array.schema
        if schema.ndim != 2:
            raise NotImplementedError("Only 2D sparse tensors are currently supported")
        self._row_dim = schema.domain.dim(0).name
        self._col_dim = schema.domain.dim(1).name
        self._row_shape = schema.shape[1:]
        self._attr_dtypes = tuple(schema.attr(attr).dtype for attr in attrs)
        super().__init__(array, attrs)

    def read_buffer(self, array_slice: slice) -> None:
        buffer = self._query[array_slice]
        # COO to CSR transformation for batching and row slicing
        row = buffer.pop(self._row_dim)
        col = buffer.pop(self._col_dim)
        # Normalize indices: We want the coords indices to be in the [0, array_slice size]
        # range. If we do not normalize the sparse tensor is being created but with a
        # dimension [0, max(coord_index)], which is overkill
        start_offset = array_slice.start
        stop_offset = array_slice.stop
        shape = (stop_offset - start_offset, *self._row_shape)
        self._buf_csrs = tuple(
            sp.csr_matrix((data, (row - start_offset, col)), shape=shape)
            for data in buffer.values()
        )

    def iter_tensors(self, buffer_slice: slice) -> Iterator[Tensor]:
        for buf_csr, dtype in zip(self._buf_csrs, self._attr_dtypes):
            batch_csr = buf_csr[buffer_slice]
            batch_coo = batch_csr.tocoo()
            data = batch_coo.data
            coords = np.stack((batch_coo.row, batch_coo.col), axis=-1)
            dense_shape = (batch_csr.shape[0], *self._row_shape)
            yield self._tensor_from_coo(data, coords, dense_shape, dtype)

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
    x_array: tiledb.Array,
    y_array: tiledb.Array,
    x_buffer_size: int,
    y_buffer_size: int,
    x_attrs: Sequence[str],
    y_attrs: Sequence[str],
    start_offset: int = 0,
    stop_offset: int = 0,
    dense_tensor_generator_cls: Type[
        TileDBTensorGenerator[DenseTensor]
    ] = TileDBNumpyGenerator,
    sparse_tensor_generator_cls: Type[
        TileDBTensorGenerator[SparseTensor]
    ] = SparseTileDBTensorGenerator,
) -> Iterator[Sequence[Union[DenseTensor, SparseTensor]]]:
    """
    Generator for batches of tensors.

    Each yielded batch is a sequence of N tensors of x_array followed by M tensors
    of y_array, where `N == len(x_attrs)` and `M == len(y_attrs)`.

    :param x_array: TileDB array of the features.
    :param y_array: TileDB array of the labels.
    :param x_buffer_size: Number of rows to read at a time from x_array.
    :param y_buffer_size: Number of rows to read at a time from y_array.
    :param x_attrs: Attribute names of x_array.
    :param y_attrs: Attribute names of y_array.
    :param start_offset: Start row offset; defaults to 0.
    :param stop_offset: Stop row offset; defaults to number of rows.
    :param dense_tensor_generator_cls: Dense tensor generator type.
    :param sparse_tensor_generator_cls: Sparse tensor generator type.
    """

    def get_buffer_size_generator(
        array: tiledb.Array, attrs: Sequence[str]
    ) -> Union[TileDBTensorGenerator[DenseTensor], TileDBTensorGenerator[SparseTensor]]:
        if array.schema.sparse:
            return sparse_tensor_generator_cls(array, attrs)
        else:
            return dense_tensor_generator_cls(array, attrs)

    x_gen = get_buffer_size_generator(x_array, x_attrs)
    y_gen = get_buffer_size_generator(y_array, y_attrs)
    if not stop_offset:
        stop_offset = x_array.shape[0]
    for batch in iter_batches(x_buffer_size, y_buffer_size, start_offset, stop_offset):
        if batch.x_read_slice:
            x_gen.read_buffer(batch.x_read_slice)
        if batch.y_read_slice:
            y_gen.read_buffer(batch.y_read_slice)
        x_tensors = x_gen.iter_tensors(batch.x_buffer_slice)
        y_tensors = y_gen.iter_tensors(batch.y_buffer_slice)
        yield (*x_tensors, *y_tensors)


@dataclass(frozen=True, repr=False)
class Batch:
    size: int
    x_read_slice: Optional[slice]
    y_read_slice: Optional[slice]
    x_buffer_slice: slice
    y_buffer_slice: slice

    def __repr__(self) -> str:
        x, y = self.x_buffer_slice, self.y_buffer_slice
        s = f"Batch({self.size}, x[{x.start}:{x.stop}], y[{y.start}:{y.stop}]"
        if self.x_read_slice:
            s += f", x_read[{self.x_read_slice.start}:{self.x_read_slice.stop}]"
        if self.y_read_slice:
            s += f", y_read[{self.y_read_slice.start}:{self.y_read_slice.stop}]"
        return s + ")"


def iter_batches(
    x_buffer_size: int,
    y_buffer_size: int,
    start_offset: int,
    stop_offset: int,
) -> Iterator[Batch]:
    """
    Generate `Batch` instances describing each batch.

    Each yielded `Batch` instance describes:
    - Its size
    - The slice to read from the x array (if the current x buffer is consumed).
    - The slice to read from the y array (if the current y buffer is consumed).
    - The batch slice to read from the x buffer.
    - The batch slice to read from the y buffer.

    :param x_buffer_size: (Max) size of the x buffer.
    :param y_buffer_size: (Max) size of the y buffer.
    :param start_offset: Start row offset.
    :param stop_offset: Stop row offset.
    """
    x_read_slices = iter_slices(start_offset, stop_offset, x_buffer_size)
    y_read_slices = iter_slices(start_offset, stop_offset, y_buffer_size)
    x_buf_offset = x_buffer_size
    y_buf_offset = y_buffer_size
    offset = start_offset
    while offset < stop_offset:
        if x_buf_offset == x_buffer_size:
            x_read_slice = next(x_read_slices)
            x_read_size = x_read_slice.stop - x_read_slice.start
            x_buf_offset = 0
        else:
            x_read_slice = None

        if y_buf_offset == y_buffer_size:
            y_read_slice = next(y_read_slices)
            y_read_size = y_read_slice.stop - y_read_slice.start
            y_buf_offset = 0
        else:
            y_read_slice = None

        buffer_size = min(x_read_size - x_buf_offset, y_read_size - y_buf_offset)
        x_next_buf_offset = x_buf_offset + buffer_size
        y_next_buf_offset = y_buf_offset + buffer_size
        yield Batch(
            buffer_size,
            x_read_slice,
            y_read_slice,
            slice(x_buf_offset, x_next_buf_offset),
            slice(y_buf_offset, y_next_buf_offset),
        )
        x_buf_offset = x_next_buf_offset
        y_buf_offset = y_next_buf_offset
        offset += buffer_size


def iter_slices(start: int, stop: int, step: int) -> Iterator[slice]:
    offsets = range(start, stop, step)
    yield from map(slice, offsets, offsets[1:])
    yield slice(offsets[-1], stop)
