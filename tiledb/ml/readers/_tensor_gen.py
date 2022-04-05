from abc import ABC, abstractmethod
from typing import Generic, Iterator, Sequence, Type, TypeVar, Union

import numpy as np
import sparse

import tiledb

from ._batch_utils import iter_batches

Tensor = TypeVar("Tensor")


class TileDBNumpyGenerator:
    """Base class for generating tensors read from a TileDB array."""

    def __init__(self, array: tiledb.Array, attrs: Sequence[str]) -> None:
        """
        :param array: TileDB array to read from.
        :param attrs: Attribute names of array to read.
        """
        self._query = array.query(attrs=attrs)

    def read_buffer(self, array_slice: slice) -> None:
        """
        Read an array slice and save it as the current buffer.

        :param array_slice: Requested array slice.
        """
        self._buf_arrays = tuple(self._query[array_slice].values())

    def iter_tensors(self, buffer_slice: slice) -> Iterator[np.ndarray]:
        """
        Return an iterator of tensors for the given slice, one tensor per attribute

        Must be called after `read_buffer`.

        :param buffer_slice: Slice of the current buffer to convert to tensors.
        """
        return (buf_array[buffer_slice] for buf_array in self._buf_arrays)


class TileDBSparseTensorGenerator(TileDBNumpyGenerator, ABC, Generic[Tensor]):
    def __init__(self, array: tiledb.Array, attrs: Sequence[str]) -> None:
        self._dims = tuple(array.domain.dim(i).name for i in range(array.ndim))
        self._row_shape = array.shape[1:]
        super().__init__(array, attrs)

    def read_buffer(self, array_slice: slice) -> None:
        buffer = self._query[array_slice]
        coords = [buffer.pop(dim) for dim in self._dims]
        # normalize the first coordinate dimension to start at start_offset
        start_offset = array_slice.start
        if start_offset:
            coords[0] -= start_offset
        shape = (array_slice.stop - start_offset, *self._row_shape)
        self._buf_arrays = tuple(
            sparse.COO(coords, data, shape) for data in buffer.values()
        )

    def iter_tensors(self, buffer_slice: slice) -> Iterator[Tensor]:
        return map(self._tensor_from_coo, super().iter_tensors(buffer_slice))

    @staticmethod
    @abstractmethod
    def _tensor_from_coo(coo: sparse.COO) -> Tensor:
        """Convert a sparse.COO to a Tensor"""


def tensor_generator(
    x_array: tiledb.Array,
    y_array: tiledb.Array,
    x_buffer_size: int,
    y_buffer_size: int,
    x_attrs: Sequence[str],
    y_attrs: Sequence[str],
    sparse_generator_cls: Type[TileDBSparseTensorGenerator[Tensor]],
    start_offset: int = 0,
    stop_offset: int = 0,
) -> Iterator[Sequence[Union[np.ndarray, Tensor]]]:
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
    :param sparse_generator_cls: Sparse tensor generator type.
    """
    x_gen: Union[TileDBNumpyGenerator, TileDBSparseTensorGenerator[Tensor]] = (
        sparse_generator_cls(x_array, x_attrs)
        if x_array.schema.sparse
        else TileDBNumpyGenerator(x_array, x_attrs)
    )
    y_gen: Union[TileDBNumpyGenerator, TileDBSparseTensorGenerator[Tensor]] = (
        sparse_generator_cls(y_array, y_attrs)
        if y_array.schema.sparse
        else TileDBNumpyGenerator(y_array, y_attrs)
    )
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
