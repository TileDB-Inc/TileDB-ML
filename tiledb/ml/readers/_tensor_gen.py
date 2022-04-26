from abc import ABC, abstractmethod
from operator import itemgetter
from typing import Callable, Generic, Iterator, Sequence, TypeVar, Union

import numpy as np
import sparse

import tiledb

T = TypeVar("T")


class TileDBTensorGenerator(ABC, Generic[T]):
    """Generator of tensors read from a TileDB array."""

    def __init__(self, array: tiledb.Array, attrs: Sequence[str]) -> None:
        """
        :param array: TileDB array to read from.
        :param attrs: Attribute names of array to read.
        """
        self._array = array
        self._attrs = attrs

    @property
    def single_attr(self) -> bool:
        """Whether this generator reads a single attribute."""
        return len(self._attrs) == 1

    @abstractmethod
    def iter_tensors(
        self, buffer_size: int, start_offset: int, stop_offset: int
    ) -> Union[Iterator[T], Iterator[Sequence[T]]]:
        """
        Generate batches of tensors.

        Each yielded batch is either:
        - a sequence of N tensors where `N == len(self.attrs)` if N > 1, or
        - a single tensor if N == 1.
        Each tensor is a `T` instance of shape `(buffer_size, *self.array.shape[1:])`.

        :param buffer_size: Size of each slice of rows to read.
        :param start_offset: Start row offset; defaults to 0.
        :param stop_offset: Stop row offset; defaults to number of rows.
        """


class TileDBNumpyGenerator(TileDBTensorGenerator[np.ndarray]):
    """Generator of Numpy arrays read from a TileDB array."""

    def iter_tensors(
        self, buffer_size: int, start_offset: int, stop_offset: int
    ) -> Union[Iterator[np.ndarray], Iterator[Sequence[np.ndarray]]]:
        query = self._array.query(attrs=self._attrs)
        read_slices = iter_slices(start_offset, stop_offset, buffer_size)
        buffers = map(query.__getitem__, read_slices)
        return map(itemgetter(*self._attrs), buffers)


class TileDBSparseGenerator(TileDBTensorGenerator[T]):
    """Generator of sparse.COO tensors read from a TileDB array."""

    def __init__(
        self,
        array: tiledb.Array,
        attrs: Sequence[str],
        from_coo: Callable[[sparse.COO], T],
    ) -> None:
        super().__init__(array, attrs)
        self._from_coo = from_coo
        self._dims = tuple(array.domain.dim(i).name for i in range(array.ndim))

    def iter_tensors(
        self, buffer_size: int, start_offset: int, stop_offset: int
    ) -> Union[Iterator[T], Iterator[Sequence[T]]]:
        shape = list(self._array.shape)
        query = self._array.query(attrs=self._attrs)
        get_coords = itemgetter(*self._dims)
        get_data = itemgetter(*self._attrs)
        single_attr = self.single_attr
        for read_slice in iter_slices(start_offset, stop_offset, buffer_size):
            shape[0] = read_slice.stop - read_slice.start
            buffer = query[read_slice]
            coords = get_coords(buffer)
            # normalize the first coordinate dimension to start at start_offset
            np.subtract(coords[0], read_slice.start, out=coords[0])
            data = get_data(buffer)
            if single_attr:
                yield self._from_coo(sparse.COO(coords, data, shape))
            else:
                yield tuple(self._from_coo(sparse.COO(coords, d, shape)) for d in data)


def iter_slices(start: int, stop: int, step: int) -> Iterator[slice]:
    offsets = range(start, stop, step)
    yield from map(slice, offsets, offsets[1:])
    yield slice(offsets[-1], stop)
