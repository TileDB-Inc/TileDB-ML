from typing import Iterator, Sequence

import numpy as np
import sparse

import tiledb


def iter_slices(start: int, stop: int, step: int) -> Iterator[slice]:
    offsets = range(start, stop, step)
    yield from map(slice, offsets, offsets[1:])
    yield slice(offsets[-1], stop)


class TileDBNumpyGenerator:
    """Generator of Numpy tensors read from a TileDB array."""

    def __init__(self, array: tiledb.Array, attrs: Sequence[str]) -> None:
        """
        :param array: TileDB array to read from.
        :param attrs: Attribute names of array to read.
        """
        self._array = array
        self._attrs = attrs

    def iter_tensors(
        self, buffer_size: int, start_offset: int, stop_offset: int
    ) -> Iterator[Sequence[np.ndarray]]:
        """
        Generate batches of tensors.

        Each yielded batch is a sequence of N tensors where `N == len(self.attrs)`.
        Each tensor is a NumPy array of shape `(buffer_size, *self.array.shape[1:])`.

        :param buffer_size: Size of each slice of rows to read.
        :param start_offset: Start row offset; defaults to 0.
        :param stop_offset: Stop row offset; defaults to number of rows.
        """
        query = self._array.query(attrs=self._attrs)
        for read_slice in iter_slices(start_offset, stop_offset, buffer_size):
            yield tuple(query[read_slice].values())


class TileDBSparseCOOGenerator:
    """Generator of sparse.COO tensors read from a TileDB array."""

    def __init__(self, array: tiledb.Array, attrs: Sequence[str]) -> None:
        self._array = array
        self._attrs = attrs
        self._dims = tuple(array.domain.dim(i).name for i in range(array.ndim))
        self._row_shape = array.shape[1:]

    def iter_tensors(
        self, buffer_size: int, start_offset: int, stop_offset: int
    ) -> Iterator[Sequence[sparse.COO]]:
        """
        Generate batches of tensors.

        Each yielded batch is a sequence of N tensors where `N == len(self.attrs)`.
        Each tensor is a `Tensor` of shape `(buffer_size, *self.array.shape[1:])`.

        :param buffer_size: Size of each slice of rows to read.
        :param start_offset: Start row offset; defaults to 0.
        :param stop_offset: Stop row offset; defaults to number of rows.
        """
        query = self._array.query(attrs=self._attrs)
        for read_slice in iter_slices(start_offset, stop_offset, buffer_size):
            buffer = query[read_slice]
            coords = [buffer.pop(dim) for dim in self._dims]
            # normalize the first coordinate dimension to start at start_offset
            start = read_slice.start
            if start:
                coords[0] -= start
            shape = (read_slice.stop - start, *self._row_shape)
            yield tuple(sparse.COO(coords, data, shape) for data in buffer.values())
