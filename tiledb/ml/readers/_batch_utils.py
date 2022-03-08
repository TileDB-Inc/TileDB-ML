from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from concurrent import futures
from dataclasses import dataclass
from typing import (
    Dict,
    Generic,
    Iterator,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
import scipy.sparse as sp

import tiledb

Tensor = TypeVar("Tensor")


class BaseBatch(ABC, Generic[Tensor]):
    """
    Base class used for getting tensors from batches of buffers read from a TileDB array.
    """

    def __init__(
        self,
        buffer_slices: Iterator[slice],
        array: tiledb.Array,
        attrs: Sequence[str] = (),
    ) -> None:
        """
        :param buffer_slices: Iterator of slices to be read.
        :param array: TileDB array to read from.
        :param attrs: Attribute names of array to read; defaults to all array attributes.
        """
        self._query = array.query(attrs=attrs or None)
        self._buf_slices = buffer_slices
        self._last_buf_slice = slice(0, 0)

    @abstractmethod
    def ensure_buffer(self, batch_slice: slice) -> int:
        """
        Ensure that the current buffer contains the given batch_slice, otherwise read
        the next buffer from buffer_slices.

        :param batch_slice: Requested batch slice in absolute row coordinates.
        :return: Number of rows of the buffer just read, or 0 if no buffer was read.
        """

    @abstractmethod
    def shuffle_buffer(self, row_idxs: np.ndarray) -> None:
        """
        Shuffle the current buffer.

        Must be called after `ensure_buffer`.

        :param row_idxs: Array of row indices to shuffle.
        """

    @abstractmethod
    def iter_tensors(self, batch_slice: slice) -> Iterator[Tensor]:
        """
        Return an iterator of tensors for the given batch_slice, one tensor per attribute

        Must be called after `ensure_buffer`.
        """

    def _read_next_buffer(
        self, batch_slice: slice
    ) -> Tuple[Optional[Dict[str, np.ndarray]], int]:
        """
        Read the next buffer if needed in order to access the given batch_slice.

        :param batch_slice: Requested batch slice in absolute row coordinates.
        :return: (buffer, size) tuple if the buffer is read or (None, 0) otherwise.
        """
        if slice_subsumes(self._last_buf_slice, batch_slice):
            return None, 0

        self._last_buf_slice = next(self._buf_slices)
        assert slice_subsumes(self._last_buf_slice, batch_slice)
        buffer = self._query[self._last_buf_slice]
        return buffer, self._last_buf_slice.stop - self._last_buf_slice.start

    def _translate_slice(self, batch_slice: slice) -> slice:
        """Translate batch_slice to buffer coordinates.

        :param batch_slice: Requested batch slice in absolute row coordinates.
        """
        buf_start = self._last_buf_slice.start
        return slice(
            batch_slice.start - buf_start,
            batch_slice.stop - buf_start,
            batch_slice.step,
        )


class BaseDenseBatch(BaseBatch[Tensor]):
    def ensure_buffer(self, batch_slice: slice) -> int:
        buffer, buffer_size = self._read_next_buffer(batch_slice)
        if buffer is not None:
            self._buf_arrays = tuple(buffer.values())
        return buffer_size

    def shuffle_buffer(self, row_idxs: np.ndarray) -> None:
        for buf_array in self._buf_arrays:
            buf_array[: len(row_idxs)] = buf_array[row_idxs]

    def iter_tensors(self, batch_slice: slice) -> Iterator[Tensor]:
        buf_slice = self._translate_slice(batch_slice)
        for buf_array in self._buf_arrays:
            yield self._tensor_from_numpy(buf_array[buf_slice])

    @staticmethod
    @abstractmethod
    def _tensor_from_numpy(data: np.ndarray) -> Tensor:
        """Convert a numpy array to a Tensor"""


class BaseSparseBatch(BaseBatch[Tensor]):
    def __init__(
        self,
        buffer_slices: Iterator[slice],
        array: tiledb.Array,
        attrs: Sequence[str] = (),
    ) -> None:
        schema = array.schema
        if schema.ndim != 2:
            raise NotImplementedError("Sparse batches only supported for 2D arrays")
        if not attrs:
            attrs = get_attr_names(schema)
        self._row_dim = schema.domain.dim(0).name
        self._col_dim = schema.domain.dim(1).name
        self._row_shape = schema.shape[1:]
        self._attr_dtypes = tuple(schema.attr(attr).dtype for attr in attrs)
        super().__init__(buffer_slices, array, attrs)

    def ensure_buffer(self, batch_slice: slice) -> int:
        buffer, buffer_size = self._read_next_buffer(batch_slice)
        if buffer is not None:
            # COO to CSR transformation for batching and row slicing
            row = buffer.pop(self._row_dim)
            col = buffer.pop(self._col_dim)
            # Normalize indices: We want the coords indices to be in the [0, batch_size]
            # range. If we do not normalize the sparse tensor is being created but with a
            # dimension [0, max(coord_index)], which is overkill
            offset = self._last_buf_slice.start
            shape = (buffer_size, *self._row_shape)
            self._buf_csrs = tuple(
                sp.csr_matrix((data, (row - offset, col)), shape=shape)
                for data in buffer.values()
            )
        return buffer_size

    def shuffle_buffer(self, row_idxs: np.ndarray) -> None:
        for buf_csr in self._buf_csrs:
            buf_csr[: len(row_idxs)] = buf_csr[row_idxs]

    def iter_tensors(self, batch_slice: slice) -> Iterator[Tensor]:
        buf_slice = self._translate_slice(batch_slice)
        for buf_csr, dtype in zip(self._buf_csrs, self._attr_dtypes):
            batch_csr = buf_csr[buf_slice]
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
    dense_batch_cls: Type[BaseDenseBatch[DenseTensor]],
    sparse_batch_cls: Type[BaseSparseBatch[SparseTensor]],
    x_array: tiledb.Array,
    y_array: tiledb.Array,
    batch_size: int,
    buffer_bytes: Optional[int] = None,
    batch_shuffle: bool = False,
    within_batch_shuffle: bool = False,
    x_attrs: Sequence[str] = (),
    y_attrs: Sequence[str] = (),
    start_offset: int = 0,
    stop_offset: int = 0,
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
    :param buffer_bytes: Size (in bytes) of the buffer used to read from each array.
        If not given, it is determined automatically.
    :param batch_shuffle: True for shuffling batches.
    :param within_batch_shuffle: True for shuffling records in each batch.
    :param x_attrs: Attribute names of x_array; defaults to all x_array attributes.
    :param y_attrs: Attribute names of y_array; defaults to all y_array attributes
    :param start_offset: Start row offset; defaults to 0.
    :param stop_offset: Stop row offset; defaults to number of rows.
    """
    if not stop_offset:
        stop_offset = x_array.shape[0]

    def batch_factory(
        label: str, array: tiledb.Array, attrs: Sequence[str]
    ) -> Union[BaseDenseBatch[DenseTensor], BaseSparseBatch[SparseTensor]]:
        if buffer_bytes is None:
            num_batches = 1
        else:
            num_batches = get_num_batches(
                batch_size, buffer_bytes, array, attrs, start_offset, stop_offset
            )
        buffer_size = batch_size * num_batches
        logging.info(f"{label} buffer: {num_batches} batches ({buffer_size} rows)")
        buffer_slices = iter_slices(start_offset, stop_offset, buffer_size)
        if array.schema.sparse:
            return sparse_batch_cls(buffer_slices, array, attrs)
        else:
            return dense_batch_cls(buffer_slices, array, attrs)

    x_batch = batch_factory("x", x_array, x_attrs)
    y_batch = batch_factory("y", y_array, y_attrs)
    with futures.ThreadPoolExecutor(max_workers=2) as executor:
        for batch_slice in iter_slices(start_offset, stop_offset, batch_size):
            done_futures, _ = futures.wait(
                (
                    executor.submit(x_batch.ensure_buffer, batch_slice),
                    executor.submit(y_batch.ensure_buffer, batch_slice),
                )
            )
            min_read_buffer_size = min(f.result() for f in done_futures)
            if batch_shuffle and min_read_buffer_size > 0:
                row_idxs = np.arange(min_read_buffer_size)
                np.random.shuffle(row_idxs)
                x_batch.shuffle_buffer(row_idxs)
                y_batch.shuffle_buffer(row_idxs)

            x_tensors = x_batch.iter_tensors(batch_slice)
            y_tensors = y_batch.iter_tensors(batch_slice)
            yield (*x_tensors, *y_tensors)


@dataclass(frozen=True)
class Shuffling:
    size: int
    x_buffer_slice: slice
    y_buffer_slice: slice


@dataclass(frozen=True)
class Batch:
    x_read_slice: Optional[slice]
    y_read_slice: Optional[slice]
    shuffling: Optional[Shuffling]
    x_buffer_slice: slice
    y_buffer_slice: slice


def iter_batches(
    batch_size: int,
    x_buffer_size: int,
    y_buffer_size: int,
    start_offset: int,
    stop_offset: int,
) -> Iterator[Batch]:
    """
    Generate `Batch` instances describing each batch.

    Each yielded `Batch` instance describes:
    - The slice to read from the x array into x buffer (if the current x buffer is consumed).
    - The slice to read from the y array into y buffer (if the current y buffer is consumed).
    - The shuffling to apply to x and y buffers (if there is new x and/or y buffer).
      - How many buffer rows to shuffle.
      - The slice of the x buffer to shuffle.
      - The slice of the y buffer to shuffle.
    - The batch slice to read from the x buffer.
    - The batch slice to read from the y buffer.

    :param batch_size: (Max) size of each batch.
    :param x_buffer_size: (Max) size of the x buffer.
    :param y_buffer_size: (Max) size of the y buffer.
    :param start_offset: Start row offset.
    :param stop_offset: Stop row offset.
    """
    assert (
        x_buffer_size % batch_size == 0
    ), "x_buffer_size must be a multiple of batch_size"
    assert (
        y_buffer_size % batch_size == 0
    ), "y_buffer_size must be a multiple of batch_size"

    x_buf_offset = x_buffer_size
    y_buf_offset = y_buffer_size
    x_read_slices = iter_slices(start_offset, stop_offset, x_buffer_size)
    y_read_slices = iter_slices(start_offset, stop_offset, y_buffer_size)
    for batch_slice in iter_slices(start_offset, stop_offset, batch_size):
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

        if x_read_slice or y_read_slice:
            shuffling_size = min(x_read_size - x_buf_offset, y_read_size - y_buf_offset)
            shuffling = Shuffling(
                shuffling_size,
                slice(x_buf_offset, x_buf_offset + shuffling_size),
                slice(y_buf_offset, y_buf_offset + shuffling_size),
            )
        else:
            shuffling = None

        batch_slice_size = batch_slice.stop - batch_slice.start
        x_next_buf_offset = x_buf_offset + batch_slice_size
        y_next_buf_offset = y_buf_offset + batch_slice_size
        yield Batch(
            x_read_slice,
            y_read_slice,
            shuffling,
            slice(x_buf_offset, x_next_buf_offset),
            slice(y_buf_offset, y_next_buf_offset),
        )
        x_buf_offset = x_next_buf_offset
        y_buf_offset = y_next_buf_offset


def iter_slices(start: int, stop: int, step: int) -> Iterator[slice]:
    offsets = range(start, stop, step)
    yield from map(slice, offsets, offsets[1:])
    yield slice(offsets[-1], stop)


def slice_subsumes(slice1: slice, slice2: slice) -> bool:
    """Check if slice1 subsumes slice2.

    Both slices must have integer start/stop and step=None.
    """
    assert slice1.step is slice2.step is None, (slice1, slice2)
    return bool(slice1.start <= slice2.start and slice1.stop >= slice2.stop)


def get_attr_names(schema: tiledb.ArraySchema) -> Sequence[str]:
    return tuple(schema.attr(idx).name for idx in range(schema.nattr))


def get_dim_names(schema: tiledb.ArraySchema) -> Sequence[str]:
    return tuple(schema.domain.dim(idx).name for idx in range(schema.ndim))


def estimate_row_bytes(
    array: tiledb.Array,
    attrs: Sequence[str] = (),
    start_offset: int = 0,
    stop_offset: int = 0,
) -> int:
    """
    Estimate the size in bytes of a TileDB array row.

    A "row" is a slice with the first dimension fixed.
    - For dense arrays, each row consists of a fixed number of cells. The size of each
      cell depends on the given attributes (or all array attributes by default).
    - For sparse arrays, each row consists of a variable number of non-empty cells. The
      size of each non-empty cell depends on all dimension coordinates as well as the
      given attributes (or all array attributes by default).
    """
    schema = array.schema
    if not attrs:
        attrs = get_attr_names(schema)

    if not schema.sparse:
        # for dense arrays the size of each row is fixed and can be computed exactly
        row_cells = np.prod(schema.shape[1:])
        cell_bytes = sum(schema.attr(attr).dtype.itemsize for attr in attrs)
        est_row_bytes = row_cells * cell_bytes
    else:
        # for sparse arrays the size of each row is variable and can only be estimated
        if not stop_offset:
            stop_offset = schema.shape[0]
        query = array.query(return_incomplete=True)
        # .multi_index[] is inclusive, so we need to subtract 1 to stop_offset
        indexer = query.multi_index[start_offset : stop_offset - 1]
        est_rs = indexer.estimated_result_sizes()
        dims = get_dim_names(schema)
        est_total_bytes = sum(est_rs[key].data_bytes for key in (*dims, *attrs))
        est_row_bytes = est_total_bytes / (stop_offset - start_offset)
    return int(est_row_bytes)


def get_num_batches(
    batch_size: int,
    buffer_bytes: int,
    array: tiledb.Array,
    attrs: Sequence[str] = (),
    start_offset: int = 0,
    stop_offset: int = 0,
) -> int:
    """
    Determine the number of batches to read from the given array.

    The number of buffer rows is determined by dividing buffer_bytes with the (estimated)
    row size. This number is then divided with batch_size to give the number of batches.
    """
    if not stop_offset:
        stop_offset = array.shape[0]
    est_row_bytes = estimate_row_bytes(array, attrs, start_offset, stop_offset)
    num_batches = max(1, buffer_bytes / est_row_bytes / batch_size)
    # upper num_batches bound is ceil(num_rows / batch_size)
    num_batches = min(num_batches, np.ceil((stop_offset - start_offset) / batch_size))
    return int(num_batches)
