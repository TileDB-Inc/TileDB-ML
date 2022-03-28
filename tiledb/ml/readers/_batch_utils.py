from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent import futures
from dataclasses import dataclass
from math import ceil
from typing import Generic, Iterator, Optional, Sequence, Tuple, Type, TypeVar, Union

import numpy as np
import scipy.sparse as sp

import tiledb

Tensor = TypeVar("Tensor")


class TileDBTensorGenerator(ABC, Generic[Tensor]):
    """Base class for generating tensors read from a TileDB array."""

    def __init__(
        self,
        array: tiledb.Array,
        attrs: Sequence[str] = (),
    ) -> None:
        """
        :param array: TileDB array to read from.
        :param attrs: Attribute names of array to read; defaults to all array attributes.
        """
        self._query = array.query(attrs=attrs or None)

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
    def __init__(
        self,
        array: tiledb.Array,
        attrs: Sequence[str] = (),
    ) -> None:
        schema = array.schema
        if schema.ndim != 2:
            raise NotImplementedError("Only 2D sparse tensors are currently supported")
        if not attrs:
            attrs = get_attr_names(schema)
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
    buffer_bytes: Optional[int] = None,
    x_attrs: Sequence[str] = (),
    y_attrs: Sequence[str] = (),
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
    :param buffer_bytes: Maximum size (in bytes) of memory to allocate for reading from
        each array (default=`tiledb.default_ctx().config()["sm.memory_budget"]`).
    :param x_attrs: Attribute names of x_array; defaults to all x_array attributes.
    :param y_attrs: Attribute names of y_array; defaults to all y_array attributes
    :param start_offset: Start row offset; defaults to 0.
    :param stop_offset: Stop row offset; defaults to number of rows.
    :param dense_tensor_generator_cls: Dense tensor generator type.
    :param sparse_tensor_generator_cls: Sparse tensor generator type.
    """
    if not stop_offset:
        stop_offset = x_array.shape[0]

    def get_buffer_size_generator(
        array: tiledb.Array, attrs: Sequence[str]
    ) -> Union[
        Tuple[int, TileDBTensorGenerator[DenseTensor]],
        Tuple[int, TileDBTensorGenerator[SparseTensor]],
    ]:
        if array.schema.sparse:
            # TODO: implement get_max_buffer_size() for sparse arrays
            row_bytes = estimate_row_bytes(array, attrs, start_offset, stop_offset)
            buffer_size = (buffer_bytes or 100 * 1024**2) // row_bytes
        else:
            buffer_size = get_max_buffer_size(array.schema, attrs, buffer_bytes)
        # clip the buffer size between 1 and total number of rows
        buffer_size = max(1, min(buffer_size, stop_offset - start_offset))
        if array.schema.sparse:
            return buffer_size, sparse_tensor_generator_cls(array, attrs)
        else:
            return buffer_size, dense_tensor_generator_cls(array, attrs)

    x_buf_size, x_gen = get_buffer_size_generator(x_array, x_attrs)
    y_buf_size, y_gen = get_buffer_size_generator(y_array, y_attrs)
    with futures.ThreadPoolExecutor(max_workers=2) as executor:
        for batch in iter_batches(x_buf_size, y_buf_size, start_offset, stop_offset):
            if batch.x_read_slice and batch.y_read_slice:
                futures.wait(
                    (
                        executor.submit(x_gen.read_buffer, batch.x_read_slice),
                        executor.submit(y_gen.read_buffer, batch.y_read_slice),
                    )
                )
            elif batch.x_read_slice:
                x_gen.read_buffer(batch.x_read_slice)
            elif batch.y_read_slice:
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


def get_max_buffer_size(
    schema: tiledb.ArraySchema,
    attrs: Sequence[str] = (),
    memory_budget: Optional[int] = None,
) -> int:
    """
    Get the maximum number of "rows" that can be read from an array with the given schema
    without incurring incomplete reads.

    A "row" is a slice with the first dimension fixed.

    :param schema: The array schema.
    :param attrs: The attributes to read; defaults to all array attributes.
    :param memory_budget: The maximum amount of memory to use. This is bounded by
        `tiledb.default_ctx().config()["sm.memory_budget"]`, which is also used as the
        default memory_budget.
    """
    if schema.sparse:
        raise NotImplementedError(
            "get_max_buffer_size() is not implemented for sparse arrays"
        )

    config_memory_budget = int(tiledb.default_ctx().config()["sm.memory_budget"])
    if memory_budget is None or memory_budget > config_memory_budget:
        memory_budget = config_memory_budget

    # The memory budget should be large enough to read the cells of the largest attribute
    if not attrs:
        attrs = get_attr_names(schema)
    bytes_per_cell = max(schema.attr(attr).dtype.itemsize for attr in attrs)

    # We want to be reading tiles following the tile extents along each dimension.
    # The number of cells for each such tile is the product of all tile extents.
    dim_tiles = tuple(int(schema.domain.dim(idx).tile) for idx in range(schema.ndim))
    cells_per_tile = np.prod(dim_tiles)

    # Reading a slice of dim_tiles[0] rows requires reading a number of tiles that
    # depends on the size and tile extent of each dimension after the first one.
    assert len(schema.shape) == len(dim_tiles)
    tiles_per_slice = np.prod(
        tuple(
            ceil(dim_size / dim_tile)
            for dim_size, dim_tile in zip(schema.shape[1:], dim_tiles[1:])
        )
    )

    # Compute the size in bytes of each slice of dim_tiles[0] rows
    bytes_per_slice = int(bytes_per_cell * cells_per_tile * tiles_per_slice)

    # Compute the number of slices that fit within the memory budget
    num_slices = memory_budget // bytes_per_slice

    # Compute the total number of rows to slice
    return dim_tiles[0] * num_slices
