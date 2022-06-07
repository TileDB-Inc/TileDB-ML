from __future__ import annotations

from math import ceil
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple, Union, cast

import numpy as np

import tiledb

from ._ranges import InclusiveRange, IntRange


def iter_slices(start: int, stop: int, step: int) -> Iterator[slice]:
    starts = range(start, stop + 1, step)
    stops = range(start + step - 1, stop, step)
    yield from map(slice, starts, stops)
    yield slice(starts[-1], stop)


class TensorSchema:
    """
    A class to encapsulate the information needed for mapping a TileDB array to tensors.
    """

    def __init__(
        self,
        array: tiledb.Array,
        key_dim: Union[int, str] = 0,
        fields: Sequence[str] = (),
    ):
        """
        :param array: TileDB array to read from.
        :param key_dim: Name or index of the key dimension. Defaults to the first dimension.
        :param fields: Attribute and/or dimension names of the array to read. Defaults to
            all attributes.
        """
        if not np.issubdtype(array.dim(key_dim).dtype, np.integer):
            raise ValueError(f"Key dimension {key_dim} must have integer domain")

        all_attrs = [array.attr(i).name for i in range(array.nattr)]
        all_dims = [array.dim(i).name for i in range(array.ndim)]

        dims = []
        if fields:
            attrs = []
            for field in fields:
                if field in all_attrs:
                    attrs.append(field)
                elif field in all_dims:
                    dims.append(field)
                else:
                    raise ValueError(f"Unknown attribute or dimension '{field}'")
        else:
            fields = attrs = all_attrs

        ned = list(array.nonempty_domain())
        key_dim_index = key_dim if isinstance(key_dim, int) else all_dims.index(key_dim)
        if key_dim_index > 0:
            # Swap key dimension to first position
            all_dims[0], all_dims[key_dim_index] = all_dims[key_dim_index], all_dims[0]
            ned[0], ned[key_dim_index] = ned[key_dim_index], ned[0]

        self._array = array
        self._key_range = IntRange(*ned[0])
        self._ned: Sequence[Tuple[Any, Any]] = tuple(ned)
        self._dims = tuple(all_dims)
        self._fields = tuple(fields)
        self._leading_dim_slices = (slice(None),) * key_dim_index
        self._query_kwargs = {
            "attrs": tuple(attrs),
            "dims": tuple(all_dims if array.schema.sparse else dims),
        }

    @property
    def fields(self) -> Sequence[str]:
        """Names of attributes and dimensions to read."""
        return self._fields

    @property
    def field_dtypes(self) -> Sequence[np.dtype]:
        """Dtypes of attributes and dimensions to read."""
        get_dim, get_attr = self._array.dim, self._array.attr
        return tuple(
            (get_dim if field in self._dims else get_attr)(field).dtype
            for field in self._fields
        )

    @property
    def dims(self) -> Sequence[str]:
        """All dimension names of the array, with the key dimension moved first."""
        return self._dims

    @property
    def nonempty_domain(self) -> Sequence[Tuple[Any, Any]]:
        """Non-empty domain of the array, with the key dimension moved first."""
        return self._ned

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the array, with the key dimension moved first.

        **Note**: For sparse arrays, the returned shape reflects the non-empty domain of
        the array, not the full array shape.

        :raises ValueError: If the array does not have integer domain.
        """
        starts, stops = zip(*self._ned)
        if all(isinstance(i, int) for i in starts + stops):
            return tuple(stop - start + 1 for start, stop in self._ned)
        raise ValueError("Shape not defined for non-integer domain")

    @property
    def key_dim_index(self) -> int:
        """Index of the key dimension in the original TileDB schema."""
        return len(self._leading_dim_slices)

    @property
    def key_range(self) -> InclusiveRange[int, int]:
        """Inclusive range of the key dimension"""
        return self._key_range

    def __getitem__(self, key_dim_slice: slice) -> Dict[str, np.ndarray]:
        """Query the TileDB array by `dim_key=key_dim_slice`."""
        self._query: tiledb.libtiledb.Query
        try:
            query = self._query
        except AttributeError:
            self._query = query = self._array.query(**self._query_kwargs)
        return cast(
            Dict[str, np.ndarray],
            query.multi_index[(*self._leading_dim_slices, key_dim_slice)],
        )

    def partition_key_dim(
        self, memory_budget: Optional[int], start_key: int, stop_key: int
    ) -> Iterator[slice]:
        """
        Partition the keys between start and stop in slices that can fit in the given
        memory budget without incomplete retries.

        :param memory_budget: The maximum amount of memory to use. This is bounded by the
            `sm.memory_budget` config parameter for dense arrays and `py.init_buffer_bytes`
            (or 10 MB if unset) for sparse arrays. These bounds are also used as the
            default memory budget.
        :param start_key: The minimum value of the key dimension to partition.
        :param stop_key: The maximum value of the key dimension to partition.
        """
        if self._array.schema.sparse:
            buffer_size = self._get_max_buffer_size_sparse(memory_budget)
        else:
            buffer_size = self._get_max_buffer_size_dense(memory_budget)
        return iter_slices(start_key, stop_key, buffer_size)

    def _get_max_buffer_size_sparse(self, memory_budget: Optional[int] = None) -> int:
        array = self._array
        try:
            init_buffer_bytes = int(array._ctx_().config()["py.init_buffer_bytes"])
        except KeyError:
            init_buffer_bytes = 10 * 1024**2
        if memory_budget is None or memory_budget > init_buffer_bytes:
            memory_budget = init_buffer_bytes

        # the size of each row is variable and can only be estimated
        query = array.query(return_incomplete=True, **self._query_kwargs)
        res_sizes = query.multi_index[:].estimated_result_sizes()

        max_buffer_bytes = max(res_size.data_bytes for res_size in res_sizes.values())
        max_bytes_per_row = ceil(max_buffer_bytes / len(self.key_range))

        return max(1, memory_budget // max_bytes_per_row)

    def _get_max_buffer_size_dense(self, memory_budget: Optional[int] = None) -> int:
        array = self._array
        config_memory_budget = int(array._ctx_().config()["sm.memory_budget"])
        if memory_budget is None or memory_budget > config_memory_budget:
            memory_budget = config_memory_budget

        # The memory budget should be large enough to read the cells of the largest field
        bytes_per_cell = max(dtype.itemsize for dtype in self.field_dtypes)

        # We want to be reading tiles following the tile extents along each dimension.
        # The number of cells for each such tile is the product of all tile extents.
        dim_tiles = [array.dim(dim).tile for dim in self.dims]
        cells_per_tile = np.prod(dim_tiles)

        # Each slice consists of `rows_per_slice` rows along the key dimension
        rows_per_slice = dim_tiles[0]

        # Reading a slice of `rows_per_slice` rows requires reading a number of tiles that
        # depends on the size and tile extent of each dimension after the first one.
        assert len(self.shape) == len(dim_tiles)
        tiles_per_slice = np.prod(
            [ceil(size / tile) for size, tile in zip(self.shape[1:], dim_tiles[1:])]
        )

        # Compute the size in bytes of each slice of `rows_per_slice` rows
        bytes_per_slice = bytes_per_cell * cells_per_tile * tiles_per_slice

        # Compute the number of slices that fit within the memory budget
        num_slices = memory_budget // bytes_per_slice

        # Compute the total number of rows to slice
        return max(1, int(rows_per_slice * num_slices))
