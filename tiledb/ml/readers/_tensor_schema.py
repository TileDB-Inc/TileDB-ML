from __future__ import annotations

from abc import ABC, abstractmethod
from math import ceil
from operator import itemgetter
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import sparse

import tiledb

from ._ranges import InclusiveRange

Tensor = TypeVar("Tensor")


class TensorSchema(ABC):
    """
    A class to encapsulate the information needed for mapping a TileDB array to tensors.
    """

    def __init__(
        self,
        array: tiledb.Array,
        key_dim: Union[int, str] = 0,
        fields: Sequence[str] = (),
        transform: Optional[Callable[[Tensor], Tensor]] = None,
    ):
        """Create a TensorSchema.

        :param array: TileDB array to read from.
        :param key_dim: Name or index of the key dimension. Defaults to the first dimension.
        :param fields: Attribute and/or dimension names of the array to read. Defaults to
            all attributes.
        :param transform: Function to transform tensors.
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

        key_dim_min, key_dim_max = ned[0]
        self._key_range = InclusiveRange.factory(range(key_dim_min, key_dim_max + 1))
        self._array = array
        self._key_dim_index = key_dim_index
        self._ned = tuple(ned)
        self._all_dims = tuple(all_dims)
        self._fields = tuple(fields)
        self._query_kwargs = {"attrs": tuple(attrs), "dims": tuple(dims)}
        self._transform = transform

    @property
    def fields(self) -> Sequence[str]:
        """Names of attributes and dimensions to read."""
        return self._fields

    @property
    def field_dtypes(self) -> Sequence[np.dtype]:
        """Dtypes of attributes and dimensions to read."""
        return tuple(map(self._array.schema.attr_or_dim_dtype, self._fields))

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
    def key_range(self) -> InclusiveRange[Any, int]:
        """Inclusive range of the key dimension"""
        return self._key_range

    @property
    def query(self) -> KeyDimQuery:
        """A sliceable object for querying the TileDB array along the key dimension"""
        return KeyDimQuery(self._array, self._key_dim_index, **self._query_kwargs)

    @property
    @abstractmethod
    def sparse(self) -> bool:
        """Whether the underlying TileDB array is sparse"""

    @property
    @abstractmethod
    def max_partition_weight(self) -> int:
        """
        Determine the maximum partition that can be read without incomplete retries.

        What constitutes weight of a partition depends on the array type:
        - For dense arrays, it is the number of unique keys (= number of "rows").
          It depends on the `sm.memory_budget` config parameter.
        - For sparse arrays, it is the number of non-empty cells.
          It depends on the `py.init_buffer_bytes` config parameter.
        """

    @abstractmethod
    def iter_tensors(
        self, key_ranges: Iterable[InclusiveRange[Any, int]]
    ) -> Union[Iterable[Tensor], Iterable[Sequence[Tensor]]]:
        """
        Generate batches of dense or sparse tensors.

        Each yielded batch is either:
        - a sequence of N tensors if N > 1, where `N == len(self.fields)`, or
        - a single tensor if N == 1.
        where each tensor has shape `(len(key_range), *self.shape[1:])`.

        :param key_ranges: Inclusive ranges along the key dimension.
        """


class DenseTensorSchema(TensorSchema):
    sparse = False

    def iter_tensors(
        self, key_ranges: Iterable[InclusiveRange[Any, int]]
    ) -> Union[Iterable[np.ndarray], Iterable[Sequence[np.ndarray]]]:
        """
        Generate batches of Numpy arrays.

        If `key_dim_index > 0`, the generated arrays will ve reshaped so that the key_dim
        axes is first. For example, if the TileDB array `a` has shape (5, 12, 20) and
        `key_dim_index == 1`, then `a[:, 4:8, :]` returns arrays of shape (5, 4, 20) but
        this method yields arrays of shape (4, 5, 20).
        """
        query = self.query
        get_data = itemgetter(*self._fields)
        key_dim_index = self._key_dim_index
        for key_range in key_ranges:
            field_arrays = query[key_range.min : key_range.max]
            if key_dim_index > 0:
                # Move key_dim_index axes first
                for field, array in field_arrays.items():
                    field_arrays[field] = np.moveaxis(array, key_dim_index, 0)
            yield get_data(field_arrays)

    @property
    def max_partition_weight(self) -> int:
        array = self._array
        memory_budget = int(array._ctx_().config()["sm.memory_budget"])

        # The memory budget should be large enough to read the cells of the largest field
        bytes_per_cell = max(dtype.itemsize for dtype in self.field_dtypes)

        # We want to be reading tiles following the tile extents along each dimension.
        # The number of cells for each such tile is the product of all tile extents.
        dim_tiles = [array.dim(dim).tile for dim in self._all_dims]
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


class SparseTensorSchema(TensorSchema):
    sparse = True

    def __init__(
        self,
        array: tiledb.Array,
        key_dim: Union[int, str] = 0,
        fields: Sequence[str] = (),
        transform: Optional[Callable[[Tensor], Tensor]] = None,
    ):
        super().__init__(array, key_dim, fields, transform)
        self._query_kwargs["dims"] = self._all_dims

    def iter_tensors(
        self, key_ranges: Iterable[InclusiveRange[Any, int]]
    ) -> Union[Iterable[Tensor], Iterable[Sequence[Tensor]]]:
        shape = list(self.shape)
        query = self.query
        get_data = itemgetter(*self._fields)
        single_field = len(self._fields) == 1
        all_dims = self._all_dims
        dim_starts = tuple(map(itemgetter(0), self._ned))
        transform = self._transform or (lambda x: x)
        for key_range in key_ranges:
            # Set the shape of the key dimension equal to the current key range length
            shape[0] = len(key_range)
            field_arrays = query[key_range.min : key_range.max]
            data = get_data(field_arrays)

            # Convert coordinates from the original domain to zero-based
            # For the key (i.e. first) dimension, ignore the keys before the current range
            coords = tuple(field_arrays[dim] for dim in all_dims)
            for i, (coord, dim_start) in enumerate(zip(coords, dim_starts)):
                coord -= dim_start if i > 0 else key_range.min

            # yield either a single tensor or a sequence of tensors, one for each field
            if single_field:
                yield transform(sparse.COO(coords, data, shape))
            else:
                yield tuple(transform(sparse.COO(coords, d, shape)) for d in data)

    @property
    def max_partition_weight(self) -> int:
        array = self._array
        try:
            init_buffer_bytes = int(array._ctx_().config()["py.init_buffer_bytes"])
        except KeyError:
            init_buffer_bytes = 10 * 1024**2

        # the size of each row is variable and can only be estimated
        query = array.query(return_incomplete=True, **self._query_kwargs)
        res_sizes = query.multi_index[:].estimated_result_sizes()

        max_buffer_bytes = max(res_size.data_bytes for res_size in res_sizes.values())
        max_bytes_per_row = ceil(max_buffer_bytes / len(self._key_range))

        return max(1, init_buffer_bytes // max_bytes_per_row)


class KeyDimQuery:
    def __init__(self, array: tiledb.Array, key_dim_index: int, **kwargs: Any):
        self._multi_index = array.query(**kwargs).multi_index
        self._leading_dim_slices = (slice(None),) * key_dim_index

    def __getitem__(self, key_dim_slice: slice) -> Dict[str, np.ndarray]:
        """Query the TileDB array by `dim_key=key_dim_slice`."""
        return cast(
            Dict[str, np.ndarray],
            self._multi_index[(*self._leading_dim_slices, key_dim_slice)],
        )
