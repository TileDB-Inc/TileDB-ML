from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
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
from .types import ArrayParams

Tensor = TypeVar("Tensor")


@dataclass(frozen=True)  # type: ignore
class TensorSchema(ABC):
    """
    A class to encapsulate the information needed for mapping a TileDB array to tensors.
    """

    _array: tiledb.Array
    _key_dim_index: int
    _fields: Sequence[str]
    _all_dims: Sequence[str]
    _ned: Sequence[Tuple[Any, Any]]
    _query_kwargs: Dict[str, Any]
    _transform: Optional[Callable[[Tensor], Tensor]]

    @classmethod
    def from_array_params(
        cls,
        array_params: ArrayParams,
        transform: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> TensorSchema:
        kwargs = {"_" + k: v for k, v in array_params._tensor_schema_kwargs.items()}
        return cls(_transform=transform, **kwargs)

    @property
    def fields(self) -> Sequence[str]:
        """Names of attributes and dimensions to read."""
        return self._fields

    @property
    def field_dtypes(self) -> Sequence[np.dtype]:
        """Dtypes of attributes and dimensions to read."""
        return tuple(map(self._array.schema.attr_or_dim_dtype, self._fields))

    @property
    def shape(self) -> Sequence[int]:
        """Shape of the array, with the key dimension moved first.

        **Note**: For sparse arrays, the returned shape reflects the non-empty domain of
        the array, not the full array shape.

        :raises ValueError: If the array does not have integer domain.
        """
        shape = [len(self.key_range)]
        for start, stop in self._ned[1:]:
            if isinstance(start, int) and isinstance(stop, int):
                shape.append(stop - start + 1)
            else:
                raise ValueError("Shape not defined for non-integer domain")
        return shape

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
    def key_range(self) -> InclusiveRange[Any, int]:
        """Inclusive range of the key dimension.

        The values of the range are all the distinct values of the key dimension (keys).
        The weight of each key is:
        - for dense arrays: 1
        - for sparse arrays: The number of non-empty cells for this key
        """

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

    @property
    def key_range(self) -> InclusiveRange[int, int]:
        key_dim_min, key_dim_max = self._ned[0]
        return InclusiveRange.factory(range(key_dim_min, key_dim_max + 1))

    def iter_tensors(
        self, key_ranges: Iterable[InclusiveRange[int, int]]
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
        memory_budget = int(self._array._ctx_().config()["sm.memory_budget"])

        # The memory budget should be large enough to read the cells of the largest field
        bytes_per_cell = max(dtype.itemsize for dtype in self.field_dtypes)

        # We want to be reading tiles following the tile extents along each dimension.
        # The number of cells for each such tile is the product of all tile extents.
        dim_tiles = [self._array.dim(dim).tile for dim in self._all_dims]
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

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._query_kwargs["dims"] = self._all_dims
        key_counter: Counter[Any] = Counter()
        key_dim = self._all_dims[0]
        query = self._array.query(dims=(key_dim,), attrs=(), return_incomplete=True)
        for result in query.multi_index[:]:
            key_counter.update(result[key_dim])
        self._key_range = InclusiveRange.factory(key_counter)

    @property
    def key_range(self) -> InclusiveRange[Any, int]:
        return self._key_range

    def iter_tensors(
        self, key_ranges: Iterable[InclusiveRange[Any, int]]
    ) -> Union[Iterable[Tensor], Iterable[Sequence[Tensor]]]:
        shape = list(self.shape)
        query = self.query
        get_data = itemgetter(*self._fields)
        single_field = len(self._fields) == 1
        key_dim, *non_key_dims = self._all_dims
        non_key_dim_starts = tuple(map(itemgetter(0), self._ned[1:]))
        transform = self._transform or (lambda x: x)
        for key_range in key_ranges:
            # Set the shape of the key dimension equal to the current key range length
            shape[0] = len(key_range)
            field_arrays = query[key_range.min : key_range.max]
            data = get_data(field_arrays)

            # Convert coordinates from the original domain to zero-based
            # For the key (i.e. first) dimension get the indices of the keys
            coords = [key_range.indices(field_arrays.pop(key_dim))]
            # For every non-key dimension, subtract the minimum value of the dimension
            # TODO: update this for non-integer non-key dimensions
            coords.extend(
                field_arrays.pop(dim) - dim_start
                for dim, dim_start in zip(non_key_dims, non_key_dim_starts)
            )

            # yield either a single tensor or a sequence of tensors, one for each field
            if single_field:
                yield transform(sparse.COO(coords, data, shape))
            else:
                yield tuple(transform(sparse.COO(coords, d, shape)) for d in data)

    @property
    def max_partition_weight(self) -> int:
        try:
            memory_budget = int(self._array._ctx_().config()["py.init_buffer_bytes"])
        except KeyError:
            memory_budget = 10 * 1024**2

        # The memory budget should be large enough to read the cells of the largest field
        bytes_per_cell = max(
            self._array.schema.attr_or_dim_dtype(field).itemsize
            for field in self._query_kwargs["dims"] + self._query_kwargs["attrs"]
        )
        return max(1, memory_budget // int(bytes_per_cell))


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
