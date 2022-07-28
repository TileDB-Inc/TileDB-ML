from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from math import ceil
from operator import itemgetter
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import scipy.sparse
import sparse
import wrapt

import tiledb

from ._ranges import InclusiveRange


class TensorKind(enum.Enum):
    """Kind of tensor."""

    DENSE = enum.auto()
    SPARSE_COO = enum.auto()
    SPARSE_CSR = enum.auto()
    RAGGED = enum.auto()


Tensor = TypeVar("Tensor")


@dataclass(frozen=True)  # type: ignore  # https://github.com/python/mypy/issues/5374
class TensorSchema(ABC, Generic[Tensor]):
    """
    A class to encapsulate the information needed for mapping a TileDB array to tensors.
    """

    kind: TensorKind
    _array: tiledb.Array
    _key_dim_index: int
    _fields: Sequence[str]
    _all_dims: Sequence[str]
    _ned: Sequence[Tuple[Any, Any]]
    _query_kwargs: Dict[str, Any]

    @property
    def num_fields(self) -> int:
        """Number of attributes and dimensions to read."""
        return len(self._fields)

    @property
    def field_dtypes(self) -> Sequence[np.dtype]:
        """Dtypes of attributes and dimensions to read."""
        return tuple(map(self._array.schema.attr_or_dim_dtype, self._fields))

    @property
    def shape(self) -> Sequence[Optional[int]]:
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
        return tuple(shape)

    @property
    def query(self) -> KeyDimQuery:
        """A sliceable object for querying the TileDB array along the key dimension"""
        return KeyDimQuery(self._array, self._key_dim_index, **self._query_kwargs)

    @property
    def key_dim(self) -> str:
        """Key dimension of the array."""
        return self._all_dims[0]

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
        Generate batches of tensors.

        Each yielded batch is either:
        - a sequence of N tensors if N > 1, where `N == self.num_fields`, or
        - a single tensor if N == 1.
        where each tensor has shape `(len(key_range), *self.shape[1:])`.

        :param key_ranges: Inclusive ranges along the key dimension.
        """


MappedTensor = TypeVar("MappedTensor")


class MappedTensorSchema(wrapt.ObjectProxy, Generic[Tensor, MappedTensor]):
    """
    Proxy class that wraps a TensorSchema and applies a mapping function to each tensor
    yielded by `iter_tensors`.
    """

    def __init__(
        self,
        wrapped: TensorSchema[Tensor],
        map_tensor: Callable[[Tensor], MappedTensor],
    ):
        super().__init__(wrapped)

        def map_tensors(tensors: Sequence[Tensor]) -> Sequence[MappedTensor]:
            return tuple(map(map_tensor, tensors))

        self._self_map_tensor = map_tensor
        self._self_map_tensors = map_tensors

    def iter_tensors(
        self, key_ranges: Iterable[InclusiveRange[Any, int]]
    ) -> Union[Iterable[MappedTensor], Iterable[Sequence[MappedTensor]]]:
        wrapped_iter_tensors = self.__wrapped__.iter_tensors(key_ranges)
        if self.num_fields == 1:
            return map(self._self_map_tensor, wrapped_iter_tensors)
        else:
            return map(self._self_map_tensors, wrapped_iter_tensors)


class DenseTensorSchema(TensorSchema[np.ndarray]):
    """
    TensorSchema for reading dense TileDB arrays as (dense) Numpy arrays.
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        if self._array.schema.sparse:
            raise NotImplementedError(
                "Mapping a sparse TileDB array to dense tensors is not implemented"
            )

    @property
    def key_range(self) -> InclusiveRange[int, int]:
        key_dim_min, key_dim_max = self._ned[0]
        return InclusiveRange.factory(range(key_dim_min, key_dim_max + 1))

    def iter_tensors(
        self, key_ranges: Iterable[InclusiveRange[int, int]]
    ) -> Union[Iterable[np.ndarray], Iterable[Sequence[np.ndarray]]]:
        """
        Generate batches of Numpy arrays.

        If `key_dim_index > 0`, the generated arrays will be reshaped so that the key_dim
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


class BaseSparseTensorSchema(TensorSchema[Tensor]):
    """Abstract base class for reading sparse TileDB arrays."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        if not self._array.schema.sparse:
            raise NotImplementedError(
                "Mapping a dense TileDB array to sparse tensors is not implemented"
            )

    @property
    def key_range(self) -> InclusiveRange[Any, int]:
        self._key_range: InclusiveRange[Any, int]
        try:
            return self._key_range
        except AttributeError:
            key_counter: Counter[Any] = Counter()
            key_dim = self.key_dim
            query = self._array.query(dims=(key_dim,), attrs=(), return_incomplete=True)
            for result in query.multi_index[:]:
                key_counter.update(result[key_dim])
            self._key_range = InclusiveRange.factory(key_counter)
            return self._key_range

    @property
    def max_partition_weight(self) -> int:
        try:
            memory_budget = int(self._array._ctx_().config()["py.init_buffer_bytes"])
        except KeyError:
            memory_budget = 10 * 1024**2

        # Determine the bytes per (non-empty) cell for each field.
        # - For fixed size fields, this is just the `dtype.itemsize` of the field.
        # - For variable size fields, the best we can do is to estimate the average bytes
        #   size. We also need to take into account the (fixed) offset buffer size per
        #   cell (=8 bytes).
        offset_itemsize = np.dtype(np.uint64).itemsize
        attr_or_dim_dtype = self._array.schema.attr_or_dim_dtype
        query = self._array.query(return_incomplete=True, **self._query_kwargs)
        est_sizes = query.multi_index[:].estimated_result_sizes()
        bytes_per_cell = []
        for field, est_result_size in est_sizes.items():
            if est_result_size.offsets_bytes == 0:
                bytes_per_cell.append(attr_or_dim_dtype(field).itemsize)
            else:
                num_cells = est_result_size.offsets_bytes / offset_itemsize
                avg_itemsize = est_result_size.data_bytes / num_cells
                bytes_per_cell.append(max(avg_itemsize, offset_itemsize))

        # Finally, the number of cells that can fit in the memory_budget depends on the
        # maximum bytes_per_cell
        return max(1, memory_budget // ceil(max(bytes_per_cell)))


@dataclass(frozen=True)
class SparseData:
    coords: np.ndarray
    data: np.ndarray
    shape: Sequence[int]

    def to_sparse_array(self) -> Union[scipy.sparse.csr_matrix, sparse.COO]:
        if len(self.shape) == 2:
            return scipy.sparse.csr_matrix((self.data, self.coords), self.shape)
        else:
            return sparse.COO(self.coords, self.data, self.shape)


class SparseTensorSchema(BaseSparseTensorSchema[SparseData]):
    """
    TensorSchema for reading sparse TileDB arrays as SparseData instances.
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._query_kwargs["dims"] = self._all_dims

    def iter_tensors(
        self, key_ranges: Iterable[InclusiveRange[Any, int]]
    ) -> Union[Iterable[SparseData], Iterable[Sequence[SparseData]]]:
        shape = list(cast(Sequence[int], self.shape))
        query = self.query
        get_data = itemgetter(*self._fields)
        single_field = len(self._fields) == 1
        key_dim, *non_key_dims = self._all_dims
        non_key_dim_starts = tuple(map(itemgetter(0), self._ned[1:]))
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
            coords = np.array(coords)

            # yield either a single SparseData or one SparseData per field
            if single_field:
                yield SparseData(coords, data, shape)
            else:
                yield tuple(SparseData(coords, d, shape) for d in data)


RaggedArray = Sequence[np.ndarray]


class RaggedTensorSchema(BaseSparseTensorSchema[RaggedArray]):
    """TensorSchema for reading sparse TileDB arrays as ragged Numpy arrays.

    Each ragged array is represented as a sequence of 1D Numpy arrays of the same dtype
    but (in general) different size. Each item of the ragged array contains the values of
    a given field for a given key (i.e. value along the key dimension). The items of the
    ragged array are ordered by the key dimension but the values within each item are not
    (meaningfully) ordered.

    Example:
    - (key, field1, field2) cells:
        {('a', 3, 2.8), ('b', 1, 0.2), ('a', 7, 3.2), ('c', 5, 6.1), ('c', 2, 0.5), ('a', 2, 1.9)}

    - Ragged arrays:
        - key: [np.array(['a', 'a', 'a']), np.array(['b']), np.array(['c', 'c'])]
        - field1: [np.array([3, 7, 2]), np.array([1]), np.array([5, 2])]
        - field2: [np.array([2.8, 3.2, 1.9]), np.array([0.2]), np.array([6.1, 0.5])]
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        if self.key_dim not in self._query_kwargs["dims"]:
            self._query_kwargs["dims"] += (self.key_dim,)

    @property
    def shape(self) -> Sequence[Optional[int]]:
        return len(self.key_range), None

    def iter_tensors(
        self, key_ranges: Iterable[InclusiveRange[int, int]]
    ) -> Union[Iterable[RaggedArray], Iterable[Sequence[RaggedArray]]]:
        query = self.query
        get_data = itemgetter(*self._fields)
        key_dim = self.key_dim
        for key_range in key_ranges:
            field_arrays = query[key_range.min : key_range.max]
            # Sort the key dimension values and find the indices where the value changes
            sort_idx = np.argsort(field_arrays[key_dim], kind="stable")
            split_idx = argdiff(field_arrays[key_dim][sort_idx])
            # apply the same sorting and splitting to all the field arrays
            for name, values in field_arrays.items():
                field_arrays[name] = np.split(values[sort_idx], split_idx)
            yield get_data(field_arrays)


def argdiff(a: np.ndarray) -> np.ndarray:
    """Return the indices `i` of the array `a` so that `a[i] != a[i-1]`."""
    idx = np.nonzero(a[:-1] != a[1:])[0]
    # the minimum diff index (if a[0] != a[1]) is 1, not 0
    idx += 1
    return idx


TensorSchemaFactories: Dict[TensorKind, Type[TensorSchema[Any]]] = {
    TensorKind.DENSE: DenseTensorSchema,
    TensorKind.RAGGED: RaggedTensorSchema,
    TensorKind.SPARSE_COO: SparseTensorSchema,
    TensorKind.SPARSE_CSR: SparseTensorSchema,
}


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
