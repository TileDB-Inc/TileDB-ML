from __future__ import annotations

from math import ceil
from typing import Optional, Sequence, Tuple, Union

import numpy as np

import tiledb


class TensorSchema:
    """
    A class to encapsulate the information needed for mapping a TileDB array to tensors.
    """

    def __init__(
        self,
        array: tiledb.Array,
        key_dim: Union[int, str] = 0,
        attrs: Sequence[str] = (),
    ):
        """
        :param array: TileDB array to read from.
        :param key_dim: Name or index of the key dimension; defaults to the first dimension.
        :param attrs: Attribute names of array to read; defaults to all attributes.
        """
        get_dim = array.domain.dim
        if not np.issubdtype(get_dim(key_dim).dtype, np.integer):
            raise ValueError(f"Key dimension {key_dim} must have integer domain")

        all_attrs = [array.attr(i).name for i in range(array.nattr)]
        unknown_attrs = [attr for attr in attrs if attr not in all_attrs]
        if unknown_attrs:
            raise ValueError(f"Unknown attributes: {unknown_attrs}")

        ned = list(array.nonempty_domain())
        dims = [get_dim(i).name for i in range(array.ndim)]
        key_dim_index = dims.index(key_dim) if not isinstance(key_dim, int) else key_dim
        if key_dim_index > 0:
            # Swap key dimension to first position
            dims[0], dims[key_dim_index] = dims[key_dim_index], dims[0]
            ned[0], ned[key_dim_index] = ned[key_dim_index], ned[0]

        self._ned: Sequence[Tuple[int, int]] = tuple(ned)
        self._dims = tuple(dims)
        self._attrs = tuple(attrs or all_attrs)
        self._leading_dim_slices = (slice(None),) * key_dim_index

    @property
    def attrs(self) -> Sequence[str]:
        """The attribute names of the array to read."""
        return self._attrs

    @property
    def dims(self) -> Sequence[str]:
        """The dimension names of the array, with the key dimension moved first."""
        return self._dims

    @property
    def nonempty_domain(self) -> Sequence[Tuple[int, int]]:
        """The non-empty domain of the array, with the key dimension moved first."""
        return self._ned

    @property
    def shape(self) -> Tuple[int, ...]:
        """The shape of the array, with the key dimension moved first.

        **Note**: For sparse arrays, the returned shape reflects the non-empty domain of
        the array, not the full array shape.

        :raises ValueError: If the array does not have integer domain.
        """
        shape = tuple(stop - start + 1 for start, stop in self._ned)
        if all(isinstance(i, int) for i in shape):
            return shape
        raise ValueError("Shape not defined for non-integer domain")

    @property
    def key_dim_index(self) -> int:
        """The index of the key dimension in the original TileDB schema."""
        return len(self._leading_dim_slices)

    @property
    def num_keys(self) -> int:
        """The number of distinct values along the key dimension"""
        return self.stop_key - self.start_key

    @property
    def start_key(self) -> int:
        """The minimum value of the key dimension."""
        return self._ned[0][0]

    @property
    def stop_key(self) -> int:
        """The maximum value of the key dimension, plus 1."""
        return self._ned[0][1] + 1

    def __getitem__(self, key_dim_slice: slice) -> Tuple[slice, ...]:
        """Return the indexing tuple for querying the TileDB array by `dim_key=key_dim_slice`.

        For example, if `self.key_dim_index == 2`, then querying by the key dimension
        would be `array[:, :, key_dim_slice]`, which corresponds to the indexing tuple
        `(slice(None), slice(None), key_dim_slice)`.
        """
        return (*self._leading_dim_slices, key_dim_slice)

    def ensure_equal_keys(self, other: TensorSchema) -> None:
        """Ensure that the key dimension bounds of the of two schemas are equal.

        :raises ValueError: If the key dimension bounds are not equal.
        """
        if self._ned[0] != other._ned[0]:
            raise ValueError(
                f"X and Y arrays have different key domain: {self._ned[0]} != {other._ned[0]}"
            )


def get_buffer_size(
    array: tiledb.Array,
    schema: TensorSchema,
    memory_budget: Optional[int] = None,
) -> int:
    """Estimate the maximum number of "rows" than can fit in the given memory budget.

    A "row" is a slice along the `schema.key_dim_index` dimension where each cell consists
    of the `schema.attrs` attributes.

    :param array: TileDB array to read from.
    :param schema: TensorSchema for the array.
    :param memory_budget: The maximum amount of memory to use. This is bounded by the
        `sm.memory_budget` config parameter for dense arrays and `py.init_buffer_bytes`
        (or 10 MB if unset) for sparse arrays. These bounds are also used as the default
        memory budget.
    """
    if array.schema.sparse:
        buffer_size = _get_max_buffer_size_sparse(array, schema, memory_budget)
    else:
        buffer_size = _get_max_buffer_size_dense(array, schema, memory_budget)
    # clip the buffer size between 1 and total number of rows
    return max(1, min(buffer_size, schema.num_keys))


def _get_max_buffer_size_sparse(
    array: tiledb.Array,
    schema: TensorSchema,
    memory_budget: Optional[int] = None,
) -> int:
    assert array.schema.sparse
    try:
        init_buffer_bytes = int(array._ctx_().config()["py.init_buffer_bytes"])
    except KeyError:
        init_buffer_bytes = 10 * 1024**2
    if memory_budget is None or memory_budget > init_buffer_bytes:
        memory_budget = init_buffer_bytes

    # the size of each row is variable and can only be estimated
    query = array.query(attrs=schema.attrs, return_incomplete=True)
    res_sizes = query.multi_index[:].estimated_result_sizes()

    max_buffer_bytes = max(res_size.data_bytes for res_size in res_sizes.values())
    max_bytes_per_row = ceil(max_buffer_bytes / schema.num_keys)

    return memory_budget // max_bytes_per_row


def _get_max_buffer_size_dense(
    array: tiledb.Array,
    schema: TensorSchema,
    memory_budget: Optional[int] = None,
) -> int:
    assert not array.schema.sparse
    config_memory_budget = int(array._ctx_().config()["sm.memory_budget"])
    if memory_budget is None or memory_budget > config_memory_budget:
        memory_budget = config_memory_budget

    # The memory budget should be large enough to read the cells of the largest attribute
    bytes_per_cell = max(array.attr(attr).dtype.itemsize for attr in schema.attrs)

    # We want to be reading tiles following the tile extents along each dimension.
    # The number of cells for each such tile is the product of all tile extents.
    dim_tiles = [array.dim(dim).tile for dim in schema.dims]
    cells_per_tile = np.prod(dim_tiles)

    # Each slice consists of `rows_per_slice` rows along the first `schema` dimension
    rows_per_slice = dim_tiles[0]

    # Reading a slice of `rows_per_slice` rows requires reading a number of tiles that
    # depends on the size and tile extent of each dimension after the first one.
    assert len(schema.shape) == len(dim_tiles)
    tiles_per_slice = np.prod(
        [ceil(size / tile) for size, tile in zip(schema.shape[1:], dim_tiles[1:])]
    )

    # Compute the size in bytes of each slice of `rows_per_slice` rows
    bytes_per_slice = bytes_per_cell * cells_per_tile * tiles_per_slice

    # Compute the number of slices that fit within the memory budget
    num_slices = memory_budget // bytes_per_slice

    # Compute the total number of rows to slice
    return int(rows_per_slice * num_slices)
