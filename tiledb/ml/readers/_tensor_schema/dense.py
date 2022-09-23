from math import ceil
from operator import itemgetter
from typing import Any, Iterable, Sequence, Union

import numpy as np

from .base import TensorSchema
from .ranges import ConstrainedPartitionsIntRange, InclusiveRange


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
    def key_range(self) -> ConstrainedPartitionsIntRange:
        self._key_range: ConstrainedPartitionsIntRange
        try:
            return self._key_range
        except AttributeError:
            key_dim_min, key_dim_max = self._ned[0]
            key_dim_slice = self._dim_selectors.get(0)
            if key_dim_slice is not None:
                assert isinstance(key_dim_slice, slice)
                min_key = key_dim_slice.start
                max_key = key_dim_slice.stop
            else:
                min_key = key_dim_min
                max_key = key_dim_max
            key_dim_tile = self._array.dim(self.key_dim).tile
            start_offsets = range(key_dim_min, key_dim_max + 1, key_dim_tile)
            self._key_range = ConstrainedPartitionsIntRange(
                min_key, max_key, start_offsets
            )
            return self._key_range

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
        # depends on the size, tile extent and selector of each dimension after the first one.
        shape = self.shape
        tiles_per_slice = 1
        for i, dim_tile in enumerate(dim_tiles[1:], 1):
            selector = self._dim_selectors.get(i)
            if selector is None:
                tiles_per_slice *= ceil(shape[i] / dim_tile)
            else:
                dim_start, dim_stop = self._ned[i]

                def get_tile_idx(value: int) -> int:
                    """Get the index of the tile with the given value in the i-th dimension"""
                    return int((value - dim_start) / dim_tile)

                if isinstance(selector, slice):
                    # count the number of tiles between start and stop (inclusive)
                    start = selector.start if selector.start is not None else dim_start
                    stop = selector.stop if selector.stop is not None else dim_stop
                    tiles_per_slice *= get_tile_idx(stop) - get_tile_idx(start) + 1
                else:
                    # count the number of unique tiles for the i-th dimension
                    tiles_per_slice *= len(set(map(get_tile_idx, selector)))

        # Compute the size in bytes of each slice of `rows_per_slice` rows
        bytes_per_slice = bytes_per_cell * cells_per_tile * tiles_per_slice

        # Compute the number of slices that fit within the memory budget
        num_slices = memory_budget // bytes_per_slice

        # Compute the total number of rows to slice
        return max(1, int(rows_per_slice * num_slices))
