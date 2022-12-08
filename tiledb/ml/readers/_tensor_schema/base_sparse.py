from math import ceil
from typing import Any, Counter

import numpy as np

from .base import Tensor, TensorSchema
from .ranges import WeightedRange


class BaseSparseTensorSchema(TensorSchema[Tensor]):
    """Abstract base class for reading sparse TileDB arrays."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        if not self._array.schema.sparse:
            raise NotImplementedError(
                "Mapping a dense TileDB array to sparse tensors is not implemented"
            )

    @property
    def key_range(self) -> WeightedRange[Any, int]:
        self._key_range: WeightedRange[Any, int]
        try:
            return self._key_range
        except AttributeError:
            key_counter = Counter[Any]()
            key_dim = self.key_dim
            query = self._get_query(dims=(key_dim,), attrs=(), return_incomplete=True)
            key_dim_slice = self._dim_selectors.get(0, slice(None))
            assert isinstance(key_dim_slice, slice)
            for result in query[key_dim_slice]:
                key_counter.update(result[key_dim])
            self._key_range = WeightedRange.from_mapping(key_counter)
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
        query = self._get_query(return_incomplete=True, **self._query_kwargs)
        bytes_per_cell = []
        for field, est_result_size in query[:].estimated_result_sizes().items():
            if est_result_size.offsets_bytes == 0:
                bytes_per_cell.append(attr_or_dim_dtype(field).itemsize)
            else:
                num_cells = est_result_size.offsets_bytes / offset_itemsize
                avg_itemsize = est_result_size.data_bytes / num_cells
                bytes_per_cell.append(max(avg_itemsize, offset_itemsize))

        # Finally, the number of cells that can fit in the memory_budget depends on the
        # maximum bytes_per_cell
        return max(1, memory_budget // ceil(max(bytes_per_cell)))
