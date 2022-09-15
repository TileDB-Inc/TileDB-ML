from operator import itemgetter
from typing import Any, Iterable, Sequence, Union, cast

import numpy as np
import scipy.sparse
import sparse

from .base_sparse import BaseSparseTensorSchema
from .ranges import InclusiveRange

SparseArray = Union[scipy.sparse.csr_matrix, sparse.COO]


class SparseTensorSchema(BaseSparseTensorSchema[SparseArray]):
    """
    TensorSchema for reading sparse TileDB arrays as SparseArray instances.
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._query_kwargs["dims"] = self._all_dims

    def iter_tensors(
        self, key_ranges: Iterable[InclusiveRange[Any, int]]
    ) -> Union[Iterable[SparseArray], Iterable[Sequence[SparseArray]]]:
        shape = list(cast(Sequence[int], self.shape))
        SparseArrayFactory = csr_matrix if len(shape) == 2 else sparse.COO
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

            # yield either a single SparseArray or one SparseArray per field
            if single_field:
                yield SparseArrayFactory(coords, data, shape)
            else:
                yield tuple(SparseArrayFactory(coords, d, shape) for d in data)


def csr_matrix(
    coords: np.ndarray, data: np.ndarray, shape: Sequence[int]
) -> scipy.sparse.csr_matrix:
    return scipy.sparse.csr_matrix((data, coords), shape)
