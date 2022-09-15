from operator import itemgetter
from typing import Any, Iterable, Optional, Sequence, Union

import numpy as np

from .base_sparse import BaseSparseTensorSchema
from .ranges import InclusiveRange

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
