from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np

import tiledb

from ..types import Selector, TensorKind
from .query import KeyDimQuery
from .ranges import InclusiveRange

Tensor = TypeVar("Tensor")


@dataclass(frozen=True)
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
    _dim_selectors: Dict[int, Selector]
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
        """Shape of the selected array, with the key dimension moved first.

        The size of each dimension is determined by:
        - either the respective selector in `_dim_selectors` (if given)
        - or by the non-empty domain otherwise.

        :raises ValueError: If the array does not have integer domain.
        """
        shape = [len(self.key_range)]
        for i, (dim_start, dim_stop) in enumerate(self._ned[1:], 1):
            selector = self._dim_selectors.get(i)
            if selector is None:
                selector = slice(None)
            if isinstance(selector, slice):
                start = selector.start if selector.start is not None else dim_start
                stop = selector.stop if selector.stop is not None else dim_stop
                if not (isinstance(start, int) and isinstance(stop, int)):
                    raise ValueError("Shape not defined for non-integer domain")
                shape.append(stop - start + 1)
            else:
                shape.append(len(selector))
        return tuple(shape)

    @property
    def query(self) -> KeyDimQuery:
        """A sliceable object for querying the TileDB array along the key dimension"""
        return self._get_query(**self._query_kwargs)

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

    def _get_query(self, **kwargs: Any) -> KeyDimQuery:
        return KeyDimQuery(
            self._array, self._key_dim_index, self._dim_selectors, **kwargs
        )
