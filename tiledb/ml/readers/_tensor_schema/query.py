from typing import Any, List, Mapping

import tiledb

from ..types import Selector


class KeyDimQuery:
    def __init__(
        self,
        array: tiledb.Array,
        key_dim_index: int,
        dim_selectors: Mapping[int, Selector],
        **kwargs: Any,
    ):
        self._multi_index = array.query(**kwargs).multi_index
        selectors: List[Selector] = [slice(None)] * array.ndim
        for i, selector in dim_selectors.items():
            if i == 0:
                # ignore selector for the key dimension
                continue
            elif i == key_dim_index:
                # key_dim_index got swapped with 0th index
                selectors[0] = selector
            else:
                selectors[i] = selector
        self._leading_selectors = tuple(selectors[:key_dim_index])
        self._trailing_selectors = tuple(selectors[key_dim_index + 1 :])

    def __getitem__(self, key_dim_slice: slice) -> Any:
        """Query the TileDB array by `dim_key=key_dim_slice`."""
        selectors = (*self._leading_selectors, key_dim_slice, *self._trailing_selectors)
        return self._multi_index[selectors]
