import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, Union

import numpy as np

import tiledb

if TYPE_CHECKING:
    from ._tensor_schema import TensorSchema

Selector = Union[slice, Sequence[int]]


class TensorKind(enum.Enum):
    """Kind of tensor."""

    DENSE = enum.auto()
    SPARSE_COO = enum.auto()
    SPARSE_CSR = enum.auto()
    RAGGED = enum.auto()


@dataclass(frozen=True)
class ArrayParams:
    """
    Data class specifying the parameters for loading tensors from a TileDB array.

    Public attributes:
    - array: TileDB array to be accessed
    - key_dim: Name (or index) of the array key dimension. Defaults to the first dimension.
    - fields: Fields (dimensions and attributes) to be retrieved from array. Defaults to
      all attributes of the array.
    - dim_selectors: Mapping from dimension name to a slice or sequence of indices of this
      dimension to select.
    - tensor_kind: kind of tensor desired. If not specified, it is determined based on the
      array schema.
    """

    array: tiledb.Array
    key_dim: Union[int, str] = 0
    fields: Sequence[str] = ()
    dim_selectors: Mapping[str, Selector] = field(default_factory=dict)
    tensor_kind: Optional[TensorKind] = None
    _tensor_schema_kwargs: Mapping[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        all_attrs = [self.array.attr(i).name for i in range(self.array.nattr)]
        all_dims = [self.array.dim(i).name for i in range(self.array.ndim)]
        dims = []

        if self.fields:
            attrs = []
            for f in self.fields:
                if f in all_attrs:
                    attrs.append(f)
                elif f in all_dims:
                    dims.append(f)
                else:
                    raise ValueError(f"Unknown attribute or dimension '{f}'")
            final_fields = self.fields
        else:
            final_fields = attrs = all_attrs

        ned = list(self.array.nonempty_domain())
        key_dim_index = (
            self.key_dim
            if isinstance(self.key_dim, int)
            else all_dims.index(self.key_dim)
        )

        if key_dim_index > 0:
            # Swap key dimension to first position
            all_dims[0], all_dims[key_dim_index] = all_dims[key_dim_index], all_dims[0]
            ned[0], ned[key_dim_index] = ned[key_dim_index], ned[0]

        dim_selector_indices = {}
        for dim, selector in self.dim_selectors.items():
            i = all_dims.index(dim)
            if not isinstance(selector, (slice, Sequence)):
                raise TypeError("dim_selectors values must be slices or sequences")
            if i == 0:
                # key dimension selector
                if not isinstance(selector, slice):
                    raise TypeError("Key dimension selector must be a slice")
                if selector.step is not None:
                    raise ValueError("Stepped slice ranges are not supported")
                # intersect the non-empty domain of the key dimension with the selected slice
                min_key, max_key = ned[0]
                if selector.start is not None and selector.start > min_key:
                    min_key = selector.start
                if selector.stop is not None and selector.stop < max_key:
                    max_key = selector.stop
                selector = slice(min_key, max_key)
            dim_selector_indices[i] = selector

        if self.tensor_kind is not None:
            tensor_kind = self.tensor_kind
        elif not self.array.schema.sparse:
            tensor_kind = TensorKind.DENSE
        elif not all(
            np.issubdtype(self.array.dim(dim).dtype, np.integer) for dim in all_dims[1:]
        ):
            tensor_kind = TensorKind.RAGGED
        else:
            tensor_kind = TensorKind.SPARSE_COO

        tensor_schema_kwargs = dict(
            kind=tensor_kind,
            _array=self.array,
            _key_dim_index=key_dim_index,
            _fields=tuple(final_fields),
            _all_dims=tuple(all_dims),
            _ned=tuple(ned),
            _dim_selectors=dim_selector_indices,
            _query_kwargs={"attrs": tuple(attrs), "dims": tuple(dims)},
        )
        object.__setattr__(self, "_tensor_schema_kwargs", tensor_schema_kwargs)

    @property
    def tensor_schema(self) -> "TensorSchema[Any]":
        # local import to avoid cyclic import
        from ._tensor_schema import TensorSchemaFactories

        kwargs = self._tensor_schema_kwargs
        factory = TensorSchemaFactories[self.array.schema.sparse, kwargs["kind"]]
        return factory(**kwargs)
