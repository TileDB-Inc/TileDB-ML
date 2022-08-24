from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np

import tiledb

from ._tensor_schema import TensorKind, TensorSchema, TensorSchemaFactories


@dataclass(frozen=True)
class ArrayParams:
    array: tiledb.Array
    key_dim: Union[int, str] = 0
    fields: Sequence[str] = ()
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

        tensor_schema_kwargs = dict(
            _array=self.array,
            _key_dim_index=key_dim_index,
            _fields=tuple(final_fields),
            _all_dims=tuple(all_dims),
            _ned=tuple(ned),
            _query_kwargs={"attrs": tuple(attrs), "dims": tuple(dims)},
        )
        object.__setattr__(self, "_tensor_schema_kwargs", tensor_schema_kwargs)

    @property
    def tensor_schema(self) -> TensorSchema[Any]:
        """Create a `TensorSchema` from this `ArrayParams` instance."""
        if self.tensor_kind is not None:
            tensor_kind = self.tensor_kind
        elif not self.array.schema.sparse:
            tensor_kind = TensorKind.DENSE
        elif not all(
            np.issubdtype(self.array.dim(dim).dtype, np.integer)
            for dim in self._tensor_schema_kwargs["_all_dims"][1:]
        ):
            tensor_kind = TensorKind.RAGGED
        else:
            tensor_kind = TensorKind.SPARSE_COO

        factory = TensorSchemaFactories[tensor_kind]
        return factory(kind=tensor_kind, **self._tensor_schema_kwargs)
