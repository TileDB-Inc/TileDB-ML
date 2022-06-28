from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, TypeVar, Union

import tiledb

Tensor = TypeVar("Tensor")


@dataclass(frozen=True)
class ArrayParams:
    array: tiledb.Array
    key_dim: Union[int, str] = 0
    fields: Sequence[str] = ()
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
                    raise ValueError(f"Unknown attribute or dimension '{field}'")
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

        object.__setattr__(
            self,
            "_tensor_schema_kwargs",
            dict(
                array=self.array,
                fields=tuple(final_fields),
                key_dim_index=key_dim_index,
                ned=tuple(ned),
                all_dims=tuple(all_dims),
                query_kwargs={"attrs": tuple(attrs), "dims": tuple(dims)},
            ),
        )
