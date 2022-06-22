from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import tiledb

Tensor = TypeVar("Tensor")


@dataclass(frozen=False)
class ArrayParams:
    array: tiledb.Array
    key_dim: Union[int, str]
    ned: Sequence[Tuple[int, int]]
    all_dims: Sequence[str]
    fields: Sequence[str]
    query_kwargs: MutableMapping[str, Any]
    key_dim_index: int
    transform: Optional[Callable[[Tensor], Tensor]]

    def __init__(
        self,
        array: tiledb.Array,
        key_dim: Union[int, str] = 0,
        fields: Sequence[str] = (),
        transform: Optional[Callable[[Tensor], Tensor]] = None,
    ):
        """Create a TensorSchema.

        :param array: TileDB array to read from.
        :param key_dim: Name or index of the key dimension. Defaults to the first dimension.
        :param fields: Attribute and/or dimension names of the array to read. Defaults to
            all attributes.
        :param transform: Function to transform tensors.
        """

        all_attrs = [array.attr(i).name for i in range(array.nattr)]
        all_dims = [array.dim(i).name for i in range(array.ndim)]
        dims = []
        if fields:
            attrs = []
            for field in fields:
                if field in all_attrs:
                    attrs.append(field)
                elif field in all_dims:
                    dims.append(field)
                else:
                    raise ValueError(f"Unknown attribute or dimension '{field}'")
        else:
            fields = attrs = all_attrs

        ned = list(array.nonempty_domain())
        key_dim_index = key_dim if isinstance(key_dim, int) else all_dims.index(key_dim)
        if key_dim_index > 0:
            # Swap key dimension to first position
            all_dims[0], all_dims[key_dim_index] = all_dims[key_dim_index], all_dims[0]
            ned[0], ned[key_dim_index] = ned[key_dim_index], ned[0]

        self.array = array
        self.key_dim_index = key_dim_index
        self.ned = tuple(ned)
        self.all_dims = tuple(all_dims)
        self.fields = tuple(fields)
        self.query_kwargs = {"attrs": tuple(attrs), "dims": tuple(dims)}
        self.transform = transform
