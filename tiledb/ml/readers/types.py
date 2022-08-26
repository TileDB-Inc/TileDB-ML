from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np

import tiledb

from ._tensor_schema import Selector, TensorKind, TensorSchema, TensorSchemaFactories


@dataclass(frozen=True)
class ArrayParams:
    """
    Class for defining the parameters for accessing an array.
    Constructor arguments:
    :param array: TileDB Array to be accessed
    :param key_dim: name or index of key dimension of array
    :param fields: fields (dimensions and attributes) to be retrieved from array
    :param secondary_slices: additional slices to slice on non-key indices. Should be a
        mapping from the desired dimension name to be sliced to a slice object, list of
        indices or single index.
    :param tensor_kind: kind of tensor desired
    :param _tensor_schema_kwargs: other kwargs to pass to the tensor schema instantiated in
    `to_tensor_schema`.
    """

    array: tiledb.Array
    key_dim: Union[int, str] = 0
    fields: Sequence[str] = ()
    secondary_slices: Mapping[str, Selector] = field(default_factory=dict)
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

        secondary_slices_indices = {}
        for dim, secondary_slice in self.secondary_slices.items():
            dim_index = all_dims.index(dim)
            if dim_index == 0:
                raise NotImplementedError("Key dimension slicing is not implemented")
            secondary_slices_indices[dim_index] = secondary_slice

        tensor_schema_kwargs = dict(
            _array=self.array,
            _key_dim_index=key_dim_index,
            _fields=tuple(final_fields),
            _all_dims=tuple(all_dims),
            _ned=tuple(ned),
            _secondary_slices=secondary_slices_indices,
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

        if self.secondary_slices and tensor_kind is not TensorKind.DENSE:
            raise NotImplementedError(
                "Slicing on secondary indices is only implemented for dense arrays"
            )

        factory = TensorSchemaFactories[tensor_kind]
        return factory(kind=tensor_kind, **self._tensor_schema_kwargs)
