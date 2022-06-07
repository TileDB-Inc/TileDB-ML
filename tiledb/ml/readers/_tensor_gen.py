from __future__ import annotations

from operator import itemgetter
from typing import Callable, Generic, Iterable, Sequence, TypeVar, Union

import numpy as np
import sparse

import tiledb

from ._tensor_schema import KeyRange, TensorSchema

T = TypeVar("T")


class TileDBNumpyGenerator:
    """Generator of Numpy arrays read from a TileDB array."""

    def __init__(self, array: tiledb.Array, schema: TensorSchema):
        """
        :param array: TileDB array to read from.
        :param schema: Tensor schema.
        """
        self._array = array
        self._schema = schema

    def __call__(
        self, key_ranges: Iterable[KeyRange]
    ) -> Union[Iterable[np.ndarray], Iterable[Sequence[np.ndarray]]]:
        """
        Generate batches of Numpy arrays read from a TileDB array.

        Each yielded batch is either:
        - a sequence of N arrays if N > 1, where `N == len(self._schema.fields)`, or
        - a single array if N == 1.
        Each array has shape `(len(key_range), *self._schema.shape[1:])`.

        If `self._schema.key_dim index > 0`, the returned Numpy arrays will ve reshaped
        so that the key_dim axes is first. For example, is the TileDB array `a` has shape
        (5, 12, 20) and `key_dim_index==1`, the returned Numpy arrays of `a[:, 4:8, :]`
        have shape (5, 4, 20) but this method returns arrays of shape (4, 5, 20).

        :param key_ranges: Inclusive ranges along the key dimension.
        """
        get_data = itemgetter(*self._schema.fields)
        key_dim_index = self._schema.key_dim_index
        for key_range in key_ranges:
            field_arrays = self._schema[key_range.min : key_range.max]
            if key_dim_index > 0:
                # Move key_dim_index axes first
                for field, array in field_arrays.items():
                    field_arrays[field] = np.moveaxis(array, key_dim_index, 0)
            yield get_data(field_arrays)


class TileDBSparseGenerator(Generic[T]):
    """Generator of sparse tensors read from a TileDB array."""

    def __init__(
        self,
        array: tiledb.Array,
        schema: TensorSchema,
        from_coo: Callable[[sparse.COO], T],
    ):
        """
        :param array: TileDB array to read from.
        :param schema: Tensor schema.
        :param from_coo: Function to convert a sparse.COO instance to a `T` sparse tensor.
        """
        self._array = array
        self._schema = schema
        self._from_coo = from_coo

    def __call__(
        self, key_ranges: Iterable[KeyRange]
    ) -> Union[Iterable[T], Iterable[Sequence[T]]]:
        """
        Generate batches of sparse tensors read from a TileDB array.

        Each yielded batch is either:
        - a sequence of N arrays if N > 1, where `N == len(self._schema.fields)`, or
        - a single tensor if N == 1.
        Each tensor is a `T` instance of shape `(len(key_ranges), *self._schema.shape[1:])`.

        :param key_ranges: Inclusive ranges along the key dimension.
        """
        single_field = len(self._schema.fields) == 1
        get_data = itemgetter(*self._schema.fields)

        dims = self._schema.dims
        dim_starts = tuple(map(itemgetter(0), self._schema.nonempty_domain))
        shape = list(self._schema.shape)

        for key_range in key_ranges:
            # Set the shape of the key dimension equal to the current key range length
            shape[0] = len(key_range)
            field_arrays = self._schema[key_range.min : key_range.max]
            data = get_data(field_arrays)

            # Convert coordinates from the original domain to zero-based
            # For the key (i.e. first) dimension, ignore the keys before the current range
            coords = tuple(field_arrays[dim] for dim in dims)
            for i, (coord, dim_start) in enumerate(zip(coords, dim_starts)):
                coord -= dim_start if i > 0 else key_range.min

            # yield either a single tensor or a sequence of tensors, one for each field
            if single_field:
                yield self._from_coo(sparse.COO(coords, data, shape))
            else:
                yield tuple(self._from_coo(sparse.COO(coords, d, shape)) for d in data)
