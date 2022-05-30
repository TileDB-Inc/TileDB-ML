from __future__ import annotations

from operator import itemgetter
from typing import Callable, Generic, Iterable, Sequence, TypeVar, Union

import numpy as np
import sparse

import tiledb

from ._tensor_schema import TensorSchema

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
        self, key_dim_slices: Iterable[slice]
    ) -> Union[Iterable[np.ndarray], Iterable[Sequence[np.ndarray]]]:
        """
        Generate batches of Numpy arrays read from a TileDB array.

        Each yielded batch is either:
        - a sequence of N arrays if N > 1, where `N == len(self._schema.fields)`, or
        - a single array if N == 1.
        Each array has shape `(slice_size, *self._schema.shape[1:])`.

        If `self._schema.key_dim index > 0`, the returned Numpy arrays will ve reshaped
        so that the key_dim axes is first. For example, is the TileDB array `a` has shape
        (5, 12, 20) and `key_dim_index==1`, the returned Numpy arrays of `a[:, 4:8, :]`
        have shape (5, 4, 20) but this method returns arrays of shape (4, 5, 20).

        :param key_dim_slices: Slices along the key dimension, where both start and stop
            are inclusive.
        """
        multi_index = self._schema.multi_index
        get_data = itemgetter(*self._schema.fields)
        key_dim_index = self._schema.key_dim_index
        for key_dim_slice in key_dim_slices:
            field_arrays = multi_index[self._schema[key_dim_slice]]
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
        self, key_dim_slices: Iterable[slice]
    ) -> Union[Iterable[T], Iterable[Sequence[T]]]:
        """
        Generate batches of sparse tensors read from a TileDB array.

        Each yielded batch is either:
        - a sequence of N arrays if N > 1, where `N == len(self._schema.fields)`, or
        - a single tensor if N == 1.
        Each tensor is a `T` instance of shape `(slice_size, *self._schema.shape[1:])`.

        :param key_dim_slices: Slices along the key dimension, where both start and stop
            are inclusive.
        """
        single_field = len(self._schema.fields) == 1
        multi_index = self._schema.multi_index
        get_data = itemgetter(*self._schema.fields)

        dims = self._schema.dims
        dim_starts = tuple(map(itemgetter(0), self._schema.nonempty_domain))
        shape = list(self._schema.shape)

        for key_dim_slice in key_dim_slices:
            # set the shape of the key dimension equal to the current slice size
            shape[0] = key_dim_slice.stop - key_dim_slice.start + 1
            field_arrays = multi_index[self._schema[key_dim_slice]]
            data = get_data(field_arrays)

            # convert coordinates from the original domain to zero-based
            # for the key (i.e. first) dimension, ignore the keys before the current slice
            coords = tuple(field_arrays[dim] for dim in dims)
            for i, (coord, dim_start) in enumerate(zip(coords, dim_starts)):
                coord -= dim_start if i > 0 else key_dim_slice.start

            # yield either a single tensor or a sequence of tensors, one for each field
            if single_field:
                yield self._from_coo(sparse.COO(coords, data, shape))
            else:
                yield tuple(self._from_coo(sparse.COO(coords, d, shape)) for d in data)
