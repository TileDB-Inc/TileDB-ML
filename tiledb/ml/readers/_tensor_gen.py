from __future__ import annotations

from abc import ABC, abstractmethod
from operator import itemgetter
from typing import Callable, Generic, Iterator, Sequence, Tuple, TypeVar, Union

import numpy as np
import sparse

import tiledb


class TensorSchema:
    """
    A class to encapsulate the information needed for mapping a TileDB array to tensors.
    """

    def __init__(
        self,
        schema: tiledb.ArraySchema,
        key_dim: Union[int, str] = 0,
        attrs: Sequence[str] = (),
    ):
        """
        :param schema: Schema of the TileDB array to read from.
        :param key_dim: Name or index of the key dimension; defaults to the first dimension.
        :param attrs: Attribute names of array to read; defaults to all attributes.
        """
        shape = list(schema.shape)
        dims = [schema.domain.dim(idx).name for idx in range(schema.ndim)]
        if isinstance(key_dim, int):
            key_dim_index = key_dim
        else:
            key_dim_index = dims.index(key_dim)
        if key_dim_index > 0:
            # Swap key dimension to first position
            shape[0], shape[key_dim_index] = shape[key_dim_index], shape[0]
            dims[0], dims[key_dim_index] = dims[key_dim_index], dims[0]

        all_attrs = [schema.attr(idx).name for idx in range(schema.nattr)]
        unknown_attrs = [attr for attr in attrs if attr not in all_attrs]
        if unknown_attrs:
            raise ValueError(f"Unknown attributes: {unknown_attrs}")

        self._shape = tuple(shape)
        self._dims = tuple(dims)
        self._attrs = tuple(attrs or all_attrs)
        self._key_bounds = tuple(map(int, schema.domain.dim(key_dim_index).domain))
        self._leading_dim_slices = (slice(None),) * key_dim_index

    @property
    def attrs(self) -> Sequence[str]:
        """The attribute names of the array to read."""
        return self._attrs

    @property
    def dims(self) -> Sequence[str]:
        """The dimension names of the array."""
        return self._dims

    @property
    def shape(self) -> Tuple[int, ...]:
        """The shape of the array, with the key dimension moved first."""
        return self._shape

    @property
    def key_dim_index(self) -> int:
        """The index of the key dimension in the original TileDB schema."""
        return len(self._leading_dim_slices)

    @property
    def start_key(self) -> int:
        """The minimum value of the key dimension."""
        return self._key_bounds[0]

    @property
    def stop_key(self) -> int:
        """The maximum value of the key dimension, plus 1."""
        return self._key_bounds[1] + 1

    def __getitem__(self, key_dim_slice: slice) -> Tuple[slice, ...]:
        """Return the indexing tuple for querying the TileDB array by `dim_key=key_dim_slice`.

        For example, if `self.key_dim_index == 2`, then querying by the key dimension
        would be `array[:, :, key_dim_slice]`, which corresponds to the indexing tuple
        `(slice(None), slice(None), key_dim_slice)`.
        """
        return (*self._leading_dim_slices, key_dim_slice)

    def ensure_equal_keys(self, other: TensorSchema) -> None:
        """Ensure that the key dimension bounds of the of two schemas are equal.

        :raises ValueError: If the key dimension bounds are not equal.
        """
        if self._key_bounds != other._key_bounds:
            raise ValueError(
                f"X and Y arrays have different keys: "
                f"{self._key_bounds} != {other._key_bounds}"
            )


T = TypeVar("T")


class TileDBTensorGenerator(ABC, Generic[T]):
    """Generator of tensors read from a TileDB array."""

    def __init__(self, array: tiledb.Array, schema: TensorSchema):
        """
        :param array: TileDB array to read from.
        :param schema: Tensor schema.
        """
        self._array = array
        self._schema = schema

    @property
    def single_attr(self) -> bool:
        """Whether this generator reads a single attribute."""
        return len(self._schema.attrs) == 1

    @abstractmethod
    def iter_tensors(
        self, slice_size: int, start_key: int, stop_key: int
    ) -> Union[Iterator[T], Iterator[Sequence[T]]]:
        """
        Generate batches of tensors.

        Each yielded batch is either:
        - a sequence of N tensors where `N == len(self.attrs)` if N > 1, or
        - a single tensor if N == 1.
        Each tensor is a `T` instance of shape `(slice_size, *self._schema.shape[1:])`.

        :param slice_size: Size of each slice along the key dimension.
        :param start_key: Start value along the key dimension.
        :param stop_key: Stop value along the key dimension.
        """


class TileDBNumpyGenerator(TileDBTensorGenerator[np.ndarray]):
    """Generator of Numpy arrays read from a TileDB array."""

    def iter_tensors(
        self, slice_size: int, start_key: int, stop_key: int
    ) -> Union[Iterator[np.ndarray], Iterator[Sequence[np.ndarray]]]:
        """
        If `self._schema.key_dim index > 0`, the returned Numpy arrays will ve reshaped
        so that the key_dim axes is first. For example, is the TileDB array `a` has shape
        (5, 12, 20) and `key_dim_index==1`, the returned Numpy arrays of `a[:, 4:8, :]`
        have shape (5, 4, 20) but this method returns arrays of shape (4, 5, 20).
        """
        query = self._array.query(attrs=self._schema.attrs)
        get_data = itemgetter(*self._schema.attrs)
        key_dim_index = self._schema.key_dim_index
        for key_dim_slice in iter_slices(start_key, stop_key, slice_size):
            attr_dict = query[self._schema[key_dim_slice]]
            if key_dim_index > 0:
                # Move key_dim_index axes first
                for attr, array in attr_dict.items():
                    attr_dict[attr] = np.moveaxis(array, key_dim_index, 0)
            yield get_data(attr_dict)


class TileDBSparseGenerator(TileDBTensorGenerator[T]):
    """Generator of sparse.COO tensors read from a TileDB array."""

    def __init__(
        self,
        array: tiledb.Array,
        schema: TensorSchema,
        from_coo: Callable[[sparse.COO], T],
    ):
        super().__init__(array, schema)
        self._from_coo = from_coo

    def iter_tensors(
        self, slice_size: int, start_key: int, stop_key: int
    ) -> Union[Iterator[T], Iterator[Sequence[T]]]:
        shape = list(self._schema.shape)
        query = self._array.query(attrs=self._schema.attrs)
        get_coords = itemgetter(*self._schema.dims)
        get_data = itemgetter(*self._schema.attrs)
        single_attr = self.single_attr
        for key_dim_slice in iter_slices(start_key, stop_key, slice_size):
            shape[0] = key_dim_slice.stop - key_dim_slice.start
            attr_dict = query[self._schema[key_dim_slice]]
            coords = get_coords(attr_dict)
            # normalize the key dimension so that it starts at zero
            np.subtract(coords[0], key_dim_slice.start, out=coords[0])
            data = get_data(attr_dict)
            if single_attr:
                yield self._from_coo(sparse.COO(coords, data, shape))
            else:
                yield tuple(self._from_coo(sparse.COO(coords, d, shape)) for d in data)


def iter_slices(start: int, stop: int, step: int) -> Iterator[slice]:
    offsets = range(start, stop, step)
    yield from map(slice, offsets, offsets[1:])
    yield slice(offsets[-1], stop)
