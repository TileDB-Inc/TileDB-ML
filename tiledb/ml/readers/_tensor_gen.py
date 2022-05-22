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
        get_dim = schema.domain.dim
        if not np.issubdtype(get_dim(key_dim).dtype, np.integer):
            raise ValueError(f"Key dimension {key_dim} must have integer domain")

        dims = [get_dim(i).name for i in range(schema.ndim)]
        key_dim_index = dims.index(key_dim) if not isinstance(key_dim, int) else key_dim
        if key_dim_index > 0:
            # Swap key dimension to first position
            dims[0], dims[key_dim_index] = dims[key_dim_index], dims[0]

        all_attrs = [schema.attr(i).name for i in range(schema.nattr)]
        unknown_attrs = [attr for attr in attrs if attr not in all_attrs]
        if unknown_attrs:
            raise ValueError(f"Unknown attributes: {unknown_attrs}")

        self._dims = tuple(dims)
        self._attrs = tuple(attrs or all_attrs)
        self._key_bounds = tuple(map(int, get_dim(key_dim).domain))
        self._leading_dim_slices = (slice(None),) * key_dim_index
        if all(np.issubdtype(get_dim(name).dtype, np.integer) for name in dims):
            domains = [get_dim(name).domain for name in dims]
            self._shape = tuple(int(stop - start + 1) for start, stop in domains)

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
        """The shape of the array, with the key dimension moved first.

        :raises ValueError: If the array does not have integer domain.
        """
        try:
            return self._shape
        except AttributeError:
            raise ValueError("Cannot infer shape from non-integer dimensions")

    @property
    def key_dim_index(self) -> int:
        """The index of the key dimension in the original TileDB schema."""
        return len(self._leading_dim_slices)

    @property
    def num_keys(self) -> int:
        """The number of distinct values along the key dimension"""
        return self.stop_key - self.start_key

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
        attrs = self._schema.attrs
        multi_index = self._array.query(attrs=attrs).multi_index
        get_data = itemgetter(*attrs)
        key_dim_index = self._schema.key_dim_index
        for key_dim_slice in iter_slices(start_key, stop_key, slice_size):
            # multi_index needs inclusive slices
            idx = self._schema[key_dim_slice.start : key_dim_slice.stop - 1]
            attr_dict = multi_index[idx]
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
        attrs = self._schema.attrs
        multi_index = self._array.query(attrs=attrs).multi_index
        get_data = itemgetter(*attrs)

        dims = self._schema.dims
        dim_starts = tuple(self._array.domain.dim(d).domain[0] for d in dims)
        single_attr = self.single_attr
        shape = list(self._schema.shape)

        # generate slices of the array along the key dimension
        for key_dim_slice in iter_slices(start_key, stop_key, slice_size):
            # set the shape of the key dimension equal to the current slice size
            shape[0] = key_dim_slice.stop - key_dim_slice.start
            # multi_index needs inclusive slices
            idx = self._schema[key_dim_slice.start : key_dim_slice.stop - 1]
            attr_dict = multi_index[idx]
            data = get_data(attr_dict)

            # convert coordinates from the original domain to zero-based
            # for the key (i.e. first) dimension, ignore the keys before the current slice
            coords = tuple(attr_dict[d] for d in dims)
            for i, (coord, dim_start) in enumerate(zip(coords, dim_starts)):
                coord -= dim_start if i > 0 else key_dim_slice.start

            # yield either a single tensor or a sequence of tensors, one for each attr
            if single_attr:
                yield self._from_coo(sparse.COO(coords, data, shape))
            else:
                yield tuple(self._from_coo(sparse.COO(coords, d, shape)) for d in data)


def iter_slices(start: int, stop: int, step: int) -> Iterator[slice]:
    offsets = range(start, stop, step)
    yield from map(slice, offsets, offsets[1:])
    yield slice(offsets[-1], stop)
