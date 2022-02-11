from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Generic, Iterator, Mapping, Optional, Sequence, Type, TypeVar, Union

import numpy as np
import scipy.sparse as sp

import tiledb

Tensor = TypeVar("Tensor")


class BaseBatch(ABC, Generic[Tensor]):
    """
    Base class used for getting tensors from batches of buffers read from a TileDB array.
    """

    def __init__(
        self, schema: tiledb.ArraySchema, attrs: Sequence[str], batch_size: int
    ):
        self._schema = schema
        self._attrs = attrs
        self._batch_size = batch_size

    @abstractmethod
    def set_buffer_offset(self, buffer: Mapping[str, np.ndarray], offset: int) -> None:
        """Set the current buffer from which subsequent batches are to be read.

        :param buffer: Mapping of attribute names to numpy arrays.
        :param offset: Start offset of the buffer in the TileDB array.
        """

    @abstractmethod
    def set_batch_slice(self, batch_slice: slice) -> None:
        """Set the current batch as a slice of the set buffer.

        Must be called after `set_buffer_offset`.

        :param batch_slice: Slice of the buffer to be used as the current batch.
        """

    @abstractmethod
    def iter_tensors(self, perm_idxs: Optional[np.ndarray] = None) -> Iterator[Tensor]:
        """
        Return an iterator of tensors for the current batch, one tensor per attribute
        given in the constructor.

        Must be called after `set_batch_slice`.

        :param perm_idxs: Optional permutation indices for the current batch.
            If given, it must be an array equal to `np.arange(len(self))` after sorting.
        """

    @abstractmethod
    def __len__(self) -> int:
        """Get the size (i.e. number of rows) of the current batch.

        Must be called after `set_batch_slice`.
        """


class BaseDenseBatch(BaseBatch[Tensor]):
    def set_buffer_offset(self, buffer: Mapping[str, np.ndarray], offset: int) -> None:
        self._buffer = buffer

    def set_batch_slice(self, batch_slice: slice) -> None:
        assert hasattr(self, "_buffer"), "set_buffer_offset() not called"
        self._attr_batches = [self._buffer[attr][batch_slice] for attr in self._attrs]

    def iter_tensors(self, perm_idxs: Optional[np.ndarray] = None) -> Iterator[Tensor]:
        assert hasattr(self, "_attr_batches"), "set_batch_slice() not called"
        if perm_idxs is None:
            attr_batches = iter(self._attr_batches)
        else:
            n = len(self)
            assert (
                len(perm_idxs) == n and (np.sort(perm_idxs) == np.arange(n)).all()
            ), f"Invalid permutation of size {n}: {perm_idxs}"
            attr_batches = (attr_batch[perm_idxs] for attr_batch in self._attr_batches)
        return map(self._tensor_from_numpy, attr_batches)

    def __len__(self) -> int:
        assert hasattr(self, "_attr_batches"), "set_batch_slice() not called"
        return len(self._attr_batches[0])

    @staticmethod
    @abstractmethod
    def _tensor_from_numpy(data: np.ndarray) -> Tensor:
        """Convert a numpy array to a Tensor"""


class BaseSparseBatch(BaseBatch[Tensor]):
    def __init__(
        self, schema: tiledb.ArraySchema, attrs: Sequence[str], batch_size: int
    ):
        super().__init__(schema, attrs, batch_size)
        self._row_dim = schema.domain.dim(0).name
        self._col_dim = schema.domain.dim(1).name
        self._dense_shape = (batch_size, schema.shape[1])
        self._attr_dtypes = tuple(schema.attr(attr).dtype for attr in self._attrs)

    def set_buffer_offset(self, buffer: Mapping[str, np.ndarray], offset: int) -> None:
        # COO to CSR transformation for batching and row slicing
        row = buffer[self._row_dim]
        col = buffer[self._col_dim]
        # Normalize indices: We want the coords indices to be in the [0, batch_size]
        # range. If we do not normalize the sparse tensor is being created but with a
        # dimension [0, max(coord_index)], which is overkill
        row_size_norm = row.max() - row.min() + 1
        col_size_norm = col.max() + 1
        self._buffer_csr = sp.csr_matrix(
            (buffer[self._attrs[0]], (row - offset, col)),
            shape=(row_size_norm, col_size_norm),
        )

    def set_batch_slice(self, batch_slice: slice) -> None:
        assert hasattr(self, "_buffer_csr"), "set_buffer_offset() not called"
        self._batch_csr = self._buffer_csr[batch_slice]

    def iter_tensors(self, perm_idxs: Optional[np.ndarray] = None) -> Iterator[Tensor]:
        assert hasattr(self, "_batch_csr"), "set_batch_slice() not called"
        if perm_idxs is not None:
            raise NotImplementedError(
                "within_batch_shuffle not implemented for sparse arrays"
            )
        batch_coo = self._batch_csr.tocoo()
        data = batch_coo.data
        coords = np.stack((batch_coo.row, batch_coo.col), axis=-1)
        for dtype in self._attr_dtypes:
            yield self._tensor_from_coo(data, coords, self._dense_shape, dtype)

    def __len__(self) -> int:
        assert hasattr(self, "_batch_csr"), "set_batch_slice() not called"
        # return number of non-zero rows
        return int((self._batch_csr.getnnz(axis=1) > 0).sum())

    def __bool__(self) -> bool:
        assert hasattr(self, "_batch_csr"), "set_batch_slice() not called"
        # faster version of __len__() > 0
        return len(self._batch_csr.data) > 0

    @staticmethod
    @abstractmethod
    def _tensor_from_coo(
        data: np.ndarray,
        coords: np.ndarray,
        dense_shape: Sequence[int],
        dtype: np.dtype,
    ) -> Tensor:
        """Convert a scipy.sparse.coo_matrix to a Tensor"""


DenseTensor = TypeVar("DenseTensor")
SparseTensor = TypeVar("SparseTensor")


def tensor_generator(
    dense_batch_cls: Type[BaseDenseBatch[DenseTensor]],
    sparse_batch_cls: Type[BaseSparseBatch[SparseTensor]],
    x_array: tiledb.Array,
    y_array: tiledb.Array,
    batch_size: int,
    buffer_size: Optional[int] = None,
    batch_shuffle: bool = False,
    within_batch_shuffle: bool = False,
    x_attrs: Sequence[str] = (),
    y_attrs: Sequence[str] = (),
    start_offset: int = 0,
    stop_offset: Optional[int] = None,
) -> Iterator[Sequence[Union[DenseTensor, SparseTensor]]]:
    """
    Generator for batches of tensors.

    Each yielded batch is a sequence of N tensors of x_array followed by M tensors
    of y_array, where `N == len(x_attrs)` and `M == len(y_attrs)`.

    :param dense_batch_cls: Type of dense batches.
    :param sparse_batch_cls: Type of sparse batches.
    :param x_array: TileDB array of the features.
    :param y_array: TileDB array of the labels.
    :param batch_size: Size of each batch.
    :param buffer_size: Size of the buffer used to read the data; defaults to batch_size.
    :param batch_shuffle: True for shuffling batches.
    :param within_batch_shuffle: True for shuffling records in each batch.
    :param x_attrs: Attribute names of x_array; defaults to all x_array attributes.
    :param y_attrs: Attribute names of y_array; defaults to all y_array attributes
    :param start_offset: Start row offset; defaults to 0.
    :param stop_offset: Stop row offset; defaults to number of rows.
    """
    if buffer_size is None:
        buffer_size = batch_size
    elif buffer_size < batch_size:
        raise ValueError("Buffer size should be greater or equal to batch size")

    if stop_offset is None:
        stop_offset = x_array.shape[0]

    def batch_factory(
        schema: tiledb.ArraySchema, attrs: Sequence[str]
    ) -> Union[BaseDenseBatch[DenseTensor], BaseSparseBatch[SparseTensor]]:
        if not attrs:
            attrs = get_attr_names(schema)
        if schema.sparse:
            return sparse_batch_cls(schema, attrs, batch_size)
        return dense_batch_cls(schema, attrs, batch_size)

    x_batch = batch_factory(x_array.schema, x_attrs)
    y_batch = batch_factory(y_array.schema, y_attrs)
    with ThreadPoolExecutor(max_workers=2) as executor:
        for offset in range(start_offset, stop_offset, buffer_size):
            x_buffer, y_buffer = executor.map(
                lambda array: array[offset : offset + buffer_size],  # type: ignore
                (x_array, y_array),
            )
            x_batch.set_buffer_offset(x_buffer, offset)
            y_batch.set_buffer_offset(y_buffer, offset)

            # Split the buffer_size into batch_size chunks
            batch_offsets = np.arange(
                0, min(buffer_size, stop_offset - offset), batch_size
            )
            if batch_shuffle:
                np.random.shuffle(batch_offsets)

            for batch_offset in batch_offsets:
                batch_slice = slice(batch_offset, batch_offset + batch_size)
                x_batch.set_batch_slice(batch_slice)
                y_batch.set_batch_slice(batch_slice)
                if len(x_batch) != len(y_batch):
                    raise ValueError(
                        "x_array and y_array should have the same number of rows, "
                        "i.e. the first dimension of x_array and y_array should be "
                        "of equal domain extent inside the batch"
                    )
                if x_batch:
                    if within_batch_shuffle:
                        perm_idxs = np.arange(len(x_batch))
                        np.random.shuffle(perm_idxs)
                    else:
                        perm_idxs = None
                    yield (
                        *x_batch.iter_tensors(perm_idxs),
                        *y_batch.iter_tensors(perm_idxs),
                    )


def get_attr_names(schema: tiledb.ArraySchema) -> Sequence[str]:
    return tuple(schema.attr(idx).name for idx in range(schema.nattr))
