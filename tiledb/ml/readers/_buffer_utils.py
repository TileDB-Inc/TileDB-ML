from math import ceil
from typing import Optional

import numpy as np

import tiledb

from ._tensor_gen import TensorSchema


def get_buffer_size(
    array: tiledb.Array,
    schema: TensorSchema,
    memory_budget: Optional[int] = None,
) -> int:
    """Estimate the minimum number of "rows" than can fit in the given memory budget.

    A "row" is a slice along the `schema.key_dim_index` dimension where each cell consists
    of the `schema.attrs` attributes.

    :param array: TileDB array to read from.
    :param schema: TensorSchema for the array.
    :param memory_budget: The maximum amount of memory to use. This is bounded by
        `tiledb.default_ctx().config()["sm.memory_budget"]`, which is also used as the
        default memory_budget.
    """
    if array.schema.sparse:
        # TODO: implement get_max_buffer_size() for sparse arrays
        row_bytes = estimate_row_bytes(array, schema)
        buffer_size = (memory_budget or 100 * 1024**2) // row_bytes
    else:
        buffer_size = get_max_buffer_size(array, schema, memory_budget)
    # clip the buffer size between 1 and total number of rows
    return max(1, min(buffer_size, schema.stop_key - schema.start_key))


def estimate_row_bytes(array: tiledb.Array, schema: TensorSchema) -> int:
    """
    Estimate the size in bytes of a TileDB array "row".

    For dense arrays, each row consists of a fixed number of cells. The size of each
    cell depends on `schema.attrs`.
    For sparse arrays, each row consists of a variable number of non-empty cells. The
    size of each non-empty cell depends on `schema.dims` and `schema.attrs`.
    """
    if not array.schema.sparse:
        # for dense arrays the size of each row is fixed and can be computed exactly
        row_cells = np.prod(schema.shape[1:])
        cell_bytes = sum(array.attr(attr).dtype.itemsize for attr in schema.attrs)
        est_row_bytes = row_cells * cell_bytes
    else:
        # for sparse arrays the size of each row is variable and can only be estimated
        query = array.query(return_incomplete=True)
        # .multi_index[] is inclusive, so we need to subtract 1 from stop_key
        index_tuple = schema[schema.start_key : schema.stop_key - 1]
        est_rs = query.multi_index[index_tuple].estimated_result_sizes()
        est_total_bytes = sum(
            est_rs[key].data_bytes for key in (*schema.dims, *schema.attrs)
        )
        est_row_bytes = est_total_bytes / (schema.stop_key - schema.start_key)
    return int(est_row_bytes)


def get_max_buffer_size(
    array: tiledb.Array,
    schema: TensorSchema,
    memory_budget: Optional[int] = None,
) -> int:
    """
    Get the maximum number of "rows" that can be read from an array with the given schema
    without incurring incomplete reads.
    """
    if array.schema.sparse:
        raise NotImplementedError(
            "get_max_buffer_size() is not implemented for sparse arrays"
        )

    config_memory_budget = int(tiledb.default_ctx().config()["sm.memory_budget"])
    if memory_budget is None or memory_budget > config_memory_budget:
        memory_budget = config_memory_budget

    # The memory budget should be large enough to read the cells of the largest attribute
    bytes_per_cell = max(array.attr(attr).dtype.itemsize for attr in schema.attrs)

    # We want to be reading tiles following the tile extents along each dimension.
    # The number of cells for each such tile is the product of all tile extents.
    dim_tiles = [int(array.dim(dim).tile) for dim in schema.dims]
    cells_per_tile = np.prod(dim_tiles)

    # Each slice consists of `rows_per_slice` rows along the first `schema` dimension
    rows_per_slice = dim_tiles[0]

    # Reading a slice of `rows_per_slice` rows requires reading a number of tiles that
    # depends on the size and tile extent of each dimension after the first one.
    assert len(schema.shape) == len(dim_tiles)
    tiles_per_slice = np.prod(
        [ceil(size / tile) for size, tile in zip(schema.shape[1:], dim_tiles[1:])]
    )

    # Compute the size in bytes of each slice of `rows_per_slice` rows
    bytes_per_slice = int(bytes_per_cell * cells_per_tile * tiles_per_slice)

    # Compute the number of slices that fit within the memory budget
    num_slices = memory_budget // bytes_per_slice

    # Compute the total number of rows to slice
    return rows_per_slice * num_slices
