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
    """Estimate the maximum number of "rows" than can fit in the given memory budget.

    A "row" is a slice along the `schema.key_dim_index` dimension where each cell consists
    of the `schema.attrs` attributes.

    :param array: TileDB array to read from.
    :param schema: TensorSchema for the array.
    :param memory_budget: The maximum amount of memory to use. This is bounded by the
        `sm.memory_budget` config parameter for dense arrays and `py.init_buffer_bytes`
        (or 10 MB if unset) for sparse arrays. These bounds are also used as the default
        memory budget.
    """
    if array.schema.sparse:
        buffer_size = _get_max_buffer_size_sparse(array, schema, memory_budget)
    else:
        buffer_size = _get_max_buffer_size_dense(array, schema, memory_budget)
    # clip the buffer size between 1 and total number of rows
    return max(1, min(buffer_size, schema.stop_key - schema.start_key))


def _get_max_buffer_size_sparse(
    array: tiledb.Array,
    schema: TensorSchema,
    memory_budget: Optional[int] = None,
) -> int:
    assert array.schema.sparse
    try:
        init_buffer_bytes = int(array._ctx_().config()["py.init_buffer_bytes"])
    except KeyError:
        init_buffer_bytes = 10 * 1024**2
    if memory_budget is None or memory_budget > init_buffer_bytes:
        memory_budget = init_buffer_bytes

    # the size of each row is variable and can only be estimated
    query = array.query(attrs=schema.attrs, return_incomplete=True)
    res_sizes = query.multi_index[:].estimated_result_sizes()

    max_buffer_bytes = max(res_size.data_bytes for res_size in res_sizes.values())
    max_bytes_per_row = max_buffer_bytes // (schema.stop_key - schema.start_key)

    return int(memory_budget // max_bytes_per_row)


def _get_max_buffer_size_dense(
    array: tiledb.Array,
    schema: TensorSchema,
    memory_budget: Optional[int] = None,
) -> int:
    assert not array.schema.sparse
    config_memory_budget = int(array._ctx_().config()["sm.memory_budget"])
    if memory_budget is None or memory_budget > config_memory_budget:
        memory_budget = config_memory_budget

    # The memory budget should be large enough to read the cells of the largest attribute
    bytes_per_cell = max(array.attr(attr).dtype.itemsize for attr in schema.attrs)

    # We want to be reading tiles following the tile extents along each dimension.
    # The number of cells for each such tile is the product of all tile extents.
    dim_tiles = [array.dim(dim).tile for dim in schema.dims]
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
    bytes_per_slice = bytes_per_cell * cells_per_tile * tiles_per_slice

    # Compute the number of slices that fit within the memory budget
    num_slices = memory_budget // bytes_per_slice

    # Compute the total number of rows to slice
    return int(rows_per_slice * num_slices)
