from math import ceil
from typing import Optional, Sequence

import numpy as np

import tiledb


def get_buffer_size(
    array: tiledb.Array,
    attrs: Sequence[str] = (),
    memory_budget: Optional[int] = None,
    start_offset: int = 0,
    stop_offset: int = 0,
) -> int:
    if not stop_offset:
        stop_offset = array.shape[0]
    if array.schema.sparse:
        # TODO: implement get_max_buffer_size() for sparse arrays
        row_bytes = estimate_row_bytes(array, attrs, start_offset, stop_offset)
        buffer_size = (memory_budget or 100 * 1024**2) // row_bytes
    else:
        buffer_size = get_max_buffer_size(array.schema, attrs, memory_budget)
    # clip the buffer size between 1 and total number of rows
    return max(1, min(buffer_size, stop_offset - start_offset))


def estimate_row_bytes(
    array: tiledb.Array,
    attrs: Sequence[str] = (),
    start_offset: int = 0,
    stop_offset: int = 0,
) -> int:
    """
    Estimate the size in bytes of a TileDB array row.

    A "row" is a slice with the first dimension fixed.
    - For dense arrays, each row consists of a fixed number of cells. The size of each
      cell depends on the given attributes (or all array attributes by default).
    - For sparse arrays, each row consists of a variable number of non-empty cells. The
      size of each non-empty cell depends on all dimension coordinates as well as the
      given attributes (or all array attributes by default).
    """
    schema = array.schema
    if not attrs:
        attrs = get_attr_names(schema)

    if not schema.sparse:
        # for dense arrays the size of each row is fixed and can be computed exactly
        row_cells = np.prod(schema.shape[1:])
        cell_bytes = sum(schema.attr(attr).dtype.itemsize for attr in attrs)
        est_row_bytes = row_cells * cell_bytes
    else:
        # for sparse arrays the size of each row is variable and can only be estimated
        if not stop_offset:
            stop_offset = schema.shape[0]
        query = array.query(return_incomplete=True)
        # .multi_index[] is inclusive, so we need to subtract 1 to stop_offset
        indexer = query.multi_index[start_offset : stop_offset - 1]
        est_rs = indexer.estimated_result_sizes()
        dims = get_dim_names(schema)
        est_total_bytes = sum(est_rs[key].data_bytes for key in (*dims, *attrs))
        est_row_bytes = est_total_bytes / (stop_offset - start_offset)
    return int(est_row_bytes)


def get_max_buffer_size(
    schema: tiledb.ArraySchema,
    attrs: Sequence[str] = (),
    memory_budget: Optional[int] = None,
) -> int:
    """
    Get the maximum number of "rows" that can be read from an array with the given schema
    without incurring incomplete reads.

    A "row" is a slice with the first dimension fixed.

    :param schema: The array schema.
    :param attrs: The attributes to read; defaults to all array attributes.
    :param memory_budget: The maximum amount of memory to use. This is bounded by
        `tiledb.default_ctx().config()["sm.memory_budget"]`, which is also used as the
        default memory_budget.
    """
    if schema.sparse:
        raise NotImplementedError(
            "get_max_buffer_size() is not implemented for sparse arrays"
        )

    config_memory_budget = int(tiledb.default_ctx().config()["sm.memory_budget"])
    if memory_budget is None or memory_budget > config_memory_budget:
        memory_budget = config_memory_budget

    # The memory budget should be large enough to read the cells of the largest attribute
    if not attrs:
        attrs = get_attr_names(schema)
    bytes_per_cell = max(schema.attr(attr).dtype.itemsize for attr in attrs)

    # We want to be reading tiles following the tile extents along each dimension.
    # The number of cells for each such tile is the product of all tile extents.
    dim_tiles = tuple(int(schema.domain.dim(idx).tile) for idx in range(schema.ndim))
    cells_per_tile = np.prod(dim_tiles)

    # Reading a slice of dim_tiles[0] rows requires reading a number of tiles that
    # depends on the size and tile extent of each dimension after the first one.
    assert len(schema.shape) == len(dim_tiles)
    tiles_per_slice = np.prod(
        tuple(
            ceil(dim_size / dim_tile)
            for dim_size, dim_tile in zip(schema.shape[1:], dim_tiles[1:])
        )
    )

    # Compute the size in bytes of each slice of dim_tiles[0] rows
    bytes_per_slice = int(bytes_per_cell * cells_per_tile * tiles_per_slice)

    # Compute the number of slices that fit within the memory budget
    num_slices = memory_budget // bytes_per_slice

    # Compute the total number of rows to slice
    return dim_tiles[0] * num_slices


def get_attr_names(schema: tiledb.ArraySchema) -> Sequence[str]:
    return tuple(schema.attr(idx).name for idx in range(schema.nattr))


def get_dim_names(schema: tiledb.ArraySchema) -> Sequence[str]:
    return tuple(schema.domain.dim(idx).name for idx in range(schema.ndim))
