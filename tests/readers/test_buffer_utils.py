import numpy as np
import pytest

import tiledb
from tiledb.ml.readers._buffer_utils import estimate_row_bytes, get_max_buffer_size
from tiledb.ml.readers._tensor_gen import TensorSchema


@pytest.fixture
def dense_uri(tmp_path):
    uri = str(tmp_path / "dense")
    schema = tiledb.ArraySchema(
        sparse=False,
        domain=tiledb.Domain(
            tiledb.Dim(name="d0", domain=(0, 9999), dtype=np.uint32, tile=123),
            tiledb.Dim(name="d1", domain=(1, 5), dtype=np.uint32, tile=2),
            tiledb.Dim(name="d2", domain=(1, 2), dtype=np.uint32, tile=1),
        ),
        attrs=[
            tiledb.Attr(name="af8", dtype=np.float64),
            tiledb.Attr(name="af4", dtype=np.float32),
            tiledb.Attr(name="au1", dtype=np.uint8),
        ],
    )
    tiledb.Array.create(uri, schema)
    with tiledb.open(uri, "w") as a:
        size = a.schema.domain.size
        a[:] = {
            "af8": np.random.rand(size),
            "af4": np.random.rand(size).astype(np.float32),
            "au1": np.random.randint(128, size=size, dtype=np.uint8),
        }
    return uri


@pytest.fixture
def sparse_uri(tmp_path):
    uri = str(tmp_path / "sparse")
    schema = tiledb.ArraySchema(
        sparse=True,
        allows_duplicates=True,
        domain=tiledb.Domain(
            tiledb.Dim(name="d0", domain=(0, 999), dtype=np.int32),
            tiledb.Dim(name="d1", domain=(-250, 249), dtype=np.int32),
            tiledb.Dim(name="d2", domain=(1, 10), dtype=np.int32),
        ),
        attrs=[
            tiledb.Attr(name="af8", dtype=np.float64),
            tiledb.Attr(name="af4", dtype=np.float32),
            tiledb.Attr(name="au1", dtype=np.uint8),
        ],
    )
    tiledb.Array.create(uri, schema)
    with tiledb.open(uri, "w") as a:
        num_rows = 1000
        cells_per_row = 3
        num_cells = num_rows * cells_per_row
        d0 = np.concatenate(
            [np.arange(num_rows, dtype=np.uint32) for _ in range(cells_per_row)]
        )
        d1 = np.random.randint(-250, 250, num_cells).astype(np.int32)
        d2 = np.random.randint(1, 11, num_cells).astype(np.uint16)
        a[d0, d1, d2] = {
            "af8": np.random.rand(num_cells),
            "af4": np.random.rand(num_cells).astype(np.float32),
            "au1": np.random.randint(128, size=num_cells, dtype=np.uint8),
        }
    return uri


@pytest.mark.parametrize("key_dim,row_cells", [(0, 10), (1, 20000), (2, 50000)])
def test_estimate_row_bytes_dense(dense_uri, key_dim, row_cells):
    with tiledb.open(dense_uri) as a:
        # 8+4+1=13 bytes/cell
        schema = TensorSchema(a.schema, key_dim)
        assert estimate_row_bytes(a, schema) == row_cells * 13
        # 8+1=9 bytes/cell
        schema = TensorSchema(a.schema, key_dim, ["af8", "au1"])
        assert estimate_row_bytes(a, schema) == row_cells * 9
        # 4 bytes/cell
        schema = TensorSchema(a.schema, key_dim, ["af4"])
        assert estimate_row_bytes(a, schema) == row_cells * 4


@pytest.mark.parametrize("key_dim,row_cells", [(0, 3), (1, 6), (2, 300)])
def test_estimate_row_bytes_sparse(sparse_uri, key_dim, row_cells):
    with tiledb.open(sparse_uri) as a:
        # 3*4 bytes for dims + 8+4+1=13 bytes for attrs = 25 bytes/cell
        schema = TensorSchema(a.schema, key_dim)
        assert estimate_row_bytes(a, schema) == row_cells * 25
        # 3*4 bytes for dims + 8+1=9 bytes for attrs = 21 bytes/cell
        schema = TensorSchema(a.schema, key_dim, ["af8", "au1"])
        assert estimate_row_bytes(a, schema) == row_cells * 21
        # 3*4 bytes for dims + 4 bytes for attrs = 16 bytes/cell
        schema = TensorSchema(a.schema, key_dim, ["af4"])
        assert estimate_row_bytes(a, schema) == row_cells * 16


@pytest.mark.parametrize(
    "attrs",
    [(), ("af8",), ("af4",), ("au1",), ("af8", "af4"), ("af8", "au1"), ("af4", "au1")],
)
@pytest.mark.parametrize(
    "key_dim_index,memory_budget",
    [
        (0, 16_000),
        (0, 32_000),
        (0, 64_000),
        (1, 500_000),
        (1, 600_000),
        (1, 700_000),
    ],
)
def test_get_max_buffer_size(dense_uri, attrs, key_dim_index, memory_budget):
    config = {
        "sm.memory_budget": memory_budget,
        "py.max_incomplete_retries": 0,
    }
    with tiledb.scope_ctx(config), tiledb.open(dense_uri) as a:
        schema = TensorSchema(a.schema, key_dim_index, attrs)
        buffer_size = get_max_buffer_size(a, schema)
        # Check that the buffer size is a multiple of the row tile extent
        assert buffer_size % a.dim(key_dim_index).tile == 0

        # Check that we can slice with buffer_size without incomplete reads
        offsets = range(schema.start_key, schema.stop_key, buffer_size)
        query = a.query(attrs=schema.attrs)
        for offset in offsets:
            query[schema[offset : offset + buffer_size]]

        if buffer_size < a.shape[key_dim_index]:
            # Check that buffer_size is the max size we can slice without incomplete reads
            with pytest.raises(tiledb.TileDBError) as ex:
                query[schema[offsets[0] : offsets[0] + buffer_size + 1]]
            assert "py.max_incomplete_retries" in str(ex.value)
