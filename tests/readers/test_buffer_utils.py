import numpy as np
import pytest

import tiledb
from tiledb.ml.readers._buffer_utils import estimate_row_bytes, get_max_buffer_size


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
            tiledb.Dim(name="d1", domain=(-5000, 5000), dtype=np.int32),
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
        d1 = np.random.randint(-5000, 5001, num_cells).astype(np.int32)
        d2 = np.random.randint(1, 11, num_cells).astype(np.uint16)
        a[d0, d1, d2] = {
            "af8": np.random.rand(num_cells),
            "af4": np.random.rand(num_cells).astype(np.float32),
            "au1": np.random.randint(128, size=num_cells, dtype=np.uint8),
        }
    return uri


def test_estimate_row_bytes_dense(dense_uri):
    with tiledb.open(dense_uri) as a:
        # 10 cells/row, 8+4+1=13 bytes/cell
        assert estimate_row_bytes(a) == 130
        # 10 cells/row, 8+1=9 bytes/cell
        assert estimate_row_bytes(a, attrs=["af8", "au1"]) == 90
        # 10 cells/row, 4 bytes/cell
        assert estimate_row_bytes(a, attrs=["af4"]) == 40


def test_estimate_row_bytes_sparse(sparse_uri):
    with tiledb.open(sparse_uri) as a:
        # 3 cells/row, 3*4 bytes for dims + 8+4+1=13 bytes for attrs = 25 bytes/cell
        assert estimate_row_bytes(a) == 75
        # 3 cells/row, 3*4 bytes for dims + 8+1=9 bytes for attrs = 21 bytes/cell
        assert estimate_row_bytes(a, attrs=["af8", "au1"]) == 63
        # 3 cells/row, 3*4 bytes for dims + 4 bytes for attrs = 16 bytes/cell
        assert estimate_row_bytes(a, attrs=["af4"]) == 48


@pytest.mark.parametrize("memory_budget", [2**i for i in range(14, 20)])
@pytest.mark.parametrize(
    "attrs",
    [(), ("af8",), ("af4",), ("au1",), ("af8", "af4"), ("af8", "au1"), ("af4", "au1")],
)
def test_get_max_buffer_size(dense_uri, memory_budget, attrs):
    config = {
        "sm.memory_budget": memory_budget,
        "py.max_incomplete_retries": 0,
    }
    with tiledb.scope_ctx(config), tiledb.open(dense_uri) as a:
        buffer_size = get_max_buffer_size(a.schema, attrs)
        # Check that the buffer size is a multiple of the row tile extent
        assert buffer_size % a.dim(0).tile == 0

        # Check that we can slice with buffer_size without incomplete reads
        query = a.query(attrs=attrs or None)
        for offset in range(0, a.shape[0], buffer_size):
            query[offset : offset + buffer_size]

        if buffer_size < a.shape[0]:
            # Check that buffer_size is the max size we can slice without incomplete reads
            buffer_size += 1
            with pytest.raises(tiledb.TileDBError):
                for offset in range(0, a.shape[0], buffer_size):
                    query[offset : offset + buffer_size]
