import numpy as np
import pytest

import tiledb
from tiledb.ml.readers._batch_utils import (
    estimate_row_bytes,
    get_max_buffer_size,
    iter_batches,
)


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


def test_iter_batches():
    batches = iter_batches(
        x_buffer_size=17, y_buffer_size=41, start_offset=104, stop_offset=213
    )
    assert list(map(str, batches)) == [
        "Batch(17, x[0:17], y[0:17], x_read[104:121], y_read[104:145])",
        "Batch(17, x[0:17], y[17:34], x_read[121:138])",
        "Batch(7, x[0:7], y[34:41], x_read[138:155])",
        "Batch(10, x[7:17], y[0:10], y_read[145:186])",
        "Batch(17, x[0:17], y[10:27], x_read[155:172])",
        "Batch(14, x[0:14], y[27:41], x_read[172:189])",
        "Batch(3, x[14:17], y[0:3], y_read[186:213])",
        "Batch(17, x[0:17], y[3:20], x_read[189:206])",
        "Batch(7, x[0:7], y[20:27], x_read[206:213])",
    ]
    batches = iter_batches(
        x_buffer_size=18, y_buffer_size=27, start_offset=104, stop_offset=213
    )
    assert list(map(str, batches)) == [
        "Batch(18, x[0:18], y[0:18], x_read[104:122], y_read[104:131])",
        "Batch(9, x[0:9], y[18:27], x_read[122:140])",
        "Batch(9, x[9:18], y[0:9], y_read[131:158])",
        "Batch(18, x[0:18], y[9:27], x_read[140:158])",
        "Batch(18, x[0:18], y[0:18], x_read[158:176], y_read[158:185])",
        "Batch(9, x[0:9], y[18:27], x_read[176:194])",
        "Batch(9, x[9:18], y[0:9], y_read[185:212])",
        "Batch(18, x[0:18], y[9:27], x_read[194:212])",
        "Batch(1, x[0:1], y[0:1], x_read[212:213], y_read[212:213])",
    ]


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
