import numpy as np
import pytest

import tiledb
from tiledb.ml.readers._batch_utils import (
    estimate_row_bytes,
    get_num_batches,
    iter_batches,
)


@pytest.fixture
def dense_uri(tmp_path):
    uri = str(tmp_path / "dense")
    schema = tiledb.ArraySchema(
        sparse=False,
        domain=tiledb.Domain(
            tiledb.Dim(name="d0", domain=(0, 999), dtype=np.uint32),
            tiledb.Dim(name="d1", domain=(1, 5), dtype=np.uint32),
            tiledb.Dim(name="d2", domain=(1, 2), dtype=np.uint32),
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


def test_get_num_batches_dense(dense_uri):
    batch_size = 16
    buffer_bytes = 50000
    with tiledb.open(dense_uri) as a:
        # int(50000 / 16 / 130) == 24
        assert get_num_batches(batch_size, buffer_bytes, a) == 24
        # int(50000 / 16 / 90) == 34
        assert get_num_batches(batch_size, buffer_bytes, a, attrs=["af8", "au1"]) == 34
        # int(50000 / 16 / 40) == 78 but there are at most ceil(1000 / 16) == 63 batches
        assert get_num_batches(batch_size, buffer_bytes, a, attrs=["af4"]) == 63


def test_estimate_row_bytes_sparse(sparse_uri):
    with tiledb.open(sparse_uri) as a:
        # 3 cells/row, 3*4 bytes for dims + 8+4+1=13 bytes for attrs = 25 bytes/cell
        assert estimate_row_bytes(a) == 75
        # 3 cells/row, 3*4 bytes for dims + 8+1=9 bytes for attrs = 21 bytes/cell
        assert estimate_row_bytes(a, attrs=["af8", "au1"]) == 63
        # 3 cells/row, 3*4 bytes for dims + 4 bytes for attrs = 16 bytes/cell
        assert estimate_row_bytes(a, attrs=["af4"]) == 48


def test_get_num_batches_sparse(sparse_uri):
    batch_size = 16
    buffer_bytes = 50000
    with tiledb.open(sparse_uri) as a:
        # int(50000 / 16 / 75) == 41
        assert get_num_batches(batch_size, buffer_bytes, a) == 41
        # int(50000 / 16 / 63) == 49
        assert get_num_batches(batch_size, buffer_bytes, a, attrs=["af8", "au1"]) == 49
        # int(50000 / 16 / 48) == 65 but there are at most ceil(1000 / 16) == 63 batches
        assert get_num_batches(batch_size, buffer_bytes, a, attrs=["af4"]) == 63


def test_iter_batches():
    batches = iter_batches(
        batch_size=8,
        x_buffer_size=8 * 5,
        y_buffer_size=8 * 7,
        start_offset=104,
        stop_offset=209,
    )
    assert list(map(str, batches)) == [
        "Batch(x[0:8], y[0:8], x_read[104:144], y_read[104:160], Shuffling(x[0:40], y[0:40]))",
        "Batch(x[8:16], y[8:16])",
        "Batch(x[16:24], y[16:24])",
        "Batch(x[24:32], y[24:32])",
        "Batch(x[32:40], y[32:40])",
        "Batch(x[0:8], y[40:48], x_read[144:184], Shuffling(x[0:16], y[40:56]))",
        "Batch(x[8:16], y[48:56])",
        "Batch(x[16:24], y[0:8], y_read[160:209], Shuffling(x[16:40], y[0:24]))",
        "Batch(x[24:32], y[8:16])",
        "Batch(x[32:40], y[16:24])",
        "Batch(x[0:8], y[24:32], x_read[184:209], Shuffling(x[0:25], y[24:49]))",
        "Batch(x[8:16], y[32:40])",
        "Batch(x[16:24], y[40:48])",
        "Batch(x[24:25], y[48:49])",
    ]
