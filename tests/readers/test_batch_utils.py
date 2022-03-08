import numpy as np
import pytest

import tiledb
from tiledb.ml.readers._batch_utils import (
    Batch,
    Shuffling,
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
        start_offset=0,
        stop_offset=219,
    )
    assert list(batches) == [
        Batch(
            x_read_slice=slice(0, 40, None),
            y_read_slice=slice(0, 56, None),
            shuffling=Shuffling(
                size=40,
                x_buffer_slice=slice(0, 40, None),
                y_buffer_slice=slice(0, 40, None),
            ),
            x_buffer_slice=slice(0, 8, None),
            y_buffer_slice=slice(0, 8, None),
        ),
        Batch(
            x_read_slice=None,
            y_read_slice=None,
            shuffling=None,
            x_buffer_slice=slice(8, 16, None),
            y_buffer_slice=slice(8, 16, None),
        ),
        Batch(
            x_read_slice=None,
            y_read_slice=None,
            shuffling=None,
            x_buffer_slice=slice(16, 24, None),
            y_buffer_slice=slice(16, 24, None),
        ),
        Batch(
            x_read_slice=None,
            y_read_slice=None,
            shuffling=None,
            x_buffer_slice=slice(24, 32, None),
            y_buffer_slice=slice(24, 32, None),
        ),
        Batch(
            x_read_slice=None,
            y_read_slice=None,
            shuffling=None,
            x_buffer_slice=slice(32, 40, None),
            y_buffer_slice=slice(32, 40, None),
        ),
        Batch(
            x_read_slice=slice(40, 80, None),
            y_read_slice=None,
            shuffling=Shuffling(
                size=16,
                x_buffer_slice=slice(0, 16, None),
                y_buffer_slice=slice(40, 56, None),
            ),
            x_buffer_slice=slice(0, 8, None),
            y_buffer_slice=slice(40, 48, None),
        ),
        Batch(
            x_read_slice=None,
            y_read_slice=None,
            shuffling=None,
            x_buffer_slice=slice(8, 16, None),
            y_buffer_slice=slice(48, 56, None),
        ),
        Batch(
            x_read_slice=None,
            y_read_slice=slice(56, 112, None),
            shuffling=Shuffling(
                size=24,
                x_buffer_slice=slice(16, 40, None),
                y_buffer_slice=slice(0, 24, None),
            ),
            x_buffer_slice=slice(16, 24, None),
            y_buffer_slice=slice(0, 8, None),
        ),
        Batch(
            x_read_slice=None,
            y_read_slice=None,
            shuffling=None,
            x_buffer_slice=slice(24, 32, None),
            y_buffer_slice=slice(8, 16, None),
        ),
        Batch(
            x_read_slice=None,
            y_read_slice=None,
            shuffling=None,
            x_buffer_slice=slice(32, 40, None),
            y_buffer_slice=slice(16, 24, None),
        ),
        Batch(
            x_read_slice=slice(80, 120, None),
            y_read_slice=None,
            shuffling=Shuffling(
                size=32,
                x_buffer_slice=slice(0, 32, None),
                y_buffer_slice=slice(24, 56, None),
            ),
            x_buffer_slice=slice(0, 8, None),
            y_buffer_slice=slice(24, 32, None),
        ),
        Batch(
            x_read_slice=None,
            y_read_slice=None,
            shuffling=None,
            x_buffer_slice=slice(8, 16, None),
            y_buffer_slice=slice(32, 40, None),
        ),
        Batch(
            x_read_slice=None,
            y_read_slice=None,
            shuffling=None,
            x_buffer_slice=slice(16, 24, None),
            y_buffer_slice=slice(40, 48, None),
        ),
        Batch(
            x_read_slice=None,
            y_read_slice=None,
            shuffling=None,
            x_buffer_slice=slice(24, 32, None),
            y_buffer_slice=slice(48, 56, None),
        ),
        Batch(
            x_read_slice=None,
            y_read_slice=slice(112, 168, None),
            shuffling=Shuffling(
                size=8,
                x_buffer_slice=slice(32, 40, None),
                y_buffer_slice=slice(0, 8, None),
            ),
            x_buffer_slice=slice(32, 40, None),
            y_buffer_slice=slice(0, 8, None),
        ),
        Batch(
            x_read_slice=slice(120, 160, None),
            y_read_slice=None,
            shuffling=Shuffling(
                size=40,
                x_buffer_slice=slice(0, 40, None),
                y_buffer_slice=slice(8, 48, None),
            ),
            x_buffer_slice=slice(0, 8, None),
            y_buffer_slice=slice(8, 16, None),
        ),
        Batch(
            x_read_slice=None,
            y_read_slice=None,
            shuffling=None,
            x_buffer_slice=slice(8, 16, None),
            y_buffer_slice=slice(16, 24, None),
        ),
        Batch(
            x_read_slice=None,
            y_read_slice=None,
            shuffling=None,
            x_buffer_slice=slice(16, 24, None),
            y_buffer_slice=slice(24, 32, None),
        ),
        Batch(
            x_read_slice=None,
            y_read_slice=None,
            shuffling=None,
            x_buffer_slice=slice(24, 32, None),
            y_buffer_slice=slice(32, 40, None),
        ),
        Batch(
            x_read_slice=None,
            y_read_slice=None,
            shuffling=None,
            x_buffer_slice=slice(32, 40, None),
            y_buffer_slice=slice(40, 48, None),
        ),
        Batch(
            x_read_slice=slice(160, 200, None),
            y_read_slice=None,
            shuffling=Shuffling(
                size=8,
                x_buffer_slice=slice(0, 8, None),
                y_buffer_slice=slice(48, 56, None),
            ),
            x_buffer_slice=slice(0, 8, None),
            y_buffer_slice=slice(48, 56, None),
        ),
        Batch(
            x_read_slice=None,
            y_read_slice=slice(168, 219, None),
            shuffling=Shuffling(
                size=32,
                x_buffer_slice=slice(8, 40, None),
                y_buffer_slice=slice(0, 32, None),
            ),
            x_buffer_slice=slice(8, 16, None),
            y_buffer_slice=slice(0, 8, None),
        ),
        Batch(
            x_read_slice=None,
            y_read_slice=None,
            shuffling=None,
            x_buffer_slice=slice(16, 24, None),
            y_buffer_slice=slice(8, 16, None),
        ),
        Batch(
            x_read_slice=None,
            y_read_slice=None,
            shuffling=None,
            x_buffer_slice=slice(24, 32, None),
            y_buffer_slice=slice(16, 24, None),
        ),
        Batch(
            x_read_slice=None,
            y_read_slice=None,
            shuffling=None,
            x_buffer_slice=slice(32, 40, None),
            y_buffer_slice=slice(24, 32, None),
        ),
        Batch(
            x_read_slice=slice(200, 219, None),
            y_read_slice=None,
            shuffling=Shuffling(
                size=19,
                x_buffer_slice=slice(0, 19, None),
                y_buffer_slice=slice(32, 51, None),
            ),
            x_buffer_slice=slice(0, 8, None),
            y_buffer_slice=slice(32, 40, None),
        ),
        Batch(
            x_read_slice=None,
            y_read_slice=None,
            shuffling=None,
            x_buffer_slice=slice(8, 16, None),
            y_buffer_slice=slice(40, 48, None),
        ),
        Batch(
            x_read_slice=None,
            y_read_slice=None,
            shuffling=None,
            x_buffer_slice=slice(16, 19, None),
            y_buffer_slice=slice(48, 51, None),
        ),
    ]


def test_iter_batches_offsets():
    batches = iter_batches(
        batch_size=8,
        x_buffer_size=8 * 5,
        y_buffer_size=8 * 7,
        start_offset=73,
        stop_offset=146,
    )
    assert list(batches) == [
        Batch(
            x_read_slice=slice(73, 113, None),
            y_read_slice=slice(73, 129, None),
            shuffling=Shuffling(
                size=40,
                x_buffer_slice=slice(0, 40, None),
                y_buffer_slice=slice(0, 40, None),
            ),
            x_buffer_slice=slice(0, 8, None),
            y_buffer_slice=slice(0, 8, None),
        ),
        Batch(
            x_read_slice=None,
            y_read_slice=None,
            shuffling=None,
            x_buffer_slice=slice(8, 16, None),
            y_buffer_slice=slice(8, 16, None),
        ),
        Batch(
            x_read_slice=None,
            y_read_slice=None,
            shuffling=None,
            x_buffer_slice=slice(16, 24, None),
            y_buffer_slice=slice(16, 24, None),
        ),
        Batch(
            x_read_slice=None,
            y_read_slice=None,
            shuffling=None,
            x_buffer_slice=slice(24, 32, None),
            y_buffer_slice=slice(24, 32, None),
        ),
        Batch(
            x_read_slice=None,
            y_read_slice=None,
            shuffling=None,
            x_buffer_slice=slice(32, 40, None),
            y_buffer_slice=slice(32, 40, None),
        ),
        Batch(
            x_read_slice=slice(113, 146, None),
            y_read_slice=None,
            shuffling=Shuffling(
                size=16,
                x_buffer_slice=slice(0, 16, None),
                y_buffer_slice=slice(40, 56, None),
            ),
            x_buffer_slice=slice(0, 8, None),
            y_buffer_slice=slice(40, 48, None),
        ),
        Batch(
            x_read_slice=None,
            y_read_slice=None,
            shuffling=None,
            x_buffer_slice=slice(8, 16, None),
            y_buffer_slice=slice(48, 56, None),
        ),
        Batch(
            x_read_slice=None,
            y_read_slice=slice(129, 146, None),
            shuffling=Shuffling(
                size=17,
                x_buffer_slice=slice(16, 33, None),
                y_buffer_slice=slice(0, 17, None),
            ),
            x_buffer_slice=slice(16, 24, None),
            y_buffer_slice=slice(0, 8, None),
        ),
        Batch(
            x_read_slice=None,
            y_read_slice=None,
            shuffling=None,
            x_buffer_slice=slice(24, 32, None),
            y_buffer_slice=slice(8, 16, None),
        ),
        Batch(
            x_read_slice=None,
            y_read_slice=None,
            shuffling=None,
            x_buffer_slice=slice(32, 33, None),
            y_buffer_slice=slice(16, 17, None),
        ),
    ]
