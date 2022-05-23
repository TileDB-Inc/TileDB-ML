import numpy as np
import pytest

import tiledb
from tiledb.ml.readers._tensor_schema import TensorSchema, iter_slices


@pytest.fixture
def dense_uri(tmp_path):
    uri = str(tmp_path / "dense")
    schema = tiledb.ArraySchema(
        sparse=False,
        domain=tiledb.Domain(
            tiledb.Dim(name="d0", domain=(0, 9999), dtype=np.int32, tile=123),
            tiledb.Dim(name="d1", domain=(-2, 2), dtype=np.int32, tile=2),
            tiledb.Dim(name="d2", domain=(1, 2), dtype=np.int32, tile=1),
        ),
        attrs=[
            tiledb.Attr(name="af8", dtype=np.float64),
            tiledb.Attr(name="af4", dtype=np.float32),
            tiledb.Attr(name="au1", dtype=np.uint8),
        ],
    )
    tiledb.Array.create(uri, schema)
    with tiledb.open(uri, "w") as a:
        a[:] = {
            "af8": np.random.rand(*schema.shape),
            "af4": np.random.rand(*schema.shape).astype(np.float32),
            "au1": np.random.randint(128, size=schema.shape, dtype=np.uint8),
        }
    return uri


@pytest.fixture
def sparse_uri(tmp_path, non_zero_per_row=3):
    uri = str(tmp_path / "sparse")
    domains = [(1, 1000), (-250, 249), (0, 1)]
    schema = tiledb.ArraySchema(
        sparse=True,
        allows_duplicates=True,
        domain=tiledb.Domain(
            tiledb.Dim(name="d0", domain=domains[0], dtype=np.uint32),
            tiledb.Dim(name="d1", domain=domains[1], dtype=np.int16),
            tiledb.Dim(name="d2", domain=domains[2], dtype=np.float64),
        ),
        attrs=[
            tiledb.Attr(name="af8", dtype=np.float64),
            tiledb.Attr(name="af4", dtype=np.float32),
            tiledb.Attr(name="au1", dtype=np.uint8),
        ],
    )
    tiledb.Array.create(uri, schema)
    with tiledb.open(uri, "w") as a:
        d0 = np.concatenate(
            [np.arange(domains[0][0], domains[0][1] + 1)] * non_zero_per_row
        )
        num_cells = len(d0)
        d1 = np.random.randint(domains[1][0], domains[1][1] + 1, num_cells)
        d2 = np.random.randint(domains[2][0], domains[2][1] + 1, num_cells)
        a[d0, d1, d2] = {
            "af8": np.random.rand(num_cells),
            "af4": np.random.rand(num_cells).astype(np.float32),
            "au1": np.random.randint(128, size=num_cells, dtype=np.uint8),
        }
    return uri


parametrize_attrs = pytest.mark.parametrize(
    "attrs",
    [(), ("af8",), ("af4",), ("au1",), ("af8", "af4"), ("af8", "au1"), ("af4", "au1")],
)


@pytest.mark.parametrize(
    "key_dim_index,memory_budget",
    [(0, 16_000), (0, 32_000), (0, 64_000), (1, 500_000), (1, 600_000), (1, 700_000)],
)
@parametrize_attrs
def test_get_max_buffer_size_dense(dense_uri, attrs, key_dim_index, memory_budget):
    config = {"py.max_incomplete_retries": 0, "sm.memory_budget": memory_budget}
    with tiledb.open(dense_uri, config=config) as a:
        schema = TensorSchema(a, key_dim_index, attrs)
        buffer_size = schema._get_max_buffer_size_dense()
        query = a.query(attrs=schema.attrs)
        for key_slice in iter_slices(schema.start_key, schema.stop_key, buffer_size):
            # query succeeds without incomplete retries
            query.multi_index[schema[key_slice.start : key_slice.stop - 1]]

            if key_slice.stop < schema.stop_key:
                # querying a larger slice than buffer_size should fail
                with pytest.raises(tiledb.TileDBError) as ex:
                    query.multi_index[schema[key_slice]]
                assert "py.max_incomplete_retries" in str(ex.value)


@pytest.mark.parametrize(
    "key_dim_index,memory_budget",
    [(0, 1024), (0, 2048), (0, 4096), (1, 1024), (1, 2048), (1, 4096)],
)
@parametrize_attrs
def test_get_max_buffer_size_sparse(sparse_uri, attrs, key_dim_index, memory_budget):
    # The first dimension has a fixed number of non-zero cells per "row". The others
    # don't, so the estimated buffer_size is not necessarily the maximum number of
    # "rows" than can fit in the given memory budget. In this case relax the test by
    # allowing 1 incomplete retry
    exact = key_dim_index == 0
    config = {
        "py.max_incomplete_retries": 0 if exact else 1,
        "py.init_buffer_bytes": memory_budget,
    }
    with tiledb.open(sparse_uri, config=config) as a:
        schema = TensorSchema(a, key_dim_index, attrs)
        buffer_size = schema._get_max_buffer_size_sparse()
        query = a.query(attrs=schema.attrs)
        for key_slice in iter_slices(schema.start_key, schema.stop_key, buffer_size):
            # query succeeds without incomplete retries (or at most 1 retry if not exact)
            query.multi_index[schema[key_slice.start : key_slice.stop - 1]]

            if exact and key_slice.stop < schema.stop_key:
                # querying a larger slice than buffer_size should fail
                with pytest.raises(tiledb.TileDBError) as ex:
                    query.multi_index[schema[key_slice]]
                assert "py.max_incomplete_retries" in str(ex.value)
