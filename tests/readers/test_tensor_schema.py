import itertools as it
import string

import numpy as np
import pytest

import tiledb
from tiledb.ml.readers.types import ArrayParams


@pytest.fixture(scope="module")
def dense_uri(tmp_path_factory):
    uri = str(tmp_path_factory.mktemp("dense"))
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


@pytest.fixture(scope="module")
def sparse_uri(tmp_path_factory, non_zero_per_row=3):
    uri = str(tmp_path_factory.mktemp("sparse"))
    domains = [
        (1, 1000),
        (-250, 249),
        (0, 1),
        (np.datetime64("2020-01-01"), np.datetime64("2022-01-01")),
        ("a", "z"),
    ]
    schema = tiledb.ArraySchema(
        sparse=True,
        allows_duplicates=True,
        domain=tiledb.Domain(
            tiledb.Dim(name="d0", domain=domains[0], dtype=np.uint32),
            tiledb.Dim(name="d1", domain=domains[1], dtype=np.int16),
            tiledb.Dim(name="d2", domain=domains[2], dtype=np.float64),
            tiledb.Dim(name="d3", domain=domains[3], dtype=np.dtype("datetime64[D]")),
            tiledb.Dim(name="d4", domain=domains[4], dtype="ascii"),
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
        d2 = np.random.uniform(domains[2][0], domains[2][1], num_cells)
        d3_days = (domains[3][1] - domains[3][0]).astype(int)
        d3 = domains[3][0] + np.random.randint(0, d3_days, num_cells)
        # TODO: increase 8 to something larger after sh-19349 is fixed
        d4 = np.random.choice([c * 8 for c in string.ascii_lowercase], num_cells)
        a[d0, d1, d2, d3, d4] = {
            "af8": np.random.rand(num_cells),
            "af4": np.random.rand(num_cells).astype(np.float32),
            "au1": np.random.randint(128, size=num_cells, dtype=np.uint8),
        }
    return uri


def parametrize_fields(*fields):
    return pytest.mark.parametrize(
        "fields",
        list(it.chain.from_iterable(it.combinations(fields, i) for i in range(4))),
    )


@pytest.mark.parametrize(
    "key_dim_index,memory_budget",
    [(0, 16_000), (0, 32_000), (0, 64_000), (1, 500_000), (1, 600_000), (1, 700_000)],
)
@parametrize_fields("d0", "d1", "d2", "af8", "af4", "au1")
def test_max_partition_weight_dense(dense_uri, fields, key_dim_index, memory_budget):
    config = {"py.max_incomplete_retries": 0, "sm.memory_budget": memory_budget}
    with tiledb.open(dense_uri, config=config) as a:
        schema = ArrayParams(a, key_dim_index, fields).to_tensor_schema()
        max_weight = schema.max_partition_weight
        for key_range in schema.key_range.partition_by_weight(max_weight):
            # query succeeds without incomplete retries
            schema.query[key_range.min : key_range.max]

            if key_range.max < schema.key_range.max:
                # querying a larger slice should fail
                with pytest.raises(tiledb.TileDBError) as ex:
                    schema.query[key_range.min : key_range.max + 1]
                assert "py.max_incomplete_retries" in str(ex.value)


@pytest.mark.parametrize("memory_budget", [2048, 4096])
@pytest.mark.parametrize("key_dim_index", [0, 1, 3, 4])
@parametrize_fields("d0", "d1", "d2", "d3", "d4", "af8", "af4", "au1")
def test_max_partition_weight_sparse(sparse_uri, fields, key_dim_index, memory_budget):
    config = {
        "py.max_incomplete_retries": 0,
        "py.init_buffer_bytes": memory_budget,
    }
    with tiledb.open(sparse_uri, config=config) as a:
        key_dim = a.dim(key_dim_index)
        schema = ArrayParams(a, key_dim_index, fields).to_tensor_schema()
        max_weight = schema.max_partition_weight
        for key_range in schema.key_range.partition_by_weight(max_weight):
            # query succeeds without incomplete retries
            schema.query[key_range.min : key_range.max]

            if key_range.max < schema.key_range.max:
                # querying a larger slice should fail. `key_range.max + 1` makes sense
                # only for integer key dim so test just integer key dims
                if np.issubdtype(key_dim.dtype, np.integer):
                    with pytest.raises(tiledb.TileDBError) as ex:
                        schema.query[key_range.min : key_range.max + 1]
                    assert "py.max_incomplete_retries" in str(ex.value)
