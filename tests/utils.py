import os
import uuid

import numpy as np

import tiledb


def ingest_in_tiledb(tmpdir, data_x, data_y, sparse_x, sparse_y, batch_size, num_attrs):
    array_uuid = str(uuid.uuid4())
    uri_x = os.path.join(tmpdir, "x_" + array_uuid)
    uri_y = os.path.join(tmpdir, "y_" + array_uuid)
    _ingest_in_tiledb(uri_x, data_x, sparse_x, batch_size, num_attrs)
    _ingest_in_tiledb(uri_y, data_y, sparse_y, batch_size, num_attrs)
    return uri_x, uri_y


def _ingest_in_tiledb(
    uri: str, data: np.ndarray, sparse: bool, batch_size: int, num_attrs: int
) -> None:
    dims = [
        tiledb.Dim(
            name=f"dim_{dim}",
            domain=(0, data.shape[dim] - 1),
            tile=data.shape[dim] if dim > 0 else batch_size,
            dtype=np.int32,
        )
        for dim in range(data.ndim)
    ]

    # TileDB schema
    schema = tiledb.ArraySchema(
        domain=tiledb.Domain(*dims),
        sparse=sparse,
        attrs=[
            tiledb.Attr(name=f"features_{attr}", dtype=np.float32)
            for attr in range(num_attrs)
        ],
    )

    # Create the (empty) array on disk.
    tiledb.Array.create(uri, schema)

    # Ingest
    with tiledb.open(uri, "w") as tiledb_array:
        idx = np.nonzero(data) if sparse else slice(None)
        tiledb_array[idx] = {f"features_{attr}": data[idx] for attr in range(num_attrs)}


def create_sparse_array_one_hot_2d(rows: int, cols: int) -> np.ndarray:
    seed = np.random.randint(low=0, high=cols, size=rows)
    seed[-1] = cols - 1
    b = np.zeros((seed.size, seed.max() + 1))
    b[np.arange(seed.size), seed] = 1
    return b
