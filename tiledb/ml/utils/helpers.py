import numpy as np
import tiledb


def get_schema(data: np.array, batch_size: int, sparse: bool) -> tiledb.ArraySchema:
    dims = [
        tiledb.Dim(
            name="dim_" + str(dim),
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
        attrs=[tiledb.Attr(name="features", dtype=np.float32)],
    )

    return schema


def ingest_in_tiledb(data: np.array, batch_size: int, uri: str, sparse: bool):
    schema = get_schema(data, batch_size, sparse)

    # Create the (empty) array on disk.
    tiledb.Array.create(uri, schema)

    # Ingest
    with tiledb.open(uri, "w") as tiledb_array:
        idx = np.nonzero(data) if sparse else slice(None)
        tiledb_array[idx] = {"features": data[idx]}
