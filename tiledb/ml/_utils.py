import numpy as np
import tiledb


def ingest_in_tiledb(
    data: np.array, batch_size: int, uri: str, sparse: bool, num_of_attributes: int
):
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
        attrs=[
            tiledb.Attr(name="features_" + str(attr), dtype=np.float32)
            for attr in range(num_of_attributes)
        ],
    )

    # Create the (empty) array on disk.
    tiledb.Array.create(uri, schema)

    # Ingest
    with tiledb.open(uri, "w") as tiledb_array:
        idx = np.nonzero(data) if sparse else slice(None)
        tiledb_array[idx] = {
            "features_" + str(attr): data[idx] for attr in range(num_of_attributes)
        }


def create_sparse_array_one_hot_2d(rows: int, cols: tuple) -> np.ndarray:
    seed = np.random.randint(low=0, high=cols[0], size=(rows,))
    seed[-1] = cols[0] - 1
    b = np.zeros((seed.size, seed.max() + 1))
    b[np.arange(seed.size), seed] = 1
    return b
