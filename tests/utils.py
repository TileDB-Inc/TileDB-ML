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


def create_rand_labels(
    num_rows: int, num_classes: int, one_hot: bool = False
) -> np.ndarray:
    """Create labels for `num_rows` observations with `num_classes` classes.

    :param num_rows: Number of observations to create labels for.
    :param num_classes: Number of possible labels.
    :param one_hot: Whether to create one-hot labels.

    :returns:
    - If one-hot is False, 1-D numpy array of length `num_rows` with labels from 0 to
      `num_classes`
    - If one-hot is True, binary 2-D numpy array of shape `(num_rows, num_classes)`.
    """
    labels = np.random.randint(num_classes, size=num_rows)
    return np.eye(num_classes, dtype=np.uint8)[labels] if one_hot else labels
