import os
from typing import Any

import numpy as np
import tiledb.cloud

# Your TileDB username and password, exported as environmental variables
TILEDB_USER_NAME = os.environ.get("TILEDB_USER_NAME")
TILEDB_PASSWD = os.environ.get("TILEDB_PASSWD")

# Your TileDB namespace
TILEDB_NAMESPACE = "your_tiledb_namespace"

# Your S3 bucket
S3_BUCKET = "your_s3_bucket"

IMAGES_URI = "tiledb://{}/s3://{}/mnist_images".format(TILEDB_NAMESPACE, S3_BUCKET)
LABELS_URI = "tiledb://{}/s3://{}/mnist_labels".format(TILEDB_NAMESPACE, S3_BUCKET)


# Let's define an ingestion function
def ingest_in_tiledb(data: np.array, batch_size: int, uri: str) -> None:
    import tiledb

    # Equal number of dimensions with the numpy array.
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
        sparse=False,
        attrs=[tiledb.Attr(name="features", dtype=data.dtype)],
    )
    # Create array
    tiledb.Array.create(uri, schema)

    # Ingest
    with tiledb.open(uri, "w") as tiledb_array:
        tiledb_array[:] = {"features": data}


def mnist_ingest(ingestion_func: Any) -> None:
    import torchvision

    train_data = torchvision.datasets.MNIST("./data", train=True, download=True)

    images = train_data.data.numpy() / 255.0
    labels = train_data.targets.numpy()

    # Ingest images
    ingestion_func(data=images, batch_size=64, uri=IMAGES_URI)

    # Ingest labels
    ingestion_func(data=labels, batch_size=64, uri=LABELS_URI)


tiledb.cloud.login(username=TILEDB_USER_NAME, password=TILEDB_PASSWD)

tiledb.cloud.udf.exec(mnist_ingest, ingestion_func=ingest_in_tiledb)

print(tiledb.cloud.last_udf_task().logs)
