import os
from typing import Any

import numpy as np
import tiledb.client

# Your TileDB username and password, exported as environmental variables
TILEDB_USER_NAME = os.environ.get("TILEDB_USER_NAME")
TILEDB_PASSWD = os.environ.get("TILEDB_PASSWD")

# Your TileDB workspace/teamspace
TILEDB_WORKSPACE = "your_tiledb_WORKSPACE"
TILEDB_TEAMSPACE = "your_tiledb_TEAMSPACE"

# Your S3 bucket
S3_BUCKET = "your_s3_bucket"

IMAGES_URI = f"tiledb://{TILEDB_WORKSPACE}/{TILEDB_TEAMSPACE}/s3://{S3_BUCKET}/mnist_images"
LABELS_URI = f"tiledb://{TILEDB_WORKSPACE}/{TILEDB_TEAMSPACE}/s3://{S3_BUCKET}/mnist_labels"


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


tiledb.client.configure(username=TILEDB_USER_NAME, password=TILEDB_PASSWD, workspace=TILEDB_WORKSPACE)
tiledb.client.login()

tiledb.client.udf.exec(mnist_ingest, ingestion_func=ingest_in_tiledb)

print(tiledb.client.last_udf_task().logs)
