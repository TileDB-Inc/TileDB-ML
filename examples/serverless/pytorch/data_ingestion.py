import os
from typing import Any

import tiledb.cloud

from tests.utils import ingest_in_tiledb

# Your TileDB username and password, exported as environmental variables
TILEDB_USER_NAME = os.environ.get("TILEDB_USER_NAME")
TILEDB_PASSWD = os.environ.get("TILEDB_PASSWD")

# Your TileDB namespace
TILEDB_NAMESPACE = "your_tiledb_namespace"

# Your S3 bucket
S3_BUCKET = "your_s3_bucket"

IMAGES_URI = "tiledb://{}/s3://{}/mnist_images".format(TILEDB_NAMESPACE, S3_BUCKET)
LABELS_URI = "tiledb://{}/s3://{}/mnist_labels".format(TILEDB_NAMESPACE, S3_BUCKET)


def mnist_ingest(ingestion_func: Any) -> None:
    import torchvision

    train_data = torchvision.datasets.MNIST("./data", train=True, download=True)

    images = train_data.data.numpy() / 255.0
    labels = train_data.targets.numpy()

    # Ingest images
    ingestion_func(
        data=images, batch_size=64, uri=IMAGES_URI, sparse=False, num_of_attributes=1
    )

    # Ingest labels
    ingestion_func(
        data=labels, batch_size=64, uri=LABELS_URI, sparse=False, num_of_attributes=1
    )


tiledb.cloud.login(username=TILEDB_USER_NAME, password=TILEDB_PASSWD)

tiledb.cloud.udf.exec(mnist_ingest, ingestion_func=ingest_in_tiledb)

print(tiledb.cloud.last_udf_task().logs)
