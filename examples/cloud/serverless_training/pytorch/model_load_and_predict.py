import os
from typing import List

import tiledb.client

# Your TileDB username and password, exported as environmental variables
TILEDB_USER_NAME = os.environ.get("TILEDB_USER_NAME")
TILEDB_PASSWD = os.environ.get("TILEDB_PASSWD")

# Your TileDB namespace
TILEDB_WORKSPACE = "your_tiledb_WORKSPACE"
TILEDB_TEAMSPACE = "your_tiledb_TEAMSPACE"

# Your S3 bucket
S3_BUCKET = "your_s3_bucket"

IMAGES_URI = f"tiledb://{TILEDB_WORKSPACE}/{TILEDB_TEAMSPACE}/s3://{S3_BUCKET}/mnist_images"
LABELS_URI = f"tiledb://{TILEDB_WORKSPACE}/{TILEDB_TEAMSPACE}/s3://{S3_BUCKET}/mnist_labels"
MODEL_URI = f"tiledb://{TILEDB_WORKSPACE}/{TILEDB_TEAMSPACE}/s3://{S3_BUCKET}/mnist_model"

IO_BATCH_SIZE = 20000


def predict() -> List[int]:
    from typing import Any, Tuple

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim

    from tiledb.ml.models.pytorch import PyTorchTileDBModel
    from tiledb.ml.readers.pytorch import PyTorchTileDBDataLoader

    class Net(nn.Module):
        def __init__(self, shape: Tuple[int, int]):
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(np.product(shape), 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 10),
                nn.ReLU(),
            )

        def forward(self, x: torch.Tensor) -> Any:
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits

    # Load our model from a TileDB array
    loaded_model = Net(shape=(28, 28))
    loaded_optimizer = optim.SGD(loaded_model.parameters(), lr=0.01, momentum=0.5)

    PyTorchTileDBModel(uri=MODEL_URI).load(
        model=loaded_model, optimizer=loaded_optimizer
    )

    with tiledb.open(IMAGES_URI) as x, tiledb.open(LABELS_URI) as y:
        with torch.no_grad():
            data_loader = PyTorchTileDBDataLoader(x, y, batch_size=IO_BATCH_SIZE)
            inputs, labels = next(iter(data_loader))
            output = loaded_model(
                inputs[np.random.randint(0, 10) : np.random.randint(11, 50)].to(
                    torch.float
                )
            )

    return [np.argmax(pred) for pred in output.numpy()]


tiledb.client.configure(username=TILEDB_USER_NAME, password=TILEDB_PASSWD, workspace=TILEDB_WORKSPACE)
tiledb.client.login()

predictions = tiledb.client.udf.exec(predict)

print(predictions)
