import os

import tiledb.client
from tiledb.ml.readers.types import ArrayParams

# Your TileDB username and password, exported as environmental variables
TILEDB_USER_NAME = os.environ.get("TILEDB_USER_NAME")
TILEDB_PASSWD = os.environ.get("TILEDB_PASSWD")

# Your TileDB namespace
TILEDB_WORKSPACE = "your_tiledb_WORKSPACE"
TILEDB_TEAMSPACE = "your_tiledb_TEAMSPACE"

# Your S3 bucket
S3_BUCKET = "your_s3_bucket"

IMAGES_URI = f"tiledb://{TILEDB_WORKSPACE}/{TILEDB_TEAMSPACE}/mnist_images"
LABELS_URI = f"tiledb://{TILEDB_WORKSPACE}/{TILEDB_TEAMSPACE}/mnist_labels"
MODEL_URI = f"tiledb://{TILEDB_WORKSPACE}/{TILEDB_TEAMSPACE}/mnist_model"

# The size of each slice from a image and label TileDB arrays.
IO_BATCH_SIZE = 20000


def train() -> None:
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

    def do_random_noise(img, mag=0.1):
        noise = np.random.uniform(-1, 1,img.shape)*mag
        img = img + noise
        img = np.clip(img,0,1)
        return img

    with tiledb.open(IMAGES_URI) as x, tiledb.open(LABELS_URI) as y:
        train_loader = PyTorchTileDBDataLoader(
            ArrayParams(x, fn=do_random_noise),
            ArrayParams(y),
            batch_size=IO_BATCH_SIZE,
            num_workers=0,
            shuffle_buffer_size=256,
        )

        net = Net(shape=(28, 28))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)

        training_batch_size = 200

        for epoch in range(1, 3):
            net.train()
            training_batch_idx = 0
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                for train_batch_start_idx in range(
                    0, IO_BATCH_SIZE, training_batch_size
                ):
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward + backward + optimize
                    outputs = net(
                        inputs[
                            train_batch_start_idx : train_batch_start_idx
                            + training_batch_size
                        ].to(torch.float)
                    )
                    loss = criterion(
                        outputs,
                        labels[
                            train_batch_start_idx : train_batch_start_idx
                            + training_batch_size
                        ]
                        .to(torch.float)
                        .type(torch.LongTensor),
                    )
                    loss.backward()
                    optimizer.step()
                    if training_batch_idx % 100 == 0:
                        print(
                            "Train Epoch: {} Batch: {} Loss: {:.6f}".format(
                                epoch, training_batch_idx, loss.item()
                            )
                        )
                    training_batch_idx += 1

        model = PyTorchTileDBModel(
            uri="mnist_model",
            teamspace=TILEDB_TEAMSPACE,
            model=net,
            optimizer=optimizer,
        )

        # Save model as TileDB array.
        model.save()


tiledb.client.configure(username=TILEDB_USER_NAME, password=TILEDB_PASSWD, workspace=TILEDB_WORKSPACE)
tiledb.client.login()

tiledb.client.udf.exec(train)

print(tiledb.client.last_udf_task().logs)
