"""Tests for TileDB PyTorch model save and load."""

import pytest
import inspect
import sys
import os
import platform

import torch
import torch.nn as nn
import torch.optim as optimizers
import torch.nn.functional as F

from tiledb.ml.models.pytorch import PyTorchTileDBModel


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class SeqNeuralNetwork(nn.Module):
    def __init__(self):
        super(SeqNeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(14 * 14, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


@pytest.mark.parametrize(
    "optimizer",
    [
        getattr(optimizers, name)
        for name, obj in inspect.getmembers(optimizers)
        if inspect.isclass(obj) and name != "Optimizer"
    ],
)
@pytest.mark.parametrize(
    "net",
    [
        getattr(sys.modules[__name__], name)
        for name, obj in inspect.getmembers(sys.modules[__name__])
        if inspect.isclass(obj) and obj.__module__ == __name__
    ],
)
class TestPyTorchModel:
    def test_save(self, tmpdir, net, optimizer):
        EPOCH = 5
        LOSS = 0.4
        saved_net = net()
        saved_optimizer = optimizer(saved_net.parameters(), lr=0.001)
        tiledb_array = os.path.join(tmpdir, "model_array")
        tiledb_obj = PyTorchTileDBModel(
            uri=tiledb_array, model=saved_net, optimizer=saved_optimizer
        )

        tiledb_obj.save(
            model_info={
                "epoch": EPOCH,
                "loss": LOSS,
            }
        )
        loaded_net = net()
        loaded_optimizer = optimizer(loaded_net.parameters(), lr=0.001)
        model_out_dict = tiledb_obj.load(model=loaded_net, optimizer=loaded_optimizer)

        assert model_out_dict["epoch"] == EPOCH
        assert model_out_dict["loss"] == LOSS

        # Check model parameters
        for key_item_1, key_item_2 in zip(
            saved_net.state_dict().items(), loaded_net.state_dict().items()
        ):
            assert torch.equal(key_item_1[1], key_item_2[1])

        # Check optimizer parameters
        for key_item_1, key_item_2 in zip(
            saved_optimizer.state_dict().items(), loaded_optimizer.state_dict().items()
        ):
            assert all([a == b for a, b in zip(key_item_1[1], key_item_2[1])])

    def test_preview(self, tmpdir, net, optimizer):
        # With model given as argument
        saved_net = net()
        tiledb_array = os.path.join(tmpdir, "model_array")
        tiledb_obj = PyTorchTileDBModel(uri=tiledb_array, model=saved_net)
        assert type(tiledb_obj.preview()) == str
        tiledb_obj_none = PyTorchTileDBModel(uri=tiledb_array, model=None)
        assert tiledb_obj_none.preview() == ""

    def test_get_cloud_uri(self, tmpdir, net, optimizer, mocker):
        saved_net = net()
        tiledb_array = os.path.join(tmpdir, "model_array")
        tiledb_obj = PyTorchTileDBModel(uri=tiledb_array, model=saved_net)

        mocker.patch("tiledb.ml._cloud_utils.get_s3_prefix", return_value=None)
        with pytest.raises(ValueError):
            tiledb_obj.get_cloud_uri(tiledb_array)

        mocker.patch("tiledb.ml._cloud_utils.get_s3_prefix", return_value="bar")
        actual = tiledb_obj.get_cloud_uri(tiledb_array)
        expected = "tiledb://{}/{}".format(
            tiledb_obj.namespace, os.path.join("bar", tiledb_array)
        )
        assert actual == expected

    def test_file_properties(self, tmpdir, net, optimizer):
        saved_net = net()
        tiledb_array = os.path.join(tmpdir, "model_array")
        tiledb_obj = PyTorchTileDBModel(uri=tiledb_array, model=saved_net)
        tiledb_obj.save()

        assert tiledb_obj._file_properties["TILEDB_ML_MODEL_ML_FRAMEWORK"] == "PYTORCH"
        assert tiledb_obj._file_properties["TILEDB_ML_MODEL_STAGE"] == "STAGING"
        assert (
            tiledb_obj._file_properties["TILEDB_ML_MODEL_PYTHON_VERSION"]
            == platform.python_version()
        )
        assert (
            tiledb_obj._file_properties["TILEDB_ML_MODEL_ML_FRAMEWORK_VERSION"]
            == torch.__version__
        )
        assert tiledb_obj._file_properties["TILEDB_ML_MODEL_PREVIEW"] == str(saved_net)

    def test_file_properties_in_tiledb_cloud_case(self, tmpdir, net, optimizer, mocker):
        saved_net = net()
        tiledb_array = os.path.join(tmpdir, "model_array")

        mocker.patch(
            "tiledb.ml.models.base.TileDBModel.get_cloud_uri", return_value=tiledb_array
        )
        mocker.patch("tiledb.ml._cloud_utils.update_file_properties")

        tiledb_obj = PyTorchTileDBModel(
            uri=tiledb_array, namespace="test_namespace", model=saved_net
        )
        tiledb_obj.save()

        assert tiledb_obj._file_properties["TILEDB_ML_MODEL_ML_FRAMEWORK"] == "PYTORCH"
        assert tiledb_obj._file_properties["TILEDB_ML_MODEL_STAGE"] == "STAGING"
        assert (
            tiledb_obj._file_properties["TILEDB_ML_MODEL_PYTHON_VERSION"]
            == platform.python_version()
        )
        assert (
            tiledb_obj._file_properties["TILEDB_ML_MODEL_ML_FRAMEWORK_VERSION"]
            == torch.__version__
        )
        assert tiledb_obj._file_properties["TILEDB_ML_MODEL_PREVIEW"] == str(saved_net)

    def test_exception_raise_file_property_in_meta_error(self, tmpdir, net, optimizer):
        saved_net = net()
        tiledb_array = os.path.join(tmpdir, "model_array")
        tiledb_obj = PyTorchTileDBModel(uri=tiledb_array, model=saved_net)
        with pytest.raises(ValueError):
            tiledb_obj.save(
                meta={"TILEDB_ML_MODEL_ML_FRAMEWORK": "TILEDB_ML_MODEL_ML_FRAMEWORK"},
            )
