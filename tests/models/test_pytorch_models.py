"""Tests for TileDB PyTorch model save and load."""

import glob
import inspect
import os
import pickle
import platform
import shutil

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizers
from torch.utils.tensorboard import SummaryWriter

import tiledb
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


net = pytest.mark.parametrize("net", [ConvNet, Net, SeqNeuralNetwork])


class TestPyTorchModel:
    @net
    @pytest.mark.parametrize(
        "optimizer",
        [
            getattr(optimizers, name)
            for name, obj in inspect.getmembers(optimizers)
            if inspect.isclass(obj) and name != "Optimizer"
        ],
    )
    def test_save(self, tmpdir, net, optimizer):
        EPOCH = 5
        LOSS = 0.4
        model = net()
        saved_optimizer = optimizer(model.parameters(), lr=0.001)
        tiledb_array = os.path.join(tmpdir, "model_array")
        tiledb_obj = PyTorchTileDBModel(
            uri=tiledb_array, model=model, optimizer=saved_optimizer
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
            model.state_dict().items(), loaded_net.state_dict().items()
        ):
            assert torch.equal(key_item_1[1], key_item_2[1])

        # Check optimizer parameters
        for key_item_1, key_item_2 in zip(
            saved_optimizer.state_dict().items(), loaded_optimizer.state_dict().items()
        ):
            assert all([a == b for a, b in zip(key_item_1[1], key_item_2[1])])

    @net
    def test_preview(self, tmpdir, net):
        # With model given as argument
        model = net()
        tiledb_array = os.path.join(tmpdir, "model_array")
        tiledb_obj = PyTorchTileDBModel(uri=tiledb_array, model=model)
        assert type(tiledb_obj.preview()) == str
        tiledb_obj_none = PyTorchTileDBModel(uri=tiledb_array, model=None)
        assert tiledb_obj_none.preview() == ""

    @net
    def test_file_properties(self, tmpdir, net):
        model = net()
        tiledb_array = os.path.join(tmpdir, "model_array")
        tiledb_obj = PyTorchTileDBModel(uri=tiledb_array, model=model)

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
        assert tiledb_obj._file_properties["TILEDB_ML_MODEL_PREVIEW"] == str(model)

    @net
    def test_tensorboard_callback_meta(self, tmpdir, net):
        model = net()
        tiledb_array = os.path.join(tmpdir, "model_array")
        tiledb_obj = PyTorchTileDBModel(uri=tiledb_array, model=model)

        # SummaryWriter creates file(s) under log_dir
        log_dir = os.path.join(tmpdir, "logs")
        writer = SummaryWriter(log_dir=log_dir)
        log_files = read_files(log_dir)
        assert log_files

        tiledb_obj.save(update=False, summary_writer=writer)
        with tiledb.open(tiledb_array) as A:
            assert pickle.loads(A.meta["__TENSORBOARD__"]) == log_files
        shutil.rmtree(log_dir)

        # Loading the event data should create local files
        tiledb_obj.load_tensorboard()
        new_log_files = read_files(log_dir)
        assert new_log_files == log_files

        custom_dir = os.path.join(tmpdir, "custom_log")
        tiledb_obj.load_tensorboard(target_dir=custom_dir)
        new_log_files = read_files(custom_dir)
        assert len(new_log_files) == len(log_files)
        for new_file, old_file in zip(new_log_files.values(), log_files.values()):
            assert new_file == old_file


def read_files(dirpath):
    files = {}
    for path in glob.glob(f"{dirpath}/*"):
        with open(path, "rb") as f:
            files[path] = f.read()
    return files


class TestPyTorchModelCloud:
    def test_get_cloud_uri_call_for_models_on_tiledb_cloud(self, tmpdir, mocker):
        model = Net()
        uri = os.path.join(tmpdir, "model_array")

        mock_get_cloud_uri = mocker.patch(
            "tiledb.ml.models._base.get_cloud_uri", return_value=uri
        )

        _ = PyTorchTileDBModel(uri=uri, namespace="test_namespace", model=model)

        mock_get_cloud_uri.assert_called_once_with(uri, "test_namespace")

    def test_get_s3_prefix_call_for_models_on_tiledb_cloud(self, tmpdir, mocker):
        model = Net()
        uri = os.path.join(tmpdir, "model_array")

        mock_get_s3_prefix = mocker.patch(
            "tiledb.ml.models._cloud_utils.get_s3_prefix", return_value="s3 prefix"
        )

        _ = PyTorchTileDBModel(uri=uri, namespace="test_namespace", model=model)

        mock_get_s3_prefix.assert_called_once_with("test_namespace")

    def test_update_file_properties_call(self, tmpdir, mocker):
        model = Net()
        uri = os.path.join(tmpdir, "model_array")

        mocker.patch("tiledb.ml.models._base.get_cloud_uri", return_value=uri)

        tiledb_obj = PyTorchTileDBModel(
            uri=uri, namespace="test_namespace", model=model
        )

        mock_update_file_properties = mocker.patch(
            "tiledb.ml.models.pytorch.update_file_properties", return_value=None
        )
        mocker.patch("tiledb.ml.models.pytorch.PyTorchTileDBModel._write_array")

        tiledb_obj.save()

        file_properties_dict = {
            "TILEDB_ML_MODEL_ML_FRAMEWORK": "PYTORCH",
            "TILEDB_ML_MODEL_ML_FRAMEWORK_VERSION": torch.__version__,
            "TILEDB_ML_MODEL_STAGE": "STAGING",
            "TILEDB_ML_MODEL_PYTHON_VERSION": platform.python_version(),
            "TILEDB_ML_MODEL_PREVIEW": str(model),
        }

        mock_update_file_properties.assert_called_once_with(uri, file_properties_dict)

    def test_exception_raise_file_property_in_meta_error(self, tmpdir):
        model = Net()
        tiledb_array = os.path.join(tmpdir, "model_array")
        tiledb_obj = PyTorchTileDBModel(uri=tiledb_array, model=model)
        with pytest.raises(ValueError) as ex:
            tiledb_obj.save(
                meta={"TILEDB_ML_MODEL_ML_FRAMEWORK": "TILEDB_ML_MODEL_ML_FRAMEWORK"},
            )

        assert "Please avoid using file property key names as metadata keys!" in str(
            ex.value
        )
