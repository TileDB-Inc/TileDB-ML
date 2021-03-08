import torch
import unittest
import tempfile

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from models.pytorch_models import PyTorchTileDB


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
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
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


class TestSaveLoadTileDBModel(unittest.TestCase):

    def test_save_and_load_snn(self):
        with tempfile.TemporaryDirectory() as tiledb_array:
            net = SeqNeuralNetwork()
            optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
            EPOCH = 5
            LOSS = 0.4

            tiledb_obj = PyTorchTileDB(uri=tiledb_array)
            tiledb_obj.save(model_info={
                'epoch': EPOCH,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': LOSS
            })

            loaded_net = SeqNeuralNetwork()
            loaded_optimizer = optim.SGD(loaded_net.parameters(), lr=0.001, momentum=0.9)
            model_out_dict = tiledb_obj.load(model=loaded_net, optimizer=loaded_optimizer)

            self.assertEqual(EPOCH, model_out_dict['epoch'])
            self.assertEqual(LOSS, model_out_dict['loss'])

            # Check model parameters
            for key_item_1, key_item_2 in zip(net.state_dict().items(), loaded_net.state_dict().items()):
                self.assertTrue(torch.equal(key_item_1[1], key_item_2[1]))

            # Check optimizer parameters
            for key_item_1, key_item_2 in zip(optimizer.state_dict().items(), loaded_optimizer.state_dict().items()):
                self.assertEqual(key_item_1[1], key_item_2[1])

    def test_save_and_load_cnn(self):
        with tempfile.TemporaryDirectory() as tiledb_array:
            net = ConvNet()
            optimizer = optim.Adagrad(net.parameters(), lr=0.001)
            EPOCH = 5
            LOSS = 0.4

            tiledb_obj = PyTorchTileDB(uri=tiledb_array)
            tiledb_obj.save(model_info={
                'epoch': EPOCH,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': LOSS
            })

            loaded_net = ConvNet()
            loaded_optimizer = optim.Adagrad(loaded_net.parameters(), lr=0.001)
            model_out_dict = tiledb_obj.load(model=loaded_net, optimizer=loaded_optimizer)

            self.assertEqual(EPOCH, model_out_dict['epoch'])
            self.assertEqual(LOSS, model_out_dict['loss'])

            # Check model parameters
            for key_item_1, key_item_2 in zip(net.state_dict().items(), loaded_net.state_dict().items()):
                self.assertTrue(torch.equal(key_item_1[1], key_item_2[1]))

            # Check optimizer parameters
            optimizer_state = optimizer.state
            loaded_optimizer_state = loaded_optimizer.state

            for key_item_1, key_item_2 in zip(optimizer_state.items(), loaded_optimizer_state.items()):
                self.assertTrue(torch.equal(key_item_1[1]['sum'], key_item_2[1]['sum']))

    def test_save_and_load_nn(self):
        with tempfile.TemporaryDirectory() as tiledb_array:
            net = Net()
            optimizer = optim.Adam(net.parameters(), lr=0.001)
            EPOCH = 5
            LOSS = 0.4

            tiledb_obj = PyTorchTileDB(uri=tiledb_array)
            tiledb_obj.save(model_info={
                'epoch': EPOCH,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': LOSS
            })

            loaded_net = Net()
            loaded_optimizer = optim.Adam(loaded_net.parameters(), lr=0.001)
            model_out_dict = tiledb_obj.load(model=loaded_net, optimizer=loaded_optimizer)

            self.assertEqual(EPOCH, model_out_dict['epoch'])
            self.assertEqual(LOSS, model_out_dict['loss'])

            # Check model parameters
            for key_item_1, key_item_2 in zip(net.state_dict().items(), loaded_net.state_dict().items()):
                self.assertTrue(torch.equal(key_item_1[1], key_item_2[1]))

            # Check optimizer parameters
            optimizer_state = optimizer.state
            loaded_optimizer_state = loaded_optimizer.state

            for key_item_1, key_item_2 in zip(optimizer_state.items(), loaded_optimizer_state.items()):
                self.assertTrue(torch.equal(key_item_1[1]['sum'], key_item_2[1]['sum']))


if __name__ == '__main__':
    unittest.main()