from abc import ABC, abstractmethod


class TileDBModel(ABC):
    def __init__(self, uri: str, input_shape: tuple, output_shape: tuple, description: str):
        """
        Base class for saving machine learning models as TileDB arrays and loading machine learning models from TileDB
        arrays.
        :param uri: str
        :param input_shape: tuple
        :param output_shape: tuple
        :param description: str
        """
        self.uri = uri
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.description = description
        super().__init__()

    @abstractmethod
    def save(self):
        """
        Abstract method that saves a machine learning model as a TileDB array. Must be implemented per machine learning
        framework, i.e, Tensorflow, PyTorch etc.
        """
        raise NotImplementedError("Please implement save method!")

    @abstractmethod
    def load(self):
        """
        Abstract method that loads a machine learning model from a model TileDB array. Must be implemented per machine
        learning framework.
        :return: Model
            Could be a Tensorflow or PyTorch model
        """
        raise NotImplementedError("Please implement load method!")
