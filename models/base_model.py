"""Base Class for saving and loading machine learning models."""

import abc


class TileDBModel(abc.ABC):
    """
    This is the base class for all TileDB model storage functionalities, i.e,
    store machine learning models (Tensorflow, PyTorch, ect) as TileDB arrays.
    """
    def __init__(self, uri: str):
        """
        Base class for saving machine learning models as TileDB arrays
        and loading machine learning models from TileDB arrays.
        :param uri: str
        """
        self.uri = uri
        super().__init__()

    @abc.abstractmethod
    def save(self, **kwargs):
        """
        Abstract method that saves a machine learning model as a TileDB array.
        Must be implemented per machine learning framework, i.e, Tensorflow,
        PyTorch etc.
        """
        raise NotImplementedError("Please implement save method!")

    @abc.abstractmethod
    def load(self, **kwargs):
        """
        Abstract method that loads a machine learning model from a model TileDB array.
        Must be implemented per machine learning framework.
        :return: A Tensorflow or PyTorch model.
        """
        raise NotImplementedError("Please implement load method!")
