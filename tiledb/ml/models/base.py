"""Base Class for saving and loading machine learning models."""

import abc
import os

from .cloud_utils import get_s3_prefix

FILETYPE_ML_MODEL = "ml_model"
FilePropertyName_ML_FRAMEWORK = "ML_FRAMEWORK"
FilePropertyName_STAGE = "STAGE"


class TileDBModel(abc.ABC):
    """
    This is the base class for all TileDB model storage functionalities, i.e,
    store machine learning models (Tensorflow, PyTorch, etc) as TileDB arrays.
    """

    def __init__(self, uri: str, namespace: str = None):
        """
        Base class for saving machine learning models as TileDB arrays
        and loading machine learning models from TileDB arrays.
        :param uri: str. TileDB array uri
        :param namespace: str. In case we want to interact (save, load, update, check) with models on
        TileDB-Cloud we need the user's namespace on TileDB-Cloud. Moreover, array's uri must have an s3 prefix.
        """
        self.namespace = namespace

        # In case we work on TileDB-Cloud
        if self.namespace:
            s3_prefix = get_s3_prefix(self.namespace)
            if s3_prefix is None:
                raise Exception(
                    "You must set the default s3 prefix path for ML models in {} profile settings".format(
                        self.namespace
                    )
                )

            self.uri = "tiledb://{}/{}".format(
                self.namespace, os.path.join(s3_prefix, uri)
            )
        else:
            self.uri = uri

    @abc.abstractmethod
    def save(self, **kwargs):
        """
        Abstract method that saves a machine learning model as a TileDB array.
        Must be implemented per machine learning framework, i.e, Tensorflow,
        PyTorch etc.
        """

    @abc.abstractmethod
    def load(self, **kwargs):
        """
        Abstract method that loads a machine learning model from a model TileDB array.
        Must be implemented per machine learning framework.
        """
