"""Base Class for saving and loading machine learning models."""

import abc
import os
import tiledb
import tiledb.cloud

from .cloud_utils import get_s3_prefix
from .cloud_utils import get_s3_credentials

FILETYPE_ML_MODEL = "ml_model"


class TileDBModel(abc.ABC):
    """
    This is the base class for all TileDB model storage functionalities, i.e,
    store machine learning models (Tensorflow, PyTorch, ect) as TileDB arrays.
    """

    def __init__(self, uri: str, ctx: tiledb.cloud.Ctx = None, namespace: str = None):
        """
        Base class for saving machine learning models as TileDB arrays
        and loading machine learning models from TileDB arrays.
        :param uri: str
        :ctx:
        :namespace: str
        """
        self.ctx = ctx

        if namespace and self.ctx:
            s3_prefix = get_s3_prefix(namespace)
            if s3_prefix is None:
                raise Exception(
                    "You must set the default s3 prefix path for ML models in {} profile settings".format(namespace)
                )

            self.uri = "tiledb://{}/{}".format(
                namespace, os.path.join(s3_prefix, uri)
            )

            # Retrieving credentials is optional
            # If None, default credentials will be used
            self.s3_credentials = get_s3_credentials(namespace)

            self.tiledb_uri = "tiledb://{}/{}".format(namespace, uri)
        else:
            self.uri = uri

    def get_model_info(self) -> tiledb.cloud.rest_api.models.array_info.ArrayInfo:
        """
        Check if an model exists in TileDB Cloud
        :return: ArrayInfo
        """
        try:
            model_info = tiledb.cloud.array.info(self.tiledb_uri)
            return model_info
        except tiledb.cloud.tiledb_cloud_error.TileDBCloudError as e:
            if str(e) == "Model or Namespace Not found":
                return None
            else:
                raise Exception(e)
        except tiledb.TileDBError as e:
            raise Exception(e)

        return None

    def delete_model(self):
        """Delete the file or directory at path."""
        try:
            if self.tiledb_uri and self.ctx:
                return tiledb.cloud.array.delete_array(self.tiledb_uri, "application/octet-stream")
            else:
                return tiledb.Array.remove(self.uri)
        except tiledb.cloud.tiledb_cloud_error.TileDBCloudError as e:
            raise Exception("Error deregistering {}: ".format(self.tiledb_uri, str(e))
            )
        except tiledb.TileDBError as e:
            raise Exception(e)


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
        :return: A Tensorflow or PyTorch model.
        """
