"""Base Class for saving and loading machine learning models."""

import abc
import os
import tiledb.cloud
from urllib.error import HTTPError

FILETYPE_ML_MODEL = "ml_model"
CLOUD_MODELS = "ml_models"


class TileDBModel(abc.ABC):
    """
    This is the base class for all TileDB model storage functionalities, i.e,
    store machine learning models (Tensorflow, PyTorch, ect) as TileDB arrays.
    """

    def __init__(self, uri: str, ctx=None, namespace=None):
        """
        Base class for saving machine learning models as TileDB arrays
        and loading machine learning models from TileDB arrays.
        :param uri: str
        """
        self.uri = uri
        self.tiledb_uri = None
        self.ctx = ctx
        self.namespace = namespace

        if self.namespace and self.ctx:
            s3_prefix = self.get_s3_prefix()
            if s3_prefix is None:
                raise HTTPError(
                    403,
                    "You must set the default s3 prefix path for ML models in {} profile settings".format(
                        namespace
                    ),
                )

            self.uri = "tiledb://{}/{}".format(
                namespace, os.path.join(s3_prefix, uri)
            )

            self.tiledb_uri = "tiledb://{}/{}".format(namespace, uri)
        else:
            self.uri = uri

    def get_s3_prefix(self):
        """
        Get S3 path from the user profile or organization profile
        :return: s3 path or error
        """
        try:
            profile = tiledb.cloud.client.user_profile()

            if self.namespace == profile.username:
                if profile.default_s3_path is not None:
                    return os.path.join(profile.default_s3_path, CLOUD_MODELS)
            else:
                organization = tiledb.cloud.client.organization(self.namespace)
                if organization.default_s3_path is not None:
                    return os.path.join(organization.default_s3_path, CLOUD_MODELS)
        except tiledb.cloud.tiledb_cloud_error.TileDBCloudError as e:
            raise HTTPError(
                400,
                "Error fetching user default s3 path for new ML models {}".format(str(e)),
            )

        return None

    def get_s3_credentials(self):
        """
        Get credentials for default S3 path from the user profile or organization profile
        :return: s3 credentials or error
        """
        try:
            profile = tiledb.cloud.client.user_profile()

            if self.namespace == profile.username:
                if profile.default_s3_path_credentials_name is not None:
                    return profile.default_s3_path_credentials_name
            else:
                organization = tiledb.cloud.client.organization(self.namespace)
                if organization.default_s3_path_credentials_name is not None:
                    return organization.default_s3_path_credentials_name
        except tiledb.cloud.tiledb_cloud_error.TileDBCloudError as e:
            raise HTTPError(
                400,
                "Error fetching default credentials for {} default s3 path for new ML models {}".format(
                    self.namespace, str(e)
                ),
            )

        return None

    def _model_exists(self):
        """
        Check if an model exists in TileDB Cloud
        :return:
        """
        try:
            tiledb.cloud.array.info(self.tiledb_uri)
            return True
        except tiledb.cloud.tiledb_cloud_error.TileDBCloudError as e:
            if str(e) == "Array or Namespace Not found":
                return False

        return False

    def get_model_info(self):
        """
        Check if an model exists in TileDB Cloud
        :return:
        """
        try:
            model_info = tiledb.cloud.array.info(self.tiledb_uri)
            return model_info
        except tiledb.cloud.tiledb_cloud_error.TileDBCloudError as e:
            if str(e) == "Array or Namespace Not found":
                return None

        return None

    def delete_model(self):
        """Delete the file or directory at path."""
        try:
            return tiledb.cloud.array.delete_array(self.tiledb_uri, "application/x-ipynb+json")
        except tiledb.cloud.tiledb_cloud_error.TileDBCloudError as e:
            raise HTTPError(
                500, "Error deregistering {}: ".format(self.tiledb_uri, str(e))
            )
        except tiledb.TileDBError as e:
            raise HTTPError(
                500,
                str(e),
            )


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
