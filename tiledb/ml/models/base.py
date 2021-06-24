"""Base Class for saving and loading machine learning models."""

import abc
import os
import tiledb
import typing


class TileDBModel(abc.ABC):
    """
    This is the base class for all TileDB model storage functionalities, i.e,
    store machine learning models (Tensorflow, PyTorch, etc) as TileDB arrays.
    """

    def __init__(self, uri: str, namespace: str = None, ctx: tiledb.Ctx = None):
        """
        Base class for saving machine learning models as TileDB arrays
        and loading machine learning models from TileDB arrays. In case we need to interact
        with TileDB-Cloud we have to pass user's TileDB-Cloud namespace. If we don't
        models will be saved locally.
        :param uri: str. TileDB array uri
        :param namespace: str. In case we want to interact (save, load, update, check) with models on
        TileDB-Cloud we need the user's namespace on TileDB-Cloud. Moreover, array's uri must have an s3 prefix.
        :param ctx: tiledb.Ctx. TileDB Context.
        """
        self.namespace = namespace
        self.ctx = ctx

        # In case we work on TileDB-Cloud we need user's namespace.
        if self.namespace:
            from tiledb.ml._cloud_utils import get_s3_prefix

            s3_prefix = get_s3_prefix(self.namespace)
            if s3_prefix is None:
                raise ValueError(
                    "You must set the default s3 prefix path for ML models in {} profile settings on TileDB-Cloud".format(
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

    @abc.abstractmethod
    def preview(self, model: typing.Any, **kwargs):
        """
        Abstract method that previews a machine learning model.
        Must be implemented per machine learning framework, i.e, Tensorflow,
        PyTorch etc.
        """
