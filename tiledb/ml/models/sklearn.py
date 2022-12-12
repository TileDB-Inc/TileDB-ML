"""Functionality for saving and loading Sklearn models as TileDB arrays"""

import pickle
from typing import Optional

import numpy as np
import sklearn
from sklearn import config_context
from sklearn.base import BaseEstimator

import tiledb

from ._base import Meta, TileDBArtifact, Timestamp


class SklearnTileDBModel(TileDBArtifact[BaseEstimator]):
    """
    Class that implements all functionality needed to save Sklearn models as
    TileDB arrays and load Sklearn models from TileDB arrays.
    """

    Name = "SKLEARN"
    Version = sklearn.__version__

    def __init__(
        self,
        uri: str,
        namespace: Optional[str] = None,
        ctx: Optional[tiledb.Ctx] = None,
        model: Optional[BaseEstimator] = None,
    ):
        super().__init__(uri, namespace, ctx, model)

    def save(self, *, update: bool = False, meta: Optional[Meta] = None) -> None:
        """
        Save a Sklearn model as a TileDB array.

        :param update: Whether we should update any existing TileDB array model at the
            target location.
        :param meta: Extra metadata to save in a TileDB array.
        """
        if self.artifact is None:
            raise RuntimeError("Model is not initialized")

        # Serialize model
        serialized_model = self._serialize_model()

        # Create TileDB model array
        if not update:
            self._create_array(fields=["model"])

        self._write_array(model_params={"model": serialized_model})

        if meta:
            self._write_model_metadata(meta=meta)

    def load(self, *, timestamp: Optional[Timestamp] = None) -> BaseEstimator:
        """
        Load switch, i.e, decide between __load (TileDB-ML<=0.8.0) or __load_v2 (TileDB-ML>0.8.0).

        Load a Sklearn model from a TileDB array.
        :param timestamp: Range of timestamps to load fragments of the array which live
            in the specified time range.
        :return: A Sklearn model object.
        """
        # TODO: Change timestamp when issue in core is resolved

        with tiledb.open(self.uri, ctx=self.ctx, timestamp=timestamp) as model_array:
            # Check if we try to load models with the old 1-cell schema.
            if model_array.schema.domain.size < np.iinfo(np.uint64).max - 1025:
                return self.__load(
                    timestamp=timestamp,
                )
            return self.__load_v2(
                timestamp=timestamp,
            )

    def __load(self, *, timestamp: Optional[Timestamp] = None) -> BaseEstimator:
        """
        Load a Sklearn model from a TileDB array.
        """
        model_array = tiledb.open(self.uri, ctx=self.ctx, timestamp=timestamp)
        model_array_results = model_array[:]
        model = pickle.loads(model_array_results["model_params"].item(0))
        return model

    def __load_v2(self, *, timestamp: Optional[Timestamp] = None) -> BaseEstimator:
        """
        Load a Sklearn model from a TileDB array.
        """

        with tiledb.open(self.uri, ctx=self.ctx, timestamp=timestamp) as model_array:
            model_meta = dict(model_array.meta.items())

            try:
                model_size = model_meta["model_size"]
            except KeyError:
                raise Exception(
                    f"model_size metadata entry not present in {self.uri}"
                    f" (existing keys: {set(model_meta)})"
                )

            model_contents: np.ndarray = model_array[0:model_size]["model"]
            model_bytes = model_contents.tobytes()

            return pickle.loads(model_bytes)

    def preview(self, *, display: str = "text") -> str:
        """
        Create a text representation of the model.

        :param display. If ‘diagram’, estimators will be displayed as a diagram in an
            HTML format when shown in a jupyter notebook. If ‘text’, estimators will be
            displayed as text.
        :return. A string representation of the models internal configuration.
        """
        if self.artifact:
            with config_context(display=display):
                return str(self.artifact)
        else:
            return ""

    def _serialize_model(self) -> bytes:
        """
        Serialize a Sklearn model with pickle.

        :return: Pickled Sklearn model.
        """
        return pickle.dumps(self.artifact, protocol=4)
