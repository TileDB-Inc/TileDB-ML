"""Functionality for saving and loading Sklearn models as TileDB arrays"""

import pickle
from typing import Optional

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

        self._write_array(model_params={"model": serialized_model}, meta=meta or {})

    def load(self, *, timestamp: Optional[Timestamp] = None) -> BaseEstimator:
        """
        Load switch, i.e, decide between __load (TileDB-ML<=0.8.0) or __load_v2 (TileDB-ML>0.8.0).

        Load a Sklearn model from a TileDB array.
        :param timestamp: Range of timestamps to load fragments of the array which live
            in the specified time range.
        :return: A Sklearn model object.
        """
        # TODO: Change timestamp when issue in core is resolved

        load = (
            self.__load_legacy
            if self._use_legacy_schema(timestamp=timestamp)
            else self.__load
        )
        return load(timestamp=timestamp)

    def __load_legacy(self, *, timestamp: Optional[Timestamp]) -> BaseEstimator:
        """
        Load a Sklearn model from a TileDB array.
        """
        model_array = tiledb.open(self.uri, ctx=self.ctx, timestamp=timestamp)
        model_array_results = model_array[:]
        model = pickle.loads(model_array_results["model_params"].item(0))
        return model

    def __load(self, *, timestamp: Optional[Timestamp]) -> BaseEstimator:
        """
        Load a Sklearn model from a TileDB array.
        """

        with tiledb.open(self.uri, ctx=self.ctx, timestamp=timestamp) as model_array:
            try:
                model_size = model_array.meta["model_size"]
            except KeyError:
                raise Exception(
                    f"model_size metadata entry not present in {self.uri}"
                    f" (existing keys: {set(model_array.meta.keys())})"
                )

            model_contents = model_array[0:model_size]["model"]
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
