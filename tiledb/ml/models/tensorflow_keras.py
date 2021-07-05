"""Functionality for saving and loading Tensorflow Keras models as TileDB arrays"""

import io
import logging
import json
import pickle
import numpy as np
import tensorflow as tf
import tiledb

from typing import Optional, Tuple

from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.engine.functional import Functional
from tensorflow.python.keras import optimizer_v1
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.saving.saved_model import json_utils

from .base import TileDBModel


class TensorflowKerasTileDBModel(TileDBModel):
    """
    Class that implements all functionality needed to save Tensorflow models as
    TileDB arrays and load Tensorflow models from TileDB arrays.
    """

    Framework = "TENSORFLOW KERAS"
    FrameworkVersion = tf.__version__

    def save(
        self,
        include_optimizer: bool = False,
        update: bool = False,
        meta: Optional[dict] = {},
    ):
        """
        Saves a Tensorflow model as a TileDB array.
        :param include_optimizer: Boolean. Whether to save the optimizer or not.
        :param update: Boolean. Whether we should update any existing TileDB array model at the target location.
        :param meta: Dict. Extra metadata to save in a TileDB array.
        """
        if not isinstance(self.model, (Functional, Sequential)):
            raise NotImplementedError(
                "No support for Subclassed models at the moment. Your "
                "model should be either Sequential or Functional."
            )

        # Serialize model weights and optimizer (if needed)
        model_weights = pickle.dumps(self.model.get_weights(), protocol=4)

        # Serialize model optimizer
        optimizer_weights = self._serialize_optimizer_weights(
            include_optimizer=include_optimizer
        )

        # Create TileDB model array
        if not update:
            self._create_array()

        self._write_array(
            include_optimizer=include_optimizer,
            serialized_weights=model_weights,
            serialized_optimizer_weights=optimizer_weights,
            meta=meta,
        )

    def load(
        self,
        compile_model: bool = False,
        custom_objects: Optional[dict] = None,
        timestamp: Optional[Tuple[int, int]] = None,
    ) -> Model:
        """
        Loads a Tensorflow model from a TileDB array.
        :param compile_model: Boolean. Whether to compile the model after loading or not.
        :param custom_objects: Optional dictionary mapping names (strings) to
        custom classes or functions to be considered during deserialization.
        :param timestamp: Tuple of int. In case we want to use TileDB time travelling, we can provide a range of
        timestamps in order to load fragments of the array which live in the specified time range.
        :return: Model. Tensorflow model.
        """
        model_array = tiledb.open(self.uri, ctx=self.ctx, timestamp=timestamp)
        model_array_results = model_array[:]
        model_weights = pickle.loads(model_array_results["model_weights"].item(0))
        model_config = json.loads(model_array.meta["model_config"])

        architecture = model_config["config"]
        model_class = model_config["class_name"]

        if model_class == "Sequential":
            model = tf.keras.Sequential.from_config(architecture)
        elif model_class == "Functional":
            model = tf.keras.Model.from_config(architecture)
        else:
            raise NotImplementedError(
                "No support for Subclassed models at the moment. Your "
                "model should be either Sequential or Functional."
            )

        model.set_weights(model_weights)

        if compile_model:
            optimizer_weights = pickle.loads(
                model_array_results["optimizer_weights"].item(0)
            )
            training_config = json.loads(model_array.meta["training_config"])

            # Compile model.
            model.compile(
                **saving_utils.compile_args_from_training_config(
                    training_config, custom_objects
                )
            )
            saving_utils.try_build_compiled_arguments(model)

            # Set optimizer weights.
            if optimizer_weights:
                try:
                    model.optimizer._create_all_weights(model.trainable_variables)
                except (NotImplementedError, AttributeError):
                    logging.warning(
                        "Error when creating the weights of optimizer {}, making it "
                        "impossible to restore the saved optimizer state. As a result, "
                        "your model is starting with a freshly initialized optimizer."
                    )

                try:
                    model.optimizer.set_weights(optimizer_weights)
                except ValueError:
                    logging.warning(
                        "Error in loading the saved optimizer "
                        "state. As a result, your model is "
                        "starting with a freshly initialized "
                        "optimizer."
                    )
        return model

    def preview(self) -> str:
        """
        Creates a string representation of the model.
        :return: str. A string representation of the models internal configuration.
        """
        if self.model:
            s = io.StringIO()
            self.model.summary(print_fn=lambda x: s.write(x + "\n"))
            model_summary = s.getvalue()
            return model_summary
        else:
            return ""

    def _create_array(self):
        """
        Creates a TileDB array for a Tensorflow model
        """
        dom = tiledb.Domain(
            tiledb.Dim(
                name="model", domain=(1, 1), tile=1, dtype=np.int32, ctx=self.ctx
            ),
        )

        attrs = [
            tiledb.Attr(
                name="model_weights",
                dtype="S1",
                var=True,
                filters=tiledb.FilterList([tiledb.ZstdFilter()]),
                ctx=self.ctx,
            ),
            tiledb.Attr(
                name="optimizer_weights",
                dtype="S1",
                var=True,
                filters=tiledb.FilterList([tiledb.ZstdFilter()]),
                ctx=self.ctx,
            ),
        ]

        schema = tiledb.ArraySchema(
            domain=dom,
            sparse=False,
            attrs=attrs,
            ctx=self.ctx,
        )

        tiledb.Array.create(self.uri, schema, ctx=self.ctx)

        # In case we are on TileDB-Cloud we have to update model array's file properties
        if self.namespace:
            from tiledb.ml._cloud_utils import update_file_properties

            update_file_properties(self.uri, self._file_properties)

    def _write_array(
        self,
        include_optimizer: bool,
        serialized_weights: bytes,
        serialized_optimizer_weights: bytes,
        meta: Optional[dict],
    ):
        """
        Writes Tensorflow model to a TileDB array.
        """
        with tiledb.open(self.uri, "w", ctx=self.ctx) as tf_model_tiledb:
            # Insert weights and optimizer
            tf_model_tiledb[:] = {
                "model_weights": np.array([serialized_weights]),
                "optimizer_weights": np.array([serialized_optimizer_weights]),
            }

            # Insert all model metadata
            model_metadata = saving_utils.model_metadata(self.model, include_optimizer)
            for key, value in model_metadata.items():
                tf_model_tiledb.meta[key] = json.dumps(
                    value, default=json_utils.get_json_type
                ).encode("utf8")

            self.update_model_metadata(array=tf_model_tiledb, meta=meta)

    def _serialize_model_weights(self) -> bytes:
        """
        Serialization of model weights
        """
        return pickle.dumps(self.model.get_weights(), protocol=4)

    def _serialize_optimizer_weights(self, include_optimizer: bool = True) -> bytes:
        """
        Serialization of optimizer weights
        """
        if (
            include_optimizer
            and self.model.optimizer
            and not isinstance(self.model.optimizer, optimizer_v1.TFOptimizer)
        ):

            optimizer_weights = tf.keras.backend.batch_get_value(
                getattr(self.model.optimizer, "weights")
            )

            return pickle.dumps(optimizer_weights, protocol=4)
        else:
            return b""
