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
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.saving import model_config as model_config_lib
from tensorflow.python.keras import backend
from tensorflow.python.keras.saving.hdf5_format import preprocess_weights_for_loading

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
        # Used in this format only when model is Functional or Sequential
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
        input_shape: Optional[Tuple[int, ...]] = None,
    ) -> Model:
        """
        Loads a Tensorflow model from a TileDB array.
        :param compile_model: Boolean. Whether to compile the model after loading or not.
        :param custom_objects: Optional dictionary mapping names (strings) to
        custom classes or functions to be considered during deserialization.
        :param timestamp: Tuple of int. In case we want to use TileDB time travelling, we can provide a range of
        timestamps in order to load fragments of the array which live in the specified time range.
        :param input_shape: Tuple of integers with the shape that the custom model expects as input
        :return: Model. Tensorflow model.
        """

        with tiledb.open(self.uri, ctx=self.ctx, timestamp=timestamp) as model_array:
            model_array_results = model_array[:]
            model_config = json.loads(model_array.meta["model_config"])
            model_class = model_config["class_name"]

            if model_class != "Sequential" and model_class != "Functional":
                with generic_utils.SharedObjectLoadingScope():
                    with generic_utils.CustomObjectScope(custom_objects or {}):
                        if hasattr(model_config, "decode"):
                            model_config = model_config.decode("utf-8")
                        model = model_config_lib.model_from_config(
                            model_config, custom_objects=custom_objects
                        )
                        if not model.built:
                            model.build(input_shape)

                        # Load weights for layers
                        self._load_custom_subclassed_model(model, model_array)
            else:
                cls = (
                    tf.keras.Sequential
                    if model_class == "Sequential"
                    else tf.keras.Model
                )
                model = cls.from_config(model_config["config"])
                model_weights = pickle.loads(
                    model_array_results["model_weights"].item(0)
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
            )
            if isinstance(self.model, (Functional, Sequential))
            else tiledb.Dim(
                name="model",
                domain=(1, len(self.model.layers)),
                tile=1,
                dtype=np.int32,
                ctx=self.ctx,
            ),
        )
        if isinstance(self.model, (Functional, Sequential)):
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
        else:
            attrs = [
                # String names of weights of each layer of the model
                tiledb.Attr(
                    name="weight_names",
                    dtype="S1",
                    var=True,
                    filters=tiledb.FilterList([tiledb.ZstdFilter()]),
                    ctx=self.ctx,
                ),
                # The values of weights of each layer of the model
                tiledb.Attr(
                    name="weight_values",
                    dtype="S1",
                    var=True,
                    filters=tiledb.FilterList([tiledb.ZstdFilter()]),
                    ctx=self.ctx,
                ),
                # Layer names TF format of the saved/loaded model
                tiledb.Attr(
                    name="layer_name",
                    dtype="U1",
                    var=True,
                    filters=tiledb.FilterList([tiledb.ZstdFilter()]),
                    ctx=self.ctx,
                ),
                # The weight values of the optimizer in case the model is saved compiled
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

            if isinstance(self.model, (Functional, Sequential)):
                tf_model_tiledb[:] = {
                    "model_weights": np.array([serialized_weights]),
                    "optimizer_weights": np.array([serialized_optimizer_weights]),
                }
            else:
                # Insert weights and optimizer
                layer_names = []
                weight_names = []
                weight_values = []
                for layer in sorted(self.model.layers, key=lambda x: x.name):
                    weights = layer.trainable_weights + layer.non_trainable_weights
                    weight_values.append(pickle.dumps(backend.batch_get_value(weights)))
                    weight_names.append(
                        pickle.dumps([w.name.encode("utf8") for w in weights])
                    )
                    layer_names.append(layer.name)

                tf_model_tiledb[:] = {
                    "layer_name": np.array(layer_names),
                    "weight_values": np.array(weight_values),
                    "weight_names": np.array(weight_names),
                    # TODO (TeamML) Fix array scheme to avoid optimizer_weight repetition. Nullable
                    "optimizer_weights": np.array(
                        [
                            serialized_optimizer_weights
                            for _ in range(len(self.model.layers))
                        ]
                    ),
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

    def _load_custom_subclassed_model(self, model, model_array):

        if "keras_version" in model_array.meta:
            original_keras_version = model_array.meta["keras_version"]
            if hasattr(original_keras_version, "decode"):
                original_keras_version = original_keras_version.decode("utf8")
        else:
            original_keras_version = "1"
        if "backend" in model_array.meta:
            original_backend = model_array.meta["backend"]
            if hasattr(original_backend, "decode"):
                original_backend = original_backend.decode("utf8")
        else:
            original_backend = None

        # Load weights for layers
        self._load_weights_from_tiledb(
            model_array[:], model.layers, original_keras_version, original_backend
        )

    def _load_weights_from_tiledb(
        self,
        model_array_results,
        symbolic_layers,
        original_keras_version,
        original_backend,
    ):
        symbolic_layer_names = []
        for layer in symbolic_layers:
            weights = layer.trainable_weights + layer.non_trainable_weights
            if weights:
                symbolic_layer_names.append(layer)

        read_layer_names = []
        for k, name in enumerate(model_array_results["layer_name"]):
            layer_weight_names = pickle.loads(
                model_array_results["weight_names"].item(k)
            )
            if layer_weight_names:
                read_layer_names.append(name)

        if len(read_layer_names) != len(symbolic_layer_names):
            raise ValueError(
                f"You are trying to load a weight file containing {len(read_layer_names)} layers into a model with {len(symbolic_layer_names)} layers"
            )

        weight_value_tuples = []
        for k, name in enumerate(symbolic_layers):
            symbolic_weight_names = name.trainable_weights + name.non_trainable_weights
            read_weight_values = pickle.loads(
                model_array_results["weight_values"].item(k)
            )
            read_weight_values = preprocess_weights_for_loading(
                name, read_weight_values, original_keras_version, original_backend
            )
            if len(read_weight_values) != len(symbolic_weight_names):
                raise ValueError(
                    f'Layer #{k}  (named "{layer.name}" in the current model) was found to correspond to layer {name} in the save file. However the new layer {layer.name} expects {len(symbolic_weight_names)} weights, but the saved weights have {len(read_weight_values)} elements'
                )
            weight_value_tuples += zip(symbolic_weight_names, read_weight_values)
        backend.batch_set_value(weight_value_tuples)
