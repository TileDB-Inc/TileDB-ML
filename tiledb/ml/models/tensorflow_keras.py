"""Functionality for saving and loading Tensorflow Keras models as TileDB arrays"""

import io
import json
import logging
import os
import pickle
from operator import attrgetter
from typing import Any, List, Mapping, Optional, Tuple

import numpy as np
import tensorflow as tf

import tiledb

from ._base import Meta, TileDBArtifact, Timestamp, current_milli_time, group_create
from ._tensorboard import TensorBoardTileDB

try:
    import keras

    if keras.Model is not tf.keras.Model:
        raise ImportError
    tf_keras_is_keras = True
except ImportError:
    import tensorflow.python.keras as keras

    tf_keras_is_keras = False

SharedObjectLoadingScope = keras.utils.generic_utils.SharedObjectLoadingScope
FunctionalOrSequential = (keras.models.Functional, keras.models.Sequential)
TFOptimizer = keras.optimizer_v1.TFOptimizer
get_json_type = keras.saving.saved_model.json_utils.get_json_type
preprocess_weights_for_loading = keras.saving.hdf5_format.preprocess_weights_for_loading
saving_utils = keras.saving.saving_utils


class TensorflowKerasTileDBModel(TileDBArtifact[tf.keras.Model]):
    """
    Class that implements all functionality needed to save Tensorflow models as
    TileDB arrays and load Tensorflow models from TileDB arrays.
    """

    Name = "TENSORFLOW KERAS"
    Version = tf.__version__

    def __init__(
        self,
        uri: str,
        namespace: Optional[str] = None,
        ctx: Optional[tiledb.Ctx] = None,
        model: Optional[tf.keras.Model] = None,
    ):
        super().__init__(uri, namespace, ctx, model)

    def save(
        self,
        *,
        update: bool = False,
        meta: Optional[Meta] = None,
        include_optimizer: bool = False,
        include_callbacks: Optional[tf.keras.callbacks.CallbackList] = None,
    ) -> None:
        """
        Save a Tensorflow model as a TileDB array.

        :param update: Whether we should update any existing TileDB array model at the
            target location.
        :param meta: Extra metadata to save in a TileDB array.
        :param include_optimizer: Whether to save the optimizer or not.
        :param include_callbacks: Callbacks list to store their data in array's metadata
        """
        if self.artifact is None:
            raise RuntimeError("Model is not initialized")

        # Used in this format only when model is Functional or Sequential
        model_weights = pickle.dumps(self.artifact.get_weights(), protocol=4)

        # Serialize model optimizer
        optimizer_weights = self._serialize_optimizer_weights(
            include_optimizer=include_optimizer
        )

        # Create TileDB model array
        if not update:
            self.__create_array()

        self._write_array(
            include_optimizer=include_optimizer,
            serialized_weights=model_weights,
            serialized_optimizer_weights=optimizer_weights,
            meta=meta,
        )

        # Add callbacks to the group
        if include_callbacks:
            for cb in include_callbacks:
                if isinstance(cb, tf.keras.callbacks.TensorBoard):
                    # Save TensorBoard callback
                    tb = TensorBoardTileDB(
                        f"{self.uri}-tensorboard", self.ctx, self.namespace
                    )
                    tb.save(log_dir=os.path.join(cb.log_dir, "train"), update=update)

            # Create group for first time when callback is activated
            if not update:
                group_create(self.uri, self.ctx)

    def load(
        self,
        *,
        timestamp: Optional[Timestamp] = None,
        compile_model: bool = False,
        custom_objects: Optional[Mapping[str, Any]] = None,
        input_shape: Optional[Tuple[int, ...]] = None,
        callback: bool = False,
    ) -> tf.keras.Model:
        """
        Load a Tensorflow model from a TileDB array.

        :param callback: Boolean variable if True will store Callback data into saved directory
        :param timestamp: Range of timestamps to load fragments of the array which live
            in the specified time range.
        :param compile_model: Whether to compile the model after loading or not.
        :param custom_objects: Mapping of names to custom classes or functions to be
            considered during deserialization.
        :param input_shape: The shape that the custom model expects as input
        :return: Tensorflow model.
        """
        # TODO: Change timestamp when issue in core is resolved

        with tiledb.open(self.uri, ctx=self.ctx, timestamp=timestamp) as model_array:
            model_array_results = model_array[:]
            model_config = json.loads(model_array.meta["model_config"])
            model_class = model_config["class_name"]

            if model_class not in ("Functional", "Sequential"):
                with SharedObjectLoadingScope():
                    with tf.keras.utils.CustomObjectScope(custom_objects or {}):
                        if hasattr(model_config, "decode"):
                            model_config = model_config.decode("utf-8")
                        model = tf.keras.models.model_from_config(
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
        if callback:
            try:
                with tiledb.open(f"{self.uri}-tensorboard") as tb_array:
                    for path, file_bytes in pickle.loads(
                        tb_array[:]["tensorboard_data"][0]
                    ).items():
                        log_dir = os.path.dirname(path)
                        if not os.path.exists(log_dir):
                            os.mkdir(log_dir)
                        with open(
                            os.path.join(log_dir, os.path.basename(path)), "wb"
                        ) as f:
                            f.write(file_bytes)
            except FileNotFoundError:
                print(f"Array {self.uri}-tensorboard does not exist")
        return model

    def preview(self) -> str:
        """Create a string representation of the model."""
        if self.artifact:
            s = io.StringIO()
            self.artifact.summary(print_fn=lambda x: s.write(x + "\n"))
            model_summary = s.getvalue()
            return model_summary
        else:
            return ""

    def __create_array(self) -> None:
        """Create a TileDB array for a Tensorflow model"""
        assert self.artifact
        domain_info = (
            "model",
            (1, 1)
            if isinstance(self.artifact, FunctionalOrSequential)
            else (1, len(self.artifact.layers)),
        )
        if isinstance(self.artifact, FunctionalOrSequential):
            fields = ["model_weights", "optimizer_weights"]
        else:
            fields = [
                "weight_names",
                "weight_values",
                "layer_name",
                "optimizer_weights",
            ]
        super()._create_array(domain_info, fields)

    def _write_array(
        self,
        include_optimizer: bool,
        serialized_weights: bytes,
        serialized_optimizer_weights: bytes,
        meta: Optional[Meta],
    ) -> None:
        """Write Tensorflow model to a TileDB array."""
        assert self.artifact
        # TODO: Change timestamp when issue in core is resolved
        with tiledb.open(
            self.uri, "w", timestamp=current_milli_time(), ctx=self.ctx
        ) as tf_model_tiledb:
            if isinstance(self.artifact, FunctionalOrSequential):
                tf_model_tiledb[:] = {
                    "model_weights": np.array([serialized_weights]),
                    "optimizer_weights": np.array([serialized_optimizer_weights]),
                }
            else:
                # Insert weights and optimizer
                layer_names = []
                weight_names = []
                weight_values = []
                for layer in sorted(self.artifact.layers, key=attrgetter("name")):
                    weights = layer.trainable_weights + layer.non_trainable_weights
                    weight_values.append(
                        pickle.dumps(tf.keras.backend.batch_get_value(weights))
                    )
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
                            for _ in range(len(self.artifact.layers))
                        ]
                    ),
                }

            # Insert all model metadata
            model_metadata = saving_utils.model_metadata(
                self.artifact, include_optimizer
            )
            for key, value in model_metadata.items():
                tf_model_tiledb.meta[key] = json.dumps(
                    value, default=get_json_type
                ).encode("utf8")

            self.update_model_metadata(array=tf_model_tiledb, meta=meta)

    def _serialize_optimizer_weights(self, include_optimizer: bool = True) -> bytes:
        """Serialize optimizer weights"""
        assert self.artifact
        optimizer = self.artifact.optimizer
        if include_optimizer and optimizer and not isinstance(optimizer, TFOptimizer):
            optimizer_weights = tf.keras.backend.batch_get_value(optimizer.weights)
            return pickle.dumps(optimizer_weights, protocol=4)
        else:
            return b""

    def _load_custom_subclassed_model(
        self, model: tf.keras.Model, model_array: tiledb.Array
    ) -> None:
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
            model_array[:], model, original_keras_version, original_backend
        )

    def _load_weights_from_tiledb(
        self,
        model_array_results: Mapping[str, Any],
        model: tf.keras.Model,
        original_keras_version: Optional[str],
        original_backend: Optional[str],
    ) -> None:
        num_layers = 0
        for layer in model.layers:
            weights = layer.trainable_weights + layer.non_trainable_weights
            if weights:
                num_layers += 1

        read_layer_names = []
        for k, name in enumerate(model_array_results["layer_name"]):
            layer_weight_names = pickle.loads(
                model_array_results["weight_names"].item(k)
            )
            if layer_weight_names:
                read_layer_names.append(name)

        if len(read_layer_names) != num_layers:
            raise ValueError(
                f"You are trying to load a weight file with {len(read_layer_names)} "
                f"layers into a model with {num_layers} layers"
            )

        var_value_tuples: List[Tuple[tf.Variable, np.ndarray]] = []
        for k, layer in enumerate(model.layers):
            weight_vars = layer.trainable_weights + layer.non_trainable_weights
            read_weight_values = pickle.loads(
                model_array_results["weight_values"].item(k)
            )
            read_weight_values = preprocess_weights_for_loading(
                layer, read_weight_values, original_keras_version, original_backend
            )
            if len(read_weight_values) != len(weight_vars):
                raise ValueError(
                    f'Layer #{k}  (named "{layer.name}" in the current model) was found '
                    f"to correspond to layer {layer} in the save file. However the new "
                    f"layer {layer.name} expects {len(weight_vars)} weights, "
                    f"but the saved weights have {len(read_weight_values)} elements"
                )
            var_value_tuples.extend(zip(weight_vars, read_weight_values))
        tf.keras.backend.batch_set_value(var_value_tuples)
