"""Functionality for saving and loading Tensorflow Keras models as TileDB arrays"""

import io
import json
import logging
import os
import pickle
from collections import ChainMap
from typing import Any, Mapping, Optional, Tuple

import keras
import tensorflow as tf

import tiledb

from . import SHORT_PREVIEW_LIMIT
from ._base import Meta, TileDBArtifact, Timestamp

keras_major, keras_minor, keras_patch = keras.__version__.split(".")
FunctionalOrSequential = keras.models.Sequential
# Handle keras <=v2.10
if int(keras_major) <= 2 and int(keras_minor) <= 10:
    FunctionalOrSequential = (keras.models.Functional, keras.models.Sequential)
    TFOptimizer = keras.optimizers.TFOptimizer
    get_json_type = keras.saving.saved_model.json_utils.get_json_type
    preprocess_weights_for_loading = (
        keras.saving.hdf5_format.preprocess_weights_for_loading
    )
    saving_utils = keras.saving.saving_utils
# Handle keras >=v2.11
elif int(keras_major) <= 2 and int(keras_minor) <= 12:
    FunctionalOrSequential = (keras.models.Functional, keras.models.Sequential)
    TFOptimizer = tf.keras.optimizers.legacy.Optimizer
    get_json_type = keras.saving.legacy.saved_model.json_utils.get_json_type
    preprocess_weights_for_loading = (
        keras.saving.legacy.hdf5_format.preprocess_weights_for_loading
    )
    saving_utils = keras.saving.legacy.saving_utils
else:
    from keras.src.saving.serialization_lib import SafeModeScope

    FunctionalOrSequential = (
        keras.src.engine.functional.Functional,
        keras.src.engine.sequential.Sequential,
    )
    TFOptimizer = tf.keras.optimizers.legacy.Optimizer
    get_json_type = keras.src.saving.legacy.saved_model.json_utils.get_json_type
    preprocess_weights_for_loading = (
        keras.src.saving.legacy.hdf5_format.preprocess_weights_for_loading
    )
    saving_utils = keras.src.saving.legacy.saving_utils


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
        meta: Optional[Meta] = None,
        include_optimizer: bool = False,
        callbacks: Optional[tf.keras.callbacks.CallbackList] = None,
    ) -> None:
        """
        Save a Tensorflow model as a TileDB array.

        :param meta: Extra metadata to save in a TileDB array.
        :param include_optimizer: Whether to save the optimizer or not.
        :param callbacks: Callbacks list to store. At the moment only Tensorboard callback is supported.
        """
        if self.artifact is None:
            raise RuntimeError("Model is not initialized")

        if not isinstance(self.artifact, FunctionalOrSequential):
            raise RuntimeError(
                f"Subclassed Models (Custom Layers) for {type(self.artifact)} not supported at the moment."
            )

        # Used in this format only when model is Functional or Sequential
        model_weights = pickle.dumps(self.artifact.get_weights(), protocol=4)

        # Serialize model optimizer
        if include_optimizer:
            optimizer_weights = self._serialize_optimizer_weights()
        else:
            optimizer_weights = b""

        # Serialize Tensorboard files
        tensorboard_log_dir = None
        if callbacks:
            for cb in callbacks:
                if isinstance(cb, tf.keras.callbacks.TensorBoard):
                    tensorboard_log_dir = os.path.join(cb.log_dir, "train")
                else:
                    raise NotImplementedError(cb)

        # Create TileDB model array
        if not tiledb.array_exists(uri=self.uri):
            self._create_array(fields=["model", "optimizer", "tensorboard"])

        # Write extra metadata. Only for Tensoflow models.
        model_meta = saving_utils.model_metadata(
            model=self.artifact,
            include_optimizer=bool(optimizer_weights),
        )
        for key, value in model_meta.items():
            model_meta[key] = json.dumps(value, default=get_json_type).encode("utf8")

        self._write_array(
            model_params={
                "model": model_weights,
                "optimizer": optimizer_weights,
            },
            tensorboard_log_dir=tensorboard_log_dir,
            meta=ChainMap(model_meta, dict(meta or ())),
        )

    def load(
        self,
        *,
        timestamp: Optional[Timestamp] = None,
        compile_model: bool = False,
        custom_objects: Optional[Mapping[str, Any]] = None,
        input_shape: Optional[Tuple[int, ...]] = None,
        callback: bool = False,
        safe_mode: Optional[bool] = None,
    ) -> tf.keras.Model:
        """
        Load switch, i.e, decide between __load (TileDB-ML<=0.8.0) or __load_v2 (TileDB-ML>0.8.0).

        :param callback: Boolean variable if True will store callback data into saved directory. At the moment supports
        only Tensorboard callback.
        :param timestamp: Range of timestamps to load fragments of the array which live
            in the specified time range.
        :param compile_model: Whether to compile the model after loading or not.
        :param custom_objects: Mapping of names to custom classes or functions to be
            considered during deserialization.
        :param input_shape: The shape that the custom model expects as input
        :return: Tensorflow model.
        """
        with tiledb.open(self.uri, ctx=self.ctx, timestamp=timestamp) as model_array:
            if self._use_legacy_schema(model_array):
                return self.__load_legacy(
                    model_array, compile_model, callback, custom_objects
                )
            else:
                return self.__load(model_array, compile_model, callback, safe_mode)

    def __load_legacy(
        self,
        model_array: tiledb.Array,
        compile_model: bool,
        callback: bool,
        custom_objects: Optional[Mapping[str, Any]],
    ) -> tf.keras.Model:
        model_array_results = model_array[:]
        model_config = json.loads(model_array.meta["model_config"])
        model_class = model_config["class_name"]

        cls = tf.keras.Sequential if model_class == "Sequential" else tf.keras.Model
        model = cls.from_config(model_config["config"])
        model_weights = pickle.loads(model_array_results["model_weights"].item(0))
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

    def __load(
        self,
        model_array: tiledb.Array,
        compile_model: bool,
        callback: bool,
        safe_mode: Optional[bool],
    ) -> tf.keras.Model:
        model_config = json.loads(model_array.meta["model_config"])
        model_class = model_config["class_name"]

        cls = tf.keras.Sequential if model_class == "Sequential" else tf.keras.Model

        if int(keras_major) <= 2 and int(keras_minor) >= 13:
            if safe_mode is not None:
                with SafeModeScope(safe_mode=safe_mode):
                    model = cls.from_config(model_config["config"])
            else:
                model = cls.from_config(model_config["config"])
        else:
            model = cls.from_config(model_config["config"])
        model_weights = self._get_model_param(model_array, "model")
        model.set_weights(model_weights)

        if compile_model:
            training_config = json.loads(model_array.meta["training_config"])

            # Compile model.
            model.compile(
                **saving_utils.compile_args_from_training_config(
                    training_config,
                )
            )

            saving_utils.try_build_compiled_arguments(model)

            optimizer_weights = self._get_model_param(model_array, "optimizer")

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
            self._load_tensorboard(model_array)

        return model

    def preview(self) -> str:
        """Create a string representation of the Tensorflow model."""
        if self.artifact:
            str_rep = io.StringIO()
            self.artifact.summary(print_fn=lambda x: str_rep.write(x + "\n"))
            model_summary = str_rep.getvalue()
            return model_summary
        return ""

    def preview_short(self) -> str:
        """Create a string representation of the Tensorflow model that is under 2048 characters."""
        return self.preview()[0:SHORT_PREVIEW_LIMIT]

    def _serialize_optimizer_weights(
        self,
    ) -> bytes:
        """Serialize optimizer weights."""
        assert self.artifact
        optimizer = self.artifact.optimizer
        if optimizer and not isinstance(optimizer, TFOptimizer):
            if hasattr(optimizer, "weights"):
                optimizer_weights = tf.keras.backend.batch_get_value(optimizer.weights)
            else:
                optimizer_weights = [var.numpy() for var in optimizer.variables()]
            return pickle.dumps(optimizer_weights, protocol=4)
        return b""
