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
# Handle keras 3.x (TensorFlow >=2.16)
else:
    # SafeModeScope is not needed for Keras 3.x (major version >= 3)
    # as it's only used in the code when keras_major <= 2
    try:
        from keras.saving.serialization_lib import SafeModeScope
    except (ImportError, ModuleNotFoundError):
        # For Keras 3.x where this module structure changed
        SafeModeScope = None

    # Use public API for Keras 3.x
    # In Keras 3.x, there's no separate Functional class exposed in public API.
    # Functional models are instances of Model with _is_graph_network=True
    # We'll create a custom check for this
    class _FunctionalModel:
        """Dummy class for isinstance checks against functional models in Keras 3.x"""

        pass

    # For now, we only check Sequential. Functional model detection will be done
    # via attribute checking in the save method
    FunctionalOrSequential = keras.Sequential
    TFOptimizer = tf.keras.optimizers.legacy.Optimizer

    # In Keras 3.x, the legacy saving utilities don't exist
    # We need to provide simplified implementations or use alternatives
    def get_json_type(obj: Any) -> Any:
        """Fallback JSON type conversion for Keras 3.x"""
        # Basic type handling for JSON serialization
        if hasattr(obj, "tolist"):  # numpy arrays
            return obj.tolist()
        elif hasattr(obj, "__name__"):  # functions/classes
            return obj.__name__
        return str(obj)

    # Create a minimal saving_utils module replacement
    class _SavingUtils:
        @staticmethod
        def model_metadata(model: tf.keras.Model, include_optimizer: bool) -> dict:
            """Generate model metadata for Keras 3.x"""
            config = model.get_config()
            # Wrap config with class_name to match expected format
            model_config = {"class_name": model.__class__.__name__, "config": config}
            metadata = {
                "model_config": model_config,
            }
            if include_optimizer and model.optimizer:
                # Extract metrics properly - handle both string names and function objects
                metrics = []
                if hasattr(model, "compiled_metrics") and hasattr(
                    model.compiled_metrics, "_user_metrics"
                ):
                    user_metrics = model.compiled_metrics._user_metrics
                    # Handle if _user_metrics is wrapped in a list
                    if isinstance(user_metrics, list) and len(user_metrics) > 0:
                        # Flatten if it's a nested list
                        for item in user_metrics:
                            if isinstance(item, list):
                                for metric in item:
                                    if isinstance(metric, str):
                                        metrics.append(metric)
                                    elif hasattr(metric, "name"):
                                        metrics.append(metric.name)
                                    elif callable(metric):
                                        metrics.append(
                                            getattr(metric, "__name__", str(metric))
                                        )
                            else:
                                if isinstance(item, str):
                                    metrics.append(item)
                                elif hasattr(item, "name"):
                                    metrics.append(item.name)
                                elif callable(item):
                                    metrics.append(getattr(item, "__name__", str(item)))

                # Extract loss - handle if it's a list or dict
                loss_value = None
                if hasattr(model, "loss"):
                    loss = model.loss
                    if isinstance(loss, (list, tuple)) and len(loss) == 1:
                        loss_item = loss[0]
                        if isinstance(loss_item, str):
                            loss_value = loss_item
                        elif hasattr(loss_item, "name"):
                            loss_value = loss_item.name
                        elif hasattr(loss_item, "__name__"):
                            loss_value = loss_item.__name__
                        else:
                            loss_value = str(loss_item)
                    elif isinstance(loss, str):
                        loss_value = loss
                    elif hasattr(loss, "name"):
                        # For LossFunctionWrapper and similar objects
                        loss_value = loss.name
                    elif hasattr(loss, "__name__"):
                        loss_value = loss.__name__
                    else:
                        loss_value = str(loss)

                metadata["training_config"] = {
                    "optimizer_config": {
                        "class_name": model.optimizer.__class__.__name__,
                        "config": model.optimizer.get_config()
                        if hasattr(model.optimizer, "get_config")
                        else {},
                    },
                    "loss": loss_value,
                    "metrics": metrics if metrics else None,
                }
            return metadata

        @staticmethod
        def compile_args_from_training_config(
            training_config: dict, custom_objects: Optional[Mapping[str, Any]] = None
        ) -> dict:
            """Extract compile arguments from training config for Keras 3.x"""
            compile_args = {}
            if "optimizer_config" in training_config:
                opt_config = training_config["optimizer_config"]
                # Try to recreate optimizer from config
                try:
                    opt_class = getattr(
                        tf.keras.optimizers, opt_config["class_name"], None
                    )
                    if opt_class:
                        compile_args["optimizer"] = opt_class.from_config(
                            opt_config["config"]
                        )
                except Exception:
                    # Fallback to default optimizer
                    compile_args["optimizer"] = "adam"
            if "loss" in training_config and training_config["loss"]:
                compile_args["loss"] = training_config["loss"]
            if "metrics" in training_config and training_config["metrics"]:
                metrics = training_config["metrics"]
                # Ensure metrics is a list
                if not isinstance(metrics, list):
                    metrics = [metrics]
                compile_args["metrics"] = metrics
            return compile_args

        @staticmethod
        def try_build_compiled_arguments(model: tf.keras.Model) -> None:
            """Try to build compiled arguments for Keras 3.x"""
            # In Keras 3.x, this is typically not needed as models build automatically
            pass

    saving_utils = _SavingUtils()

    def preprocess_weights_for_loading(
        model: tf.keras.Model,
        weights: Any,
        original_keras_version: str,
        original_backend: str,
    ) -> Any:
        """Simplified weight preprocessing for Keras 3.x"""
        # In Keras 3.x, weight loading is more straightforward
        return weights


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
        teamspace: Optional[str] = None,
        ctx: Optional[tiledb.Ctx] = None,
        model: Optional[tf.keras.Model] = None,
    ):
        super().__init__(uri, teamspace, ctx, model)

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

        # Check if model is functional or sequential (not subclassed)
        is_valid_model = isinstance(self.artifact, FunctionalOrSequential)

        # For Keras 3.x, also check if it's a functional model
        # Keras 3.x functional models have class name 'Functional' or Sequential
        if not is_valid_model and int(keras_major) >= 3:
            model_class_name = self.artifact.__class__.__name__
            # Accept Sequential and Functional models (but not arbitrary Model subclasses)
            # Functional models will have _is_graph_network=True
            if model_class_name == "Sequential":
                is_valid_model = True
            elif model_class_name == "Functional":
                is_valid_model = True
            elif model_class_name == "Model":
                # Only accept Model if it was created via functional API
                is_valid_model = getattr(self.artifact, "_is_graph_network", False)

        if not is_valid_model:
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
                    # In Keras 3.x, use build() instead of _create_all_weights()
                    if int(keras_major) >= 3:
                        if hasattr(model.optimizer, 'build'):
                            model.optimizer.build(model.trainable_variables)
                    else:
                        model.optimizer._create_all_weights(model.trainable_variables)
                except (NotImplementedError, AttributeError) as e:
                    logging.warning(
                        f"Error when creating the weights of optimizer: {e}. "
                        "As a result, your model is starting with a freshly initialized optimizer."
                    )

                try:
                    model.optimizer.set_weights(optimizer_weights)
                except (ValueError, AttributeError) as e:
                    logging.warning(
                        f"Error in loading the saved optimizer state: {e}. "
                        "As a result, your model is starting with a freshly initialized optimizer."
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
                    # In Keras 3.x, use build() instead of _create_all_weights()
                    if int(keras_major) >= 3:
                        if hasattr(model.optimizer, 'build'):
                            model.optimizer.build(model.trainable_variables)
                    else:
                        model.optimizer._create_all_weights(model.trainable_variables)
                except (NotImplementedError, AttributeError) as e:
                    logging.warning(
                        f"Error when creating the weights of optimizer: {e}. "
                        "As a result, your model is starting with a freshly initialized optimizer."
                    )

                try:
                    model.optimizer.set_weights(optimizer_weights)
                except (ValueError, AttributeError) as e:
                    logging.warning(
                        f"Error in loading the saved optimizer state: {e}. "
                        "As a result, your model is starting with a freshly initialized optimizer."
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
                optimizer_weights = [var.numpy() for var in optimizer.variables]
            return pickle.dumps(optimizer_weights, protocol=4)
        return b""
