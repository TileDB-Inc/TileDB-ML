from dataclasses import dataclass
from enum import Enum, unique
from typing import Any

import tensorflow as tf

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


@dataclass
class TFSpec:
    def __init__(self, model: tf.keras.Model) -> None:
        self.domain_info = (
            "model",
            (1, 1)
            if isinstance(model, FunctionalOrSequential)
            else (1, len(model.layers)),
        )
        if isinstance(model, FunctionalOrSequential):
            self.fields = ["model_weights", "optimizer_weights"]
        else:
            self.fields = [
                "weight_names",
                "weight_values",
                "layer_name",
                "optimizer_weights",
            ]


@dataclass
class TorchSpec:
    def __init__(self, **kwargs: Any) -> None:
        self.domain_info = ("model", (1, 1))
        self.fields = ["model_state_dict"]
        if kwargs["optimizer"]:
            self.fields.append("optimizer_state_dict")
        if kwargs["model_info"]:
            self.fields.extend(kwargs["model_info"].keys())


@dataclass
class SklearnSpec:
    def __init__(self) -> None:
        self.domain_info = ("model", (1, 1))
        self.fields = ["model_params"]


@dataclass
class TensorBoardSpec:
    def __init__(self, **kwargs: Any) -> None:
        self.domain_info = ("tensorboard", (1, 1))
        self.fields = [kwargs["key"]]


@unique
class ModelFileProperties(Enum):
    """
    Enum Class that contains all model array file properties.
    """

    TILEDB_ML_MODEL_ML_FRAMEWORK = "TILEDB_ML_MODEL_ML_FRAMEWORK"
    TILEDB_ML_MODEL_ML_FRAMEWORK_VERSION = "TILEDB_ML_MODEL_ML_FRAMEWORK_VERSION"
    TILEDB_ML_MODEL_STAGE = "TILEDB_ML_MODEL_STAGE"
    TILEDB_ML_MODEL_PYTHON_VERSION = "TILEDB_ML_MODEL_PYTHON_VERSION"
    TILEDB_ML_MODEL_PREVIEW = "TILEDB_ML_MODEL_PREVIEW"


@unique
class TensorboardFileProperties(Enum):
    """
    Enum Class that contains all model array file properties.
    """

    TILEDB_ML_MODEL_TENSORBOARD_FRAMEWORK = "TILEDB_ML_MODEL_TENSORBOARD_FRAMEWORK"
    TILEDB_ML_MODEL_TENSORBOARD_VERSION = "TILEDB_ML_MODEL_TENSORBOARD_VERSION"
