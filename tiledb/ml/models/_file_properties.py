from enum import Enum, unique


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
