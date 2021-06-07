"""Tests for TileDB Tensorflow Keras model save and load."""

import os
import tiledb
import numpy as np
import pytest
import tempfile

# import tensorflow as tf
from tensorflow.python.keras.backend import batch_get_value

# from tensorflow.python.keras import layers
from tensorflow.python import keras
from tensorflow.python.feature_column import feature_column_lib
from tensorflow.python.framework import sparse_tensor

# from tensorflow.python.keras import losses
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.feature_column import dense_features
from tensorflow.python.keras.feature_column import sequence_feature_column as ksfc

# from tensorflow.python.platform import test

from tiledb.ml.models.tensorflow import TensorflowTileDB

# Suppress all Tensorflow messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


#
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


@pytest.fixture
def temp_uri():
    """
    Returns a temporary directory instance
    """
    return tempfile.TemporaryDirectory()


def test_load_tiledb_error_with_wrong_uri():
    tiledb_model_obj = TensorflowTileDB(uri="dummy_uri")
    with pytest.raises(tiledb.TileDBError):
        tiledb_model_obj.load(compile_model=False)


@pytest.mark.parametrize(
    "loss,optimizer,metrics",
    [(None, None, None), ("binary_crossentropy", "rmsprop", "accuracy")],
)
@pytest.mark.parametrize(
    "api",
    [
        testing_utils.get_small_sequential_mlp,
        testing_utils.get_small_functional_mlp,
        testing_utils.get_small_subclass_mlp,
    ],
)
def test_save_model_to_tiledb_array(temp_uri, api, loss, optimizer, metrics):
    with temp_uri as temp_dir:
        model = (
            api(num_hidden=1, num_classes=2, input_dim=3)
            if api != testing_utils.get_small_subclass_mlp
            else api(num_hidden=1, num_classes=2)
        )

        tiledb_uri = os.path.join(temp_dir, "model_array")

        # Compiles the model if optimizer is present
        if optimizer:
            model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])
        tiledb_model_obj = TensorflowTileDB(uri=tiledb_uri)

        if api != testing_utils.get_small_subclass_mlp:
            tiledb_model_obj.save(
                model=model, include_optimizer=True if optimizer else False
            )
            assert tiledb.array_exists(tiledb_uri)
        else:
            with pytest.raises(NotImplementedError):
                tiledb_model_obj.save(
                    model=model, include_optimizer=True if optimizer else False
                )


@pytest.mark.parametrize(
    "loss,optimizer,metrics",
    [(None, None, None), ("binary_crossentropy", "rmsprop", "accuracy")],
)
@pytest.mark.parametrize(
    "api",
    [
        testing_utils.get_small_sequential_mlp,
        testing_utils.get_small_functional_mlp,
        testing_utils.get_small_subclass_mlp,
    ],
)
def test_save_model_to_tiledb_array_predictions(
    temp_uri, api, loss, optimizer, metrics
):
    with temp_uri as temp_dir:
        model = (
            api(num_hidden=1, num_classes=2, input_dim=3)
            if api != testing_utils.get_small_subclass_mlp
            else api(num_hidden=1, num_classes=2)
        )

        tiledb_uri = os.path.join(temp_dir, "model_array")

        # Compiles the model if optimizer is present
        if optimizer:
            model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])
        tiledb_model_obj = TensorflowTileDB(uri=tiledb_uri)

        if api != testing_utils.get_small_subclass_mlp:
            tiledb_model_obj.save(
                model=model, include_optimizer=True if optimizer else False
            )
            loaded_model = tiledb_model_obj.load(compile_model=False)
            data = np.random.rand(100, 3)

            # Assert model predictions are equal
            np.testing.assert_array_equal(
                loaded_model.predict(data), model.predict(data)
            )
            # assert all([a == b for a, b in zip(loaded_model.predict(data), model.predict(data))])
        else:
            with pytest.raises(NotImplementedError):
                tiledb_model_obj.save(
                    model=model, include_optimizer=True if optimizer else False
                )


@pytest.mark.parametrize(
    "loss,optimizer,metrics",
    [(None, None, None), ("binary_crossentropy", "rmsprop", "accuracy")],
)
@pytest.mark.parametrize(
    "api",
    [
        testing_utils.get_small_sequential_mlp,
        testing_utils.get_small_functional_mlp,
        testing_utils.get_small_subclass_mlp,
    ],
)
def test_save_model_to_tiledb_array_weights(temp_uri, api, loss, optimizer, metrics):
    with temp_uri as temp_dir:
        model = (
            api(num_hidden=1, num_classes=2, input_dim=3)
            if api != testing_utils.get_small_subclass_mlp
            else api(num_hidden=1, num_classes=2)
        )

        tiledb_uri = os.path.join(temp_dir, "model_array")

        # Compiles the model if optimizer is present
        if optimizer:
            model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])
        tiledb_model_obj = TensorflowTileDB(uri=tiledb_uri)

        if api != testing_utils.get_small_subclass_mlp:
            tiledb_model_obj.save(
                model=model, include_optimizer=True if optimizer else False
            )
            loaded_model = tiledb_model_obj.load(
                compile_model=True if optimizer else False
            )
            data = np.random.rand(100, 3)

            if optimizer:
                model_opt_weights = batch_get_value(getattr(model.optimizer, "weights"))
                loaded_opt_weights = batch_get_value(
                    getattr(loaded_model.optimizer, "weights")
                )

                # Assert optimizer weights are equal
                for weight_model, weight_loaded_model in zip(
                    model_opt_weights, loaded_opt_weights
                ):
                    np.testing.assert_array_equal(weight_model, weight_loaded_model)

                # Assert model predictions are equal
                np.testing.assert_array_equal(
                    loaded_model.predict(data), model.predict(data)
                )
        else:
            with pytest.raises(NotImplementedError):
                tiledb_model_obj.save(
                    model=model, include_optimizer=True if optimizer else False
                )


@pytest.mark.parametrize(
    "loss,optimizer,metrics",
    [
        ("binary_crossentropy", "rmsprop", "accuracy"),
        (keras.losses.MSE, "rmsprop", keras.metrics.categorical_accuracy),
    ],
)
def test_save_load_with_dense_features(temp_uri, loss, optimizer, metrics):
    with temp_uri as temp_dir:
        cols = [
            feature_column_lib.numeric_column("a"),
            feature_column_lib.indicator_column(
                feature_column_lib.categorical_column_with_vocabulary_list(
                    "b", ["one", "two"]
                )
            ),
        ]
        input_layers = {
            "a": keras.layers.Input(shape=(1,), name="a"),
            "b": keras.layers.Input(shape=(1,), name="b", dtype="string"),
        }

        fc_layer = dense_features.DenseFeatures(cols)(input_layers)
        output = keras.layers.Dense(10)(fc_layer)

        model = keras.models.Model(input_layers, output)

        model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=[metrics],
        )

        tiledb_uri = os.path.join(temp_dir, "model_array")
        tiledb_model_obj = TensorflowTileDB(uri=tiledb_uri)
        tiledb_model_obj.save(model=model, include_optimizer=True)
        loaded_model = tiledb_model_obj.load(compile_model=True)

        model_opt_weights = batch_get_value(getattr(model.optimizer, "weights"))
        loaded_opt_weights = batch_get_value(getattr(loaded_model.optimizer, "weights"))

        # Assert optimizer weights are equal
        for weight_model, weight_loaded_model in zip(
            model_opt_weights, loaded_opt_weights
        ):
            np.testing.assert_array_equal(weight_model, weight_loaded_model)

        inputs_a = np.arange(10).reshape(10, 1)
        inputs_b = np.arange(10).reshape(10, 1).astype("str")

        # Assert model predictions are equal
        np.testing.assert_array_equal(
            loaded_model.predict({"a": inputs_a, "b": inputs_b}),
            model.predict({"a": inputs_a, "b": inputs_b}),
        )


@pytest.mark.parametrize(
    "loss,optimizer,metrics",
    [
        ("binary_crossentropy", "rmsprop", "accuracy"),
        (keras.losses.MSE, "rmsprop", keras.metrics.categorical_accuracy),
    ],
)
def test_save_load_with_sequence_features(temp_uri, loss, optimizer, metrics):
    with temp_uri as temp_dir:
        cols = [
            feature_column_lib.sequence_numeric_column("a"),
            feature_column_lib.indicator_column(
                feature_column_lib.sequence_categorical_column_with_vocabulary_list(
                    "b", ["one", "two"]
                )
            ),
        ]
        input_layers = {
            "a": keras.layers.Input(shape=(None, 1), sparse=True, name="a"),
            "b": keras.layers.Input(
                shape=(None, 1), sparse=True, name="b", dtype="string"
            ),
        }

        fc_layer, _ = ksfc.SequenceFeatures(cols)(input_layers)
        x = keras.layers.GRU(32)(fc_layer)
        output = keras.layers.Dense(10)(x)

        model = keras.models.Model(input_layers, output)

        model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=[metrics],
        )

        tiledb_uri = os.path.join(temp_dir, "model_array")
        tiledb_model_obj = TensorflowTileDB(uri=tiledb_uri)
        tiledb_model_obj.save(model=model, include_optimizer=True)
        loaded_model = tiledb_model_obj.load(compile_model=True)

        model_opt_weights = batch_get_value(getattr(model.optimizer, "weights"))
        loaded_opt_weights = batch_get_value(getattr(loaded_model.optimizer, "weights"))

        # Assert optimizer weights are equal
        for weight_model, weight_loaded_model in zip(
            model_opt_weights, loaded_opt_weights
        ):
            np.testing.assert_array_equal(weight_model, weight_loaded_model)

        batch_size = 10
        timesteps = 1

        values_a = np.arange(10, dtype=np.float32)
        indices_a = np.zeros((10, 3), dtype=np.int64)
        indices_a[:, 0] = np.arange(10)
        inputs_a = sparse_tensor.SparseTensor(
            indices_a, values_a, (batch_size, timesteps, 1)
        )

        values_b = np.zeros(10, dtype=np.str)
        indices_b = np.zeros((10, 3), dtype=np.int64)
        indices_b[:, 0] = np.arange(10)
        inputs_b = sparse_tensor.SparseTensor(
            indices_b, values_b, (batch_size, timesteps, 1)
        )

        # Assert model predictions are equal
        np.testing.assert_array_equal(
            loaded_model.predict({"a": inputs_a, "b": inputs_b}, steps=1),
            model.predict({"a": inputs_a, "b": inputs_b}, steps=1),
        )


@pytest.mark.parametrize(
    "loss,optimizer,metrics",
    [
        (
            keras.losses.SparseCategoricalCrossentropy(),
            optimizers.gradient_descent_v2.SGD(),
            keras.metrics.SparseCategoricalCrossentropy(),
        ),
    ],
)
def test_functional_model_save_load_with_custom_loss_and_metric(
    temp_uri, loss, optimizer, metrics
):
    with temp_uri as temp_dir:
        inputs = keras.Input(shape=(4,))
        x = keras.layers.Dense(8, activation="relu")(inputs)
        outputs = keras.layers.Dense(3, activation="softmax")(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        custom_loss = keras.layers.Lambda(lambda x: keras.backend.sum(x * x))(x)
        model.add_loss(custom_loss)
        model.add_metric(custom_loss, aggregation="mean", name="custom_loss")

        model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=[metrics],
        )

        data_x = np.random.normal(size=(32, 4))
        data_y = np.random.randint(0, 3, size=32)
        model.train_on_batch(data_x, data_y)

        tiledb_uri = os.path.join(temp_dir, "model_array")
        tiledb_model_obj = TensorflowTileDB(uri=tiledb_uri)
        tiledb_model_obj.save(model=model, include_optimizer=True)
        loaded_model = tiledb_model_obj.load(compile_model=True)

        # Assert all evaluation results are the same.
        assert all(
            [
                a == pytest.approx(b, 1e-9)
                for a, b in zip(
                    model.evaluate(data_x, data_y),
                    loaded_model.evaluate(data_x, data_y),
                )
            ]
        )

        # Assert model predictions are equal
        np.testing.assert_array_equal(
            loaded_model.predict(data_x), model.predict(data_x)
        )


# class TestSaveLoadTileDBModel(test.TestCase):
#
#
#
#     @testing_utils.run_v2_only
#     def test_save_load_for_rnn_layers(self):
#         inputs = keras.Input([10, 10], name="train_input")
#         rnn_layers = [
#             keras.layers.LSTMCell(size, recurrent_dropout=0, name="rnn_cell%d" % i)
#             for i, size in enumerate([32, 32])
#         ]
#         rnn_output = keras.layers.RNN(
#             rnn_layers, return_sequences=True, name="rnn_layer"
#         )(inputs)
#         pred_feat = keras.layers.Dense(10, name="prediction_features")(rnn_output)
#         pred = keras.layers.Softmax()(pred_feat)
#         model = keras.Model(inputs=[inputs], outputs=[pred, pred_feat])
#
#         tiledb_uri = os.path.join(self.get_temp_dir(), "model_array")
#         tiledb_model_obj = TensorflowTileDB(uri=tiledb_uri)
#         tiledb_model_obj.save(model=model, include_optimizer=False)
#         loaded_model = tiledb_model_obj.load(compile_model=False)
#
#         data = np.random.rand(50, 10, 10)
#
#         # Assert model predictions are equal
#         np.testing.assert_array_equal(loaded_model.predict(data), model.predict(data))
#
#     @testing_utils.run_v2_only
#     def test_sequential_model_save_load_without_input_shape(self):
#         model = keras.models.Sequential()
#         model.add(keras.layers.Dense(2))
#         model.add(keras.layers.RepeatVector(3))
#         model.add(keras.layers.TimeDistributed(keras.layers.Dense(3)))
#         model.compile(
#             loss=keras.losses.MSE,
#             optimizer="rmsprop",
#             metrics=[
#                 keras.metrics.categorical_accuracy,
#                 keras.metrics.CategoricalAccuracy(name="cat_acc"),
#             ],
#             weighted_metrics=[
#                 keras.metrics.categorical_accuracy,
#                 keras.metrics.CategoricalAccuracy(name="cat_acc2"),
#             ],
#             sample_weight_mode="temporal",
#         )
#         data_x = np.random.random((1, 3))
#         data_y = np.random.random((1, 3, 3))
#         model.train_on_batch(data_x, data_y)
#
#         tiledb_uri = os.path.join(self.get_temp_dir(), "model_array")
#         tiledb_model_obj = TensorflowTileDB(uri=tiledb_uri)
#         tiledb_model_obj.save(model=model, include_optimizer=True)
#         loaded_model = tiledb_model_obj.load(compile_model=True)
#
#         # Assert model predictions are equal
#         np.testing.assert_array_equal(
#             loaded_model.predict(data_x), model.predict(data_x)
#         )
