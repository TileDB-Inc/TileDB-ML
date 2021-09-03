"""Tests for TileDB Tensorflow Keras model save and load."""

import os
import tiledb
import numpy as np
import pytest
import platform
import io
import pickle

import tensorflow as tf
from tensorflow.python.keras.backend import batch_get_value

from tensorflow.python import keras
from tensorflow.python.feature_column import feature_column_lib
from tensorflow.python.framework import sparse_tensor

from tensorflow.python.keras import optimizers
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.feature_column import dense_features
from tensorflow.python.keras.feature_column import sequence_feature_column as ksfc

from tiledb.ml.models.tensorflow_keras import TensorflowKerasTileDBModel

# Suppress all Tensorflow messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class ConfigSubclassModel(keras.Model):
    def __init__(self, hidden_units):
        super(ConfigSubclassModel, self).__init__()
        self.hidden_units = hidden_units
        self.dense_layers = [keras.layers.Dense(u) for u in hidden_units]

    def call(self, inputs):
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        return x

    def get_config(self):
        return {"hidden_units": self.hidden_units}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def test_load_tiledb_error_with_wrong_uri():
    tiledb_model_obj = TensorflowKerasTileDBModel(uri="dummy_uri")
    with pytest.raises(tiledb.TileDBError):
        tiledb_model_obj.load(compile_model=False)


@pytest.mark.parametrize(
    "loss,optimizer,metrics",
    [
        (None, None, None),
        ("binary_crossentropy", "rmsprop", "accuracy"),
        (keras.losses.MSE, "rmsprop", keras.metrics.categorical_accuracy),
        (
            keras.losses.SparseCategoricalCrossentropy(),
            optimizers.gradient_descent_v2.SGD(),
            keras.metrics.SparseCategoricalCrossentropy(),
        ),
    ],
)
@pytest.mark.parametrize(
    "api",
    [
        testing_utils.get_small_sequential_mlp,
        testing_utils.get_small_functional_mlp,
        ConfigSubclassModel,
    ],
)
class TestTensorflowKerasModel:
    def test_save_model_to_tiledb_array(self, tmpdir, api, loss, optimizer, metrics):
        model = (
            api(num_hidden=1, num_classes=2, input_dim=3)
            if api != ConfigSubclassModel
            else api(hidden_units=[16, 16, 10])
        )

        tiledb_uri = os.path.join(tmpdir, "model_array")

        # Compiles the model if optimizer is present
        if optimizer:
            model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])

        if not model.built:
            model.build(tuple(np.random.randint(20, size=2)))
        tiledb_model_obj = TensorflowKerasTileDBModel(uri=tiledb_uri, model=model)
        tiledb_model_obj.save(include_optimizer=True if optimizer else False)
        assert tiledb.array_exists(tiledb_uri)

    def test_save_model_to_tiledb_array_predictions(
        self, tmpdir, api, loss, optimizer, metrics
    ):
        model = (
            api(num_hidden=1, num_classes=2, input_dim=3)
            if api != ConfigSubclassModel
            else api(hidden_units=[16, 16, 10])
        )

        tiledb_uri = os.path.join(tmpdir, "model_array")

        # Compiles the model if optimizer is present
        if optimizer:
            model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])

        input_shape = tuple(np.random.randint(20, size=2))
        if not model.built:
            model.build(input_shape)
        tiledb_model_obj = TensorflowKerasTileDBModel(uri=tiledb_uri, model=model)
        tiledb_model_obj.save(include_optimizer=True if optimizer else False)

        loaded_model = (
            tiledb_model_obj.load(
                compile_model=False,
                custom_objects={"ConfigSubclassModel": ConfigSubclassModel},
                input_shape=input_shape,
            )
            if api == ConfigSubclassModel
            else tiledb_model_obj.load(compile_model=False)
        )

        data = np.random.rand(100, input_shape[-1] if api == ConfigSubclassModel else 3)

        # Assert model predictions are equal
        np.testing.assert_array_equal(loaded_model.predict(data), model.predict(data))

    def test_save_model_to_tiledb_array_weights(
        self, tmpdir, api, loss, optimizer, metrics
    ):
        model = (
            api(num_hidden=1, num_classes=2, input_dim=3)
            if api != ConfigSubclassModel
            else api(hidden_units=[16, 16, 10])
        )

        tiledb_uri = os.path.join(tmpdir, "model_array")

        # Compiles the model if optimizer is present
        if optimizer:
            model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])

        input_shape = tuple(np.random.randint(20, size=2))
        if not model.built:
            model.build(input_shape)
        tiledb_model_obj = TensorflowKerasTileDBModel(uri=tiledb_uri, model=model)

        tiledb_model_obj.save(include_optimizer=True if optimizer else False)
        loaded_model = (
            tiledb_model_obj.load(compile_model=True if optimizer else False)
            if api != ConfigSubclassModel
            else tiledb_model_obj.load(
                compile_model=True if optimizer else False,
                custom_objects={"ConfigSubclassModel": ConfigSubclassModel},
                input_shape=input_shape,
            )
        )

        data = np.random.rand(100, input_shape[-1] if api == ConfigSubclassModel else 3)

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

    def test_save_load_with_dense_features(self, tmpdir, api, loss, optimizer, metrics):
        if optimizer is None:
            pytest.skip()
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

        tiledb_uri = os.path.join(tmpdir, "model_array")
        tiledb_model_obj = TensorflowKerasTileDBModel(uri=tiledb_uri, model=model)
        tiledb_model_obj.save(include_optimizer=True)
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

    def test_save_load_with_sequence_features(
        self, tmpdir, api, loss, optimizer, metrics
    ):
        if optimizer is None:
            pytest.skip()

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

        tiledb_uri = os.path.join(tmpdir, "model_array")
        tiledb_model_obj = TensorflowKerasTileDBModel(uri=tiledb_uri, model=model)
        tiledb_model_obj.save(include_optimizer=True)
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

    def test_functional_model_save_load_with_custom_loss_and_metric(
        self, tmpdir, api, loss, optimizer, metrics
    ):
        if optimizer is None or loss != keras.losses.SparseCategoricalCrossentropy():
            pytest.skip()
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

        tiledb_uri = os.path.join(tmpdir, "model_array")
        tiledb_model_obj = TensorflowKerasTileDBModel(uri=tiledb_uri, model=model)
        tiledb_model_obj.save(include_optimizer=True)
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

    def test_save_load_for_rnn_layers(self, tmpdir, api, loss, optimizer, metrics):
        inputs = keras.Input([10, 10], name="train_input")
        rnn_layers = [
            keras.layers.LSTMCell(size, recurrent_dropout=0, name="rnn_cell%d" % i)
            for i, size in enumerate([32, 32])
        ]
        rnn_output = keras.layers.RNN(
            rnn_layers, return_sequences=True, name="rnn_layer"
        )(inputs)
        pred_feat = keras.layers.Dense(10, name="prediction_features")(rnn_output)
        pred = keras.layers.Softmax()(pred_feat)
        model = keras.Model(inputs=[inputs], outputs=[pred, pred_feat])

        tiledb_uri = os.path.join(tmpdir, "model_array")
        tiledb_model_obj = TensorflowKerasTileDBModel(uri=tiledb_uri, model=model)
        tiledb_model_obj.save(include_optimizer=False)
        loaded_model = tiledb_model_obj.load(compile_model=False)

        data = np.random.rand(50, 10, 10)

        # Assert model predictions are equal
        np.testing.assert_array_equal(loaded_model.predict(data), model.predict(data))

    def test_sequential_model_save_load_without_input_shape(
        self, tmpdir, api, loss, optimizer, metrics
    ):
        if optimizer is None or loss != keras.losses.MSE:
            pytest.skip()
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(2))
        model.add(keras.layers.RepeatVector(3))
        model.add(keras.layers.TimeDistributed(keras.layers.Dense(3)))
        model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=metrics,
            weighted_metrics=metrics,
            sample_weight_mode="temporal",
        )
        data_x = np.random.random((1, 3))
        data_y = np.random.random((1, 3, 3))
        model.train_on_batch(data_x, data_y)

        tiledb_uri = os.path.join(tmpdir, "model_array")
        tiledb_model_obj = TensorflowKerasTileDBModel(uri=tiledb_uri, model=model)
        tiledb_model_obj.save(include_optimizer=True)
        loaded_model = tiledb_model_obj.load(compile_model=True)

        # Assert model predictions are equal
        np.testing.assert_array_equal(
            loaded_model.predict(data_x), model.predict(data_x)
        )

    def test_preview(self, tmpdir, api, loss, optimizer, metrics):
        model = (
            api(num_hidden=1, num_classes=2, input_dim=3)
            if api != ConfigSubclassModel
            else api(hidden_units=[16, 16, 10])
        )

        tiledb_uri = os.path.join(tmpdir, "model_array")

        # Compiles the model if optimizer is present
        if optimizer:
            model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])

        # With model given as argument
        if model.built:
            tiledb_model_obj = TensorflowKerasTileDBModel(uri=tiledb_uri, model=model)
            s = io.StringIO()
            model.summary(print_fn=lambda x: s.write(x + "\n"))
            model_summary = s.getvalue()
            assert tiledb_model_obj.preview() == model_summary
        else:
            # Model should be built before preview it
            with pytest.raises(ValueError):
                tiledb_model_obj = TensorflowKerasTileDBModel(
                    uri=tiledb_uri, model=model
                )
                tiledb_model_obj.preview()

        # When model is None then preview returns empty string
        tiledb_model_obj_none = TensorflowKerasTileDBModel(uri=tiledb_uri, model=None)
        assert tiledb_model_obj_none.preview() == ""

    def test_get_cloud_uri(self, tmpdir, api, loss, optimizer, metrics, mocker):
        model = (
            api(num_hidden=1, num_classes=2, input_dim=3)
            if api != ConfigSubclassModel
            else api(hidden_units=[16, 16, 10])
        )

        tiledb_uri = os.path.join(tmpdir, "model_array")

        # Compiles the model if optimizer is present
        if optimizer:
            model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])

        mocker.patch("tiledb.ml._cloud_utils.get_s3_prefix", return_value=None)
        # With model given as argument
        input_shape = tuple(np.random.randint(20, size=2))
        if not model.built:
            # Subclass case
            model.build(input_shape)
        tiledb_model_obj = TensorflowKerasTileDBModel(uri=tiledb_uri, model=model)
        with pytest.raises(ValueError):
            tiledb_model_obj.get_cloud_uri(tiledb_uri)

        mocker.patch("tiledb.ml._cloud_utils.get_s3_prefix", return_value="bar")
        actual = tiledb_model_obj.get_cloud_uri(tiledb_uri)
        expected = "tiledb://{}/{}".format(
            tiledb_model_obj.namespace, os.path.join("bar", tiledb_uri)
        )
        assert actual == expected


def test_file_properties(tmpdir):
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=(10, 10)))

    # Get model summary in a string
    s = io.StringIO()
    model.summary(print_fn=lambda x: s.write(x + "\n"))
    model_summary = s.getvalue()

    tiledb_array = os.path.join(tmpdir, "model_array")
    tiledb_obj = TensorflowKerasTileDBModel(uri=tiledb_array, model=model)
    tiledb_obj.save()

    assert (
        tiledb_obj._file_properties["TILEDB_ML_MODEL_ML_FRAMEWORK"]
        == "TENSORFLOW KERAS"
    )
    assert tiledb_obj._file_properties["TILEDB_ML_MODEL_STAGE"] == "STAGING"
    assert (
        tiledb_obj._file_properties["TILEDB_ML_MODEL_PYTHON_VERSION"]
        == platform.python_version()
    )
    assert (
        tiledb_obj._file_properties["TILEDB_ML_MODEL_ML_FRAMEWORK_VERSION"]
        == tf.__version__
    )
    assert tiledb_obj._file_properties["TILEDB_ML_MODEL_PREVIEW"] == model_summary


def test_file_properties_in_tiledb_cloud_case(tmpdir, mocker):
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=(10, 10)))

    # Get model summary in a string
    s = io.StringIO()
    model.summary(print_fn=lambda x: s.write(x + "\n"))
    model_summary = s.getvalue()

    tiledb_array = os.path.join(tmpdir, "model_array")
    mocker.patch(
        "tiledb.ml.models.base.TileDBModel.get_cloud_uri", return_value=tiledb_array
    )
    mocker.patch("tiledb.ml._cloud_utils.update_file_properties")

    tiledb_array = os.path.join(tmpdir, "model_array")
    tiledb_obj = TensorflowKerasTileDBModel(
        uri=tiledb_array, namespace="test_namespace", model=model
    )
    tiledb_obj.save()

    assert (
        tiledb_obj._file_properties["TILEDB_ML_MODEL_ML_FRAMEWORK"]
        == "TENSORFLOW KERAS"
    )
    assert tiledb_obj._file_properties["TILEDB_ML_MODEL_STAGE"] == "STAGING"
    assert (
        tiledb_obj._file_properties["TILEDB_ML_MODEL_PYTHON_VERSION"]
        == platform.python_version()
    )
    assert (
        tiledb_obj._file_properties["TILEDB_ML_MODEL_ML_FRAMEWORK_VERSION"]
        == tf.__version__
    )
    assert tiledb_obj._file_properties["TILEDB_ML_MODEL_PREVIEW"] == model_summary


def test_exception_raise_file_property_in_meta_error(tmpdir):
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=(10, 10)))
    tiledb_array = os.path.join(tmpdir, "model_array")
    tiledb_obj = TensorflowKerasTileDBModel(uri=tiledb_array, model=model)
    with pytest.raises(ValueError):
        tiledb_obj.save(
            meta={"TILEDB_ML_MODEL_ML_FRAMEWORK": "TILEDB_ML_MODEL_ML_FRAMEWORK"},
        )


def test_serialize_model_weights(tmpdir):
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=(10, 10)))
    tiledb_array = os.path.join(tmpdir, "model_array")
    tiledb_obj = TensorflowKerasTileDBModel(uri=tiledb_array, model=model)
    assert tiledb_obj._serialize_model_weights() == pickle.dumps(
        model.get_weights(), protocol=4
    )
