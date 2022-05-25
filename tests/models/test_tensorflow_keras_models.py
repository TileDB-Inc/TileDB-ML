"""Tests for TileDB Tensorflow Keras model save and load."""

import io
import os
import pickle
import platform
import shutil

import numpy as np
import pytest
import tensorflow as tf

import tiledb
from tiledb.ml.models.tensorflow_keras import (
    TensorflowKerasTileDBModel,
    tf_keras_is_keras,
)

if tf_keras_is_keras:
    from keras import testing_utils
else:
    from tensorflow.python.keras import testing_utils


# Suppress all Tensorflow messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

batch_get_value = tf.keras.backend.batch_get_value


class ConfigSubclassModel(tf.keras.Model):
    def __init__(self, hidden_units):
        super(ConfigSubclassModel, self).__init__()
        self.hidden_units = hidden_units
        self.dense_layers = [tf.keras.layers.Dense(u) for u in hidden_units]

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


loss_optimizer_metrics = pytest.mark.parametrize(
    "loss,optimizer,metrics",
    [
        (None, None, None),
        ("binary_crossentropy", "rmsprop", "accuracy"),
        ("mean_squared_error", "rmsprop", "categorical_accuracy"),
        ("sparse_categorical_crossentropy", "sgd", "sparse_categorical_crossentropy"),
    ],
)

api = pytest.mark.parametrize(
    "api",
    [
        testing_utils.get_small_sequential_mlp,
        testing_utils.get_small_functional_mlp,
        ConfigSubclassModel,
    ],
)


class TestTensorflowKerasModel:
    @api
    @loss_optimizer_metrics
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

    @api
    @loss_optimizer_metrics
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

    @api
    @loss_optimizer_metrics
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
            model_opt_weights = batch_get_value(model.optimizer.weights)
            loaded_opt_weights = batch_get_value(loaded_model.optimizer.weights)

            # Assert optimizer weights are equal
            for weight_model, weight_loaded_model in zip(
                model_opt_weights, loaded_opt_weights
            ):
                np.testing.assert_array_equal(weight_model, weight_loaded_model)

            # Assert model predictions are equal
            np.testing.assert_array_equal(
                loaded_model.predict(data), model.predict(data)
            )

    @loss_optimizer_metrics
    def test_save_load_with_dense_features(self, tmpdir, loss, optimizer, metrics):
        if optimizer is None:
            pytest.skip()
        cols = [
            tf.feature_column.numeric_column("a"),
            tf.feature_column.indicator_column(
                tf.feature_column.categorical_column_with_vocabulary_list(
                    "b", ["one", "two"]
                )
            ),
        ]
        input_layers = {
            "a": tf.keras.layers.Input(shape=(1,), name="a"),
            "b": tf.keras.layers.Input(shape=(1,), name="b", dtype="string"),
        }

        fc_layer = tf.keras.layers.DenseFeatures(cols)(input_layers)
        output = tf.keras.layers.Dense(10)(fc_layer)

        model = tf.keras.Model(input_layers, output)

        model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])

        tiledb_uri = os.path.join(tmpdir, "model_array")
        tiledb_model_obj = TensorflowKerasTileDBModel(uri=tiledb_uri, model=model)
        tiledb_model_obj.save(include_optimizer=True)
        loaded_model = tiledb_model_obj.load(compile_model=True)

        model_opt_weights = batch_get_value(model.optimizer.weights)
        loaded_opt_weights = batch_get_value(loaded_model.optimizer.weights)

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

    @loss_optimizer_metrics
    def test_save_load_with_sequence_features(self, tmpdir, loss, optimizer, metrics):
        if optimizer is None:
            pytest.skip()

        cols = [
            tf.feature_column.sequence_numeric_column("a"),
            tf.feature_column.indicator_column(
                tf.feature_column.sequence_categorical_column_with_vocabulary_list(
                    "b", ["one", "two"]
                )
            ),
        ]
        input_layers = {
            "a": tf.keras.layers.Input(shape=(None, 1), sparse=True, name="a"),
            "b": tf.keras.layers.Input(
                shape=(None, 1), sparse=True, name="b", dtype="string"
            ),
        }

        fc_layer, _ = tf.keras.experimental.SequenceFeatures(cols)(input_layers)
        x = tf.keras.layers.GRU(32)(fc_layer)
        output = tf.keras.layers.Dense(10)(x)

        model = tf.keras.Model(input_layers, output)

        model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])

        tiledb_uri = os.path.join(tmpdir, "model_array")
        tiledb_model_obj = TensorflowKerasTileDBModel(uri=tiledb_uri, model=model)
        tiledb_model_obj.save(include_optimizer=True)
        loaded_model = tiledb_model_obj.load(compile_model=True)

        model_opt_weights = batch_get_value(model.optimizer.weights)
        loaded_opt_weights = batch_get_value(loaded_model.optimizer.weights)

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
        inputs_a = tf.SparseTensor(indices_a, values_a, (batch_size, timesteps, 1))

        values_b = np.zeros(10, dtype=np.str)
        indices_b = np.zeros((10, 3), dtype=np.int64)
        indices_b[:, 0] = np.arange(10)
        inputs_b = tf.SparseTensor(indices_b, values_b, (batch_size, timesteps, 1))

        # Assert model predictions are equal
        np.testing.assert_array_equal(
            loaded_model.predict({"a": inputs_a, "b": inputs_b}, steps=1),
            model.predict({"a": inputs_a, "b": inputs_b}, steps=1),
        )

    def test_functional_model_save_load_with_custom_loss_and_metric(self, tmpdir):
        inputs = tf.keras.layers.Input(shape=(4,))
        x = tf.keras.layers.Dense(8, activation="relu")(inputs)
        outputs = tf.keras.layers.Dense(3, activation="softmax")(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        custom_loss = tf.keras.layers.Lambda(lambda x: tf.keras.backend.sum(x * x))(x)
        model.add_loss(custom_loss)
        model.add_metric(custom_loss, aggregation="mean", name="custom_loss")

        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="sgd",
            metrics=["sparse_categorical_crossentropy"],
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

    def test_save_load_for_rnn_layers(self, tmpdir):
        inputs = tf.keras.layers.Input([10, 10], name="train_input")
        rnn_layers = [
            tf.keras.layers.LSTMCell(size, recurrent_dropout=0, name="rnn_cell%d" % i)
            for i, size in enumerate([32, 32])
        ]
        rnn_output = tf.keras.layers.RNN(
            rnn_layers, return_sequences=True, name="rnn_layer"
        )(inputs)
        pred_feat = tf.keras.layers.Dense(10, name="prediction_features")(rnn_output)
        pred = tf.keras.layers.Softmax()(pred_feat)
        model = tf.keras.Model(inputs=[inputs], outputs=[pred, pred_feat])

        tiledb_uri = os.path.join(tmpdir, "model_array")
        tiledb_model_obj = TensorflowKerasTileDBModel(uri=tiledb_uri, model=model)
        tiledb_model_obj.save(include_optimizer=False)
        loaded_model = tiledb_model_obj.load(compile_model=False)

        data = np.random.rand(50, 10, 10)

        # Assert model predictions are equal
        np.testing.assert_array_equal(loaded_model.predict(data), model.predict(data))

    def test_sequential_model_save_load_without_input_shape(self, tmpdir):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(2))
        model.add(tf.keras.layers.RepeatVector(3))
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(3)))
        model.compile(
            loss="mean_squared_error",
            optimizer="rmsprop",
            metrics="categorical_accuracy",
            weighted_metrics="categorical_accuracy",
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

    @api
    @loss_optimizer_metrics
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


class TestTensorflowKerasModelCloud:
    def test_file_properties(self, tmpdir):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=(10, 10)))

        # Get model summary in a string
        s = io.StringIO()
        model.summary(print_fn=lambda x: s.write(x + "\n"))
        model_summary = s.getvalue()

        uri = os.path.join(tmpdir, "model_array")
        tiledb_obj = TensorflowKerasTileDBModel(uri=uri, model=model)

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

    def test_get_cloud_uri_call_for_models_on_tiledb_cloud(self, tmpdir, mocker):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=(10, 10)))
        uri = os.path.join(tmpdir, "model_array")

        mock_get_cloud_uri = mocker.patch(
            "tiledb.ml.models._base.get_cloud_uri", return_value=uri
        )

        _ = TensorflowKerasTileDBModel(uri=uri, namespace="test_namespace", model=model)

        mock_get_cloud_uri.assert_called_once_with(uri, "test_namespace")

    def test_get_s3_prefix_call_for_models_on_tiledb_cloud(self, tmpdir, mocker):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=(10, 10)))
        uri = os.path.join(tmpdir, "model_array")

        mock_get_s3_prefix = mocker.patch(
            "tiledb.ml.models._cloud_utils.get_s3_prefix", return_value="s3 prefix"
        )

        _ = TensorflowKerasTileDBModel(uri=uri, namespace="test_namespace", model=model)

        mock_get_s3_prefix.assert_called_once_with("test_namespace")

    def test_update_file_properties_call(self, tmpdir, mocker):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=(10, 10)))

        # Get model summary in a string
        s = io.StringIO()
        model.summary(print_fn=lambda x: s.write(x + "\n"))
        model_summary = s.getvalue()

        uri = os.path.join(tmpdir, "model_array")

        mocker.patch("tiledb.ml.models._base.get_cloud_uri", return_value=uri)

        tiledb_obj = TensorflowKerasTileDBModel(
            uri=uri, namespace="test_namespace", model=model
        )

        mock_update_file_properties = mocker.patch(
            "tiledb.ml.models.tensorflow_keras.update_file_properties",
            return_value=None,
        )
        mocker.patch(
            "tiledb.ml.models.tensorflow_keras.TensorflowKerasTileDBModel._write_array"
        )

        tiledb_obj.save()

        file_properties_dict = {
            "TILEDB_ML_MODEL_ML_FRAMEWORK": "TENSORFLOW KERAS",
            "TILEDB_ML_MODEL_ML_FRAMEWORK_VERSION": tf.__version__,
            "TILEDB_ML_MODEL_STAGE": "STAGING",
            "TILEDB_ML_MODEL_PYTHON_VERSION": platform.python_version(),
            "TILEDB_ML_MODEL_PREVIEW": model_summary,
        }

        mock_update_file_properties.assert_called_once_with(uri, file_properties_dict)

    def test_exception_raise_file_property_in_meta_error(self, tmpdir):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=(10, 10)))
        tiledb_array = os.path.join(tmpdir, "model_array")
        tiledb_obj = TensorflowKerasTileDBModel(uri=tiledb_array, model=model)
        with pytest.raises(ValueError) as ex:
            tiledb_obj.save(
                meta={"TILEDB_ML_MODEL_ML_FRAMEWORK": "TILEDB_ML_MODEL_ML_FRAMEWORK"},
            )

        assert "Please avoid using file property key names as metadata keys!" in str(
            ex.value
        )

    def test_tensorboard_callback_meta(self, tmpdir):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=(10, 10)))
        tiledb_array = os.path.join(tmpdir, "model_array")
        tiledb_obj = TensorflowKerasTileDBModel(uri=tiledb_array, model=model)

        cb = [tf.keras.callbacks.TensorBoard(log_dir=tmpdir)]

        os.mkdir(os.path.join(tmpdir, "train"))
        with open(os.path.join(tmpdir, "train", "foo_tfevents_1"), "wb") as f:
            f.write(b"test_bytes_1")
        with open(os.path.join(tmpdir, "train", "bar_tfevents_2"), "wb") as f:
            f.write(b"test_bytes_2")

        tiledb_obj.save(include_callbacks=cb)
        with tiledb.open(tiledb_array) as A:
            assert pickle.loads(A.meta["__TENSORBOARD__"]) == {
                os.path.join(tmpdir, "train", "foo_tfevents_1"): b"test_bytes_1",
                os.path.join(tmpdir, "train", "bar_tfevents_2"): b"test_bytes_2",
            }
        shutil.rmtree(os.path.join(tmpdir, "train"))

        # Loading the event data should create local files
        tiledb_obj.load_tensorboard()
        with open(os.path.join(tmpdir, "train", "foo_tfevents_1"), "rb") as f:
            assert f.read() == b"test_bytes_1"
        with open(os.path.join(tmpdir, "train", "bar_tfevents_2"), "rb") as f:
            assert f.read() == b"test_bytes_2"

        custom_dir = os.path.join(tmpdir, "custom_log")
        tiledb_obj.load_tensorboard(target_dir=custom_dir)
        with open(os.path.join(custom_dir, "foo_tfevents_1"), "rb") as f:
            assert f.read() == b"test_bytes_1"
        with open(os.path.join(custom_dir, "bar_tfevents_2"), "rb") as f:
            assert f.read() == b"test_bytes_2"
