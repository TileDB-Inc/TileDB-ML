import logging
import json
import pickle
from urllib.error import HTTPError
import numpy as np
import tensorflow as tf
import tiledb

from tensorflow.keras import Model
from tensorflow.python.keras import optimizer_v1
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.saving.saved_model import json_utils

from base_model import TileDBModel

TILEDB_CONTEXT = tiledb.Ctx()


class TensorflowTileDB(TileDBModel):
    """
    Class that implements all functionality needed to save Tensorflow models as
    TileDB arrays and load Tensorflow models from TileDB arrays.
    """
    def __init__(self, model: Model, include_optimizer: bool, **kwargs):
        self.model = model
        self.include_optimizer = include_optimizer
        self.serialized_model_weights = None
        self.serialized_optimizer_weights = None

        super(TensorflowTileDB, self).__init__(**kwargs)

    def save(self, update=False):
        """
        Saves a Tensorflow model as a TileDB array
        :param update: Whether we should update any existing TileDB array
        model at the target location
        """
        if len(self.model.weights) != len(self.model._undeduplicated_weights):
            logging.warning('Found duplicated `Variable`s in Model\'s `weights`. '
                            'This is usually caused by `Variable`s being shared by '
                            'Layers in the Model. These `Variable`s will be treated '
                            'as separate `Variable`s when the Model is restored.')

        # Serialize models weights and optimizer (if needed)
        self._serialize_model_weights()

        # Serialize model optimizer
        if self.include_optimizer:
            self._serialize_model_optimizer()

        # Create TileDB model array
        if not update:
            self._create_array()

        self._write_array()

    def load(self, compile_model=True, custom_objects=None):
        """
        Loads a Tensorflow model from a TileDB array
        :return: Model. Tensorflow model
        """

        try:
            model_array = tiledb.open(self.uri)
            model_weights = pickle.loads(model_array[:]['model_weights'].item(0))
            optimizer_weights = pickle.loads(model_array[:]['optimizer_weights'].item(0))
            model_config = json.loads(model_array.meta['model_config'])
            training_config = json.loads(model_array.meta['training_config'])

            architecture = model_config['config']
            model_class = model_config['class_name']

            if model_class == 'Sequential':
                model = tf.keras.Sequential.from_config(architecture)
            else:
                model = tf.keras.Model.from_config(architecture)

            model.set_weights(model_weights)

            if compile_model:
                # Compile model.
                model.compile(**saving_utils.compile_args_from_training_config(
                    training_config, custom_objects))
                saving_utils.try_build_compiled_arguments(model)

                # Set optimizer weights.
                if optimizer_weights:
                    try:
                        model.optimizer._create_all_weights(model.trainable_variables)
                    except (NotImplementedError, AttributeError):
                        logging.warning(
                            'Error when creating the weights of optimizer {}, making it '
                            'impossible to restore the saved optimizer state. As a result, '
                            'your model is starting with a freshly initialized optimizer.')

                    try:
                        model.optimizer.set_weights(optimizer_weights)
                    except ValueError:
                        logging.warning('Error in loading the saved optimizer '
                                        'state. As a result, your model is '
                                        'starting with a freshly initialized '
                                        'optimizer.')
        except tiledb.TileDBError as error:
            raise error
        except HTTPError as error:
            raise error
        except Exception as error:
            raise error

    def _create_array(self):
        """
        Creates a TileDB array for a Tensorflow model
        """
        try:
            dom = tiledb.Domain(
                tiledb.Dim(name="model",
                           domain=(1, 1),
                           tile=1,
                           dtype=np.int32
                           ),
                ctx=TILEDB_CONTEXT,
            )

            attrs = [
                tiledb.Attr(name="model_weights",
                            dtype="S1",
                            var=True,
                            filters=tiledb.FilterList([tiledb.ZstdFilter()]),
                            ctx=TILEDB_CONTEXT
                            ),
                tiledb.Attr(name="optimizer_weights",
                            dtype="S1",
                            var=True,
                            filters=tiledb.FilterList([tiledb.ZstdFilter()]),
                            ctx=TILEDB_CONTEXT
                            ),
            ]

            schema = tiledb.ArraySchema(domain=dom,
                                        sparse=False,
                                        attrs=attrs,
                                        ctx=TILEDB_CONTEXT
                                        )

            tiledb.Array.create(self.uri, schema, ctx=TILEDB_CONTEXT)
        except tiledb.TileDBError as error:
            if "Error while listing with prefix" in str(error):
                # It is possible to land here if user sets wrong default s3 credentials
                # with respect to default s3 path
                raise HTTPError(code=400, msg="Error creating file, %s Are your S3 "
                                              "credentials valid?" % str(error))

            if "already exists" in str(error):
                logging.warning('TileDB array already exists but update=False. '
                                'Next time set update=True. Returning')
                raise error
        except HTTPError as error:
            raise error
        except Exception as error:
            raise HTTPError(code=400, msg="Error creating file %s " % str(error))

    def _write_array(self):
        """
        Writes Tensorflow model to a TileDB array.
        """
        with tiledb.open(self.uri, 'w') as tf_model_tiledb:
            # Insert weights and optimizer
            tf_model_tiledb[:] = {"model_weights": np.array([self.serialized_model_weights]),
                                  "optimizer_weights": np.array([self.serialized_optimizer_weights])}

            # Insert all model metadata
            model_metadata = saving_utils.model_metadata(self.model, self.include_optimizer)
            for key, value in model_metadata.items():
                if isinstance(value, (dict, list, tuple)):
                    tf_model_tiledb.meta[key] = json.dumps(
                        value, default=json_utils.get_json_type).encode('utf8')
                else:
                    tf_model_tiledb.meta[key] = value

            # Insert description
            tf_model_tiledb.meta["description"] = str(self.description)

    def _serialize_model_weights(self):
        """
        Serialization of model weights
        """
        self.serialized_model_weights = pickle.dumps(self.model.get_weights(), protocol=-1)

    def _serialize_model_optimizer(self):
        """
        Serialization of model optimizer
        """
        if self.include_optimizer and self.model.optimizer and \
                not isinstance(self.model.optimizer, optimizer_v1.TFOptimizer):

            optimizer_weights = tf.keras.backend.batch_get_value(
                getattr(self.model.optimizer, 'weights'))

            self.serialized_optimizer_weights = pickle.dumps(optimizer_weights, protocol=-1)