import tiledb
import logging
import pickle
import json
import numpy as np

from tensorflow.keras import Model
from tensorflow.python.keras import optimizer_v1
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.saving.saved_model import json_utils

from base_model import TileDBModel

TILEDB_CONTEXT = tiledb.Ctx()


class TensorflowTileDB(TileDBModel):

    def __init__(self, model: Model, include_optimizer: bool, **kwargs):
        super(TensorflowTileDB, self).__init__(**kwargs)
        self.model = model
        self.include_optimizer = include_optimizer

        # Initialize model metadata and model weights
        self.model_metadata = saving_utils.model_metadata(self.model, self.include_optimizer)
        self.model_weights = model.get_weights()

    def save(self, update=False):
        """
        Saves a Tensorflow model as a TileDB array

        :param update: Whether we should update any existing TileDB array model at the target location
        """

        if len(self.model.weights) != len(self.model._undeduplicated_weights):
            logging.warning('Found duplicated `Variable`s in Model\'s `weights`. '
                            'This is usually caused by `Variable`s being shared by '
                            'Layers in the Model. These `Variable`s will be treated '
                            'as separate `Variable`s when the Model is restored.')

        # Check if the array already exists but update is False.
        if tiledb.object_type(self.uri) == "array" and update is False:
            logging.warning('TileDB array already exists but update=False. Next time set update=True. Returning')
            return

        # Serialize models weights and optimizer (if needed)
        self._serialize()

        # Create TileDB model array
        if not update:
            self._create_array()

        self._write_array()

    def load(self):
        """
        Loads a Tensorflow model from a TileDB array
        :return: Model. Tensorflow model
        """

    def _create_array(self):
        """
        Creates a TileDB array for a Tensorflow model
        """

        dom = tiledb.Domain(
            tiledb.Dim(name="model",
                       domain=(0, 0),
                       tile=1,
                       dtype=np.int32
                       ),
            ctx=TILEDB_CONTEXT,
        )

        attrs = [
            tiledb.Attr(name="weights",
                        dtype="U",
                        var=True,
                        filters=tiledb.FilterList([tiledb.ZstdFilter()]),
                        ctx=TILEDB_CONTEXT
                        ),
            tiledb.Attr(name="optimizer",
                        dtype="U",
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

    def _write_array(self):
        """
        Writes Tensorflow model to a TileDB array.
        """

        with tiledb.open(self.uri, 'w') as tf_model_tiledb:
            # Insert weights
            tf_model_tiledb[:] = {"weights": np.array([self.serialized_model_weights]),
                                  "optimizer": np.array([self.serialized_optimizer_weights])}

            # Insert all model metadata
            for key, value in self.model_metadata.items():
                if isinstance(value, (dict, list, tuple)):
                    tf_model_tiledb.meta[key] = json.dumps(
                        value, default=json_utils.get_json_type).encode('UTF-8')
                else:
                    tf_model_tiledb.meta[key] = value

    def _serialize(self):
        """
        Serialization of model weights and optimizer
        """

        # Serialize model weights.
        self.serialized_model_weights = str(pickle.dumps(self.model_weights, protocol=-1))

        if self.include_optimizer and self.model.optimizer and \
                not isinstance(self.model.optimizer, optimizer_v1.TFOptimizer):
            optimizer_weights = getattr(self.model.optimizer, 'weights')
            self.serialized_optimizer_weights = str(pickle.dumps(optimizer_weights, protocol=-1))