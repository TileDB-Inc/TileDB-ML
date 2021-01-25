import tiledb
import logging
import numpy as np
import pickle
import json

from tensorflow.keras import Model
from base_model import TileDBModel
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.saving.saved_model import json_utils

TILEDB_CONTEXT = tiledb.Ctx()


class TensorflowTileDB(TileDBModel):

    def __init__(self, model: Model, **kwargs):
        self.model = model
        super(TensorflowTileDB, self).__init__(**kwargs)

    def save(self, update=False, include_optimizer=True):
        """
        Saves a Tensorflow model as a TileDB array

        :param update: Whether we should update any existing TileDB array model at the target location
        :param include_optimizer: If True, save optimizer's state together
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

        # Get model metadata and weights
        model_metadata = saving_utils.model_metadata(self.model, include_optimizer)
        model_weights = self.model.get_weights()

        # Serialize model weights
        serialized_model_weights = [str(pickle.dumps(layer_weights)) for layer_weights in model_weights]

        # Create TileDB model array
        if not update:
            self._create_array(len(model_weights))

        with tiledb.open(self.uri, 'w') as tf_model_tiledb:
            # Insert weights
            tf_model_tiledb[:] = {"weights": np.array(serialized_model_weights)}

            # Insert all model metadata
            for key, value in model_metadata.items():
                if isinstance(value, (dict, list, tuple)):
                    tf_model_tiledb.meta[key] = json.dumps(
                        value, default=json_utils.get_json_type).encode('UTF-8')
                else:
                    tf_model_tiledb.meta[key] = value

    def load(self):
        """
        Loads a Tensorflow model from a TileDB array
        :return: Model. Tensorflow model
        """

    def _create_array(self, domain_extent):
        """
        Creates a TileDB array for a Tensorflow model
        :param domain_extent: The extent of the domain of the 1st dimension, i.e, model layer index
        """

        dom = tiledb.Domain(
            tiledb.Dim(name="layer_index",
                       domain=(0, domain_extent - 1),
                       tile=domain_extent,
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
                        )
        ]

        schema = tiledb.ArraySchema(domain=dom,
                                    sparse=False,
                                    attrs=attrs,
                                    ctx=TILEDB_CONTEXT
                                    )

        tiledb.Array.create(self.uri, schema, ctx=TILEDB_CONTEXT)
