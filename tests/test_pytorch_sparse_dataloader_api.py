"""Tests for TileDB integration with Pytorch sparse Data API."""

import os
import torch
import tiledb
import numpy as np
import pytest


from tiledb.ml.readers.pytorch_sparse import PyTorchTileDBSparseDataset
from tiledb.ml._utils import ingest_in_tiledb, create_sparse_array_one_hot_2d

# Test parameters
NUM_OF_CLASSES = 5
BATCH_SIZE = 20
ROWS = 1000


@pytest.mark.parametrize(
    "input_shape",
    [
        (10,),
    ],
)
# TODO: Multiple workers require tiledb.SparseArray to be pickled hence serializable as well
# We test for single and multiple workers
@pytest.mark.parametrize(
    "workers",
    [0, 1, 2],
)
@pytest.mark.parametrize(
    "num_of_attributes",
    [1, 2, 3],
)
class TestTileDBSparsePyTorchDataloaderAPI:
    def test_tiledb_pytorch_sparse_data_api_with_sparse_data_sparse_label(
        self, tmpdir, input_shape, workers, num_of_attributes
    ):
        dataset_shape_x = (ROWS, input_shape)
        dataset_shape_y = (ROWS, (NUM_OF_CLASSES,))

        tiledb_uri_x = os.path.join(tmpdir, "x")
        tiledb_uri_y = os.path.join(tmpdir, "y")

        ingest_in_tiledb(
            uri=tiledb_uri_x,
            data=create_sparse_array_one_hot_2d(*dataset_shape_x),
            batch_size=BATCH_SIZE,
            sparse=True,
            num_of_attributes=num_of_attributes,
        )
        ingest_in_tiledb(
            uri=tiledb_uri_y,
            data=create_sparse_array_one_hot_2d(*dataset_shape_y),
            batch_size=BATCH_SIZE,
            sparse=True,
            num_of_attributes=num_of_attributes,
        )

        with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:

            tiledb_dataset = PyTorchTileDBSparseDataset(
                x_array=x,
                y_array=y,
                x_attribute_names=[
                    "features_" + str(attr) for attr in range(num_of_attributes)
                ],
                y_attribute_names=[
                    "features_" + str(attr) for attr in range(num_of_attributes)
                ],
                batch_size=BATCH_SIZE,
            )

            assert isinstance(tiledb_dataset, torch.utils.data.IterableDataset)

            # Same test without attribute names explicitly provided by the user
            tiledb_dataset = PyTorchTileDBSparseDataset(
                x_array=x,
                y_array=y,
                batch_size=BATCH_SIZE,
            )

            assert isinstance(tiledb_dataset, torch.utils.data.IterableDataset)

    def test_tiledb_pytorch_sparse_data_api_with_dense_data_sparse_label_except(
        self, tmpdir, input_shape, workers, num_of_attributes
    ):
        dataset_shape_x = (ROWS,) + input_shape
        dataset_shape_y = (ROWS, (NUM_OF_CLASSES,))

        tiledb_uri_x = os.path.join(tmpdir, "x")
        tiledb_uri_y = os.path.join(tmpdir, "y")

        ingest_in_tiledb(
            uri=tiledb_uri_x,
            data=np.random.rand(*dataset_shape_x),
            batch_size=BATCH_SIZE,
            sparse=False,
            num_of_attributes=num_of_attributes,
        )
        ingest_in_tiledb(
            uri=tiledb_uri_y,
            data=create_sparse_array_one_hot_2d(*dataset_shape_y),
            batch_size=BATCH_SIZE,
            sparse=True,
            num_of_attributes=num_of_attributes,
        )

        with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:
            with pytest.raises(TypeError):
                PyTorchTileDBSparseDataset(
                    x_array=x,
                    y_array=y,
                    x_attribute_names=[
                        "features_" + str(attr) for attr in range(num_of_attributes)
                    ],
                    y_attribute_names=[
                        "features_" + str(attr) for attr in range(num_of_attributes)
                    ],
                    batch_size=BATCH_SIZE,
                )
        # Same test without attribute names explicitly provided by the user
        with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:
            with pytest.raises(TypeError):
                PyTorchTileDBSparseDataset(x_array=x, y_array=y, batch_size=BATCH_SIZE)

    def test_tiledb_pytorch_sparse_data_api_with_sparse_data_dense_label(
        self, tmpdir, input_shape, workers, num_of_attributes
    ):
        dataset_shape_x = (ROWS, input_shape)
        dataset_shape_y = (ROWS,)

        tiledb_uri_x = os.path.join(tmpdir, "x")
        tiledb_uri_y = os.path.join(tmpdir, "y")

        ingest_in_tiledb(
            uri=tiledb_uri_x,
            data=create_sparse_array_one_hot_2d(*dataset_shape_x),
            batch_size=BATCH_SIZE,
            sparse=True,
            num_of_attributes=num_of_attributes,
        )
        ingest_in_tiledb(
            uri=tiledb_uri_y,
            data=np.random.randint(low=0, high=NUM_OF_CLASSES, size=dataset_shape_y),
            batch_size=BATCH_SIZE,
            sparse=False,
            num_of_attributes=num_of_attributes,
        )

        with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:
            tiledb_dataset = PyTorchTileDBSparseDataset(
                x_array=x,
                y_array=y,
                x_attribute_names=[
                    "features_" + str(attr) for attr in range(num_of_attributes)
                ],
                y_attribute_names=[
                    "features_" + str(attr) for attr in range(num_of_attributes)
                ],
                batch_size=BATCH_SIZE,
            )

            assert isinstance(tiledb_dataset, torch.utils.data.IterableDataset)

            # Same test without attribute names explicitly provided by the user
            tiledb_dataset = PyTorchTileDBSparseDataset(
                x_array=x,
                y_array=y,
                batch_size=BATCH_SIZE,
            )

            assert isinstance(tiledb_dataset, torch.utils.data.IterableDataset)

    def test_tiledb_pytorch_sparse_data_api_with_sparse_data_diff_number_of_x_y_rows(
        self, tmpdir, input_shape, workers, num_of_attributes
    ):
        # Add one extra row on X - Spoiled Data by adding one more row in X data
        dataset_shape_x = (ROWS + 1, input_shape)
        dataset_shape_y = (ROWS, (NUM_OF_CLASSES,))

        tiledb_uri_x = os.path.join(tmpdir, "x")
        tiledb_uri_y = os.path.join(tmpdir, "y")

        ingest_in_tiledb(
            uri=tiledb_uri_x,
            data=create_sparse_array_one_hot_2d(*dataset_shape_x),
            batch_size=BATCH_SIZE,
            sparse=True,
            num_of_attributes=num_of_attributes,
        )
        ingest_in_tiledb(
            uri=tiledb_uri_y,
            data=create_sparse_array_one_hot_2d(*dataset_shape_y),
            batch_size=BATCH_SIZE,
            sparse=True,
            num_of_attributes=num_of_attributes,
        )

        with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:
            with pytest.raises(ValueError):
                PyTorchTileDBSparseDataset(
                    x_array=x,
                    y_array=y,
                    x_attribute_names=[
                        "features_" + str(attr) for attr in range(num_of_attributes)
                    ],
                    y_attribute_names=[
                        "features_" + str(attr) for attr in range(num_of_attributes)
                    ],
                    batch_size=BATCH_SIZE,
                )
        # Same test without attribute names explicitly provided by the user
        with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:
            with pytest.raises(ValueError):
                PyTorchTileDBSparseDataset(x_array=x, y_array=y, batch_size=BATCH_SIZE)

    def test_tiledb_pytorch_sparse_data_api_with_diff_number_of_batch_x_y_rows_empty_record_except(
        self, tmpdir, input_shape, workers, num_of_attributes
    ):
        dataset_shape_x = (ROWS, input_shape)
        dataset_shape_y = (ROWS, (NUM_OF_CLASSES,))

        tiledb_uri_x = os.path.join(tmpdir, "x")
        tiledb_uri_y = os.path.join(tmpdir, "y")

        spoiled_data = create_sparse_array_one_hot_2d(*dataset_shape_x)
        spoiled_data[np.nonzero(spoiled_data[0])] = 0

        ingest_in_tiledb(
            uri=tiledb_uri_x,
            data=spoiled_data,
            batch_size=BATCH_SIZE,
            sparse=True,
            num_of_attributes=num_of_attributes,
        )
        ingest_in_tiledb(
            uri=tiledb_uri_y,
            data=create_sparse_array_one_hot_2d(*dataset_shape_y),
            batch_size=BATCH_SIZE,
            sparse=False,
            num_of_attributes=num_of_attributes,
        )

        with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:
            with pytest.raises(Exception):
                tiledb_dataset = PyTorchTileDBSparseDataset(
                    x_array=x,
                    y_array=y,
                    x_attribute_names=[
                        "features_" + str(attr) for attr in range(num_of_attributes)
                    ],
                    y_attribute_names=[
                        "features_" + str(attr) for attr in range(num_of_attributes)
                    ],
                    batch_size=BATCH_SIZE,
                )
                next(tiledb_dataset.__iter__())

        # Same test without attribute names explicitly provided by the user
        with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:
            with pytest.raises(Exception):
                tiledb_dataset = PyTorchTileDBSparseDataset(
                    x_array=x, y_array=y, batch_size=BATCH_SIZE
                )
                next(tiledb_dataset.__iter__())

    def test_tiledb_pytorch_sparse_sparse_label_data(
        self, tmpdir, input_shape, workers, num_of_attributes
    ):
        dataset_shape_x = (ROWS, input_shape)
        dataset_shape_y = (ROWS, (NUM_OF_CLASSES,))

        tiledb_uri_x = os.path.join(tmpdir, "x")
        tiledb_uri_y = os.path.join(tmpdir, "y")

        ingest_in_tiledb(
            uri=tiledb_uri_x,
            data=create_sparse_array_one_hot_2d(*dataset_shape_x),
            batch_size=BATCH_SIZE,
            sparse=True,
            num_of_attributes=num_of_attributes,
        )
        ingest_in_tiledb(
            uri=tiledb_uri_y,
            data=create_sparse_array_one_hot_2d(*dataset_shape_y),
            batch_size=BATCH_SIZE,
            sparse=True,
            num_of_attributes=num_of_attributes,
        )

        with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:
            tiledb_dataset = PyTorchTileDBSparseDataset(
                x_array=x,
                y_array=y,
                x_attribute_names=[
                    "features_" + str(attr) for attr in range(num_of_attributes)
                ],
                y_attribute_names=[
                    "features_" + str(attr) for attr in range(num_of_attributes)
                ],
                batch_size=BATCH_SIZE,
            )
            generated_data = next(tiledb_dataset.__iter__())
            for attr in range(num_of_attributes):
                assert generated_data[attr].layout == torch.sparse_coo
                assert (
                    generated_data[attr + num_of_attributes].layout == torch.sparse_coo
                )
                assert generated_data[attr].size() == (BATCH_SIZE, *input_shape)
                assert generated_data[attr + num_of_attributes].size() == (
                    BATCH_SIZE,
                    NUM_OF_CLASSES,
                )

        # Same test without attribute names explicitly provided by the user
        with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:
            tiledb_dataset = PyTorchTileDBSparseDataset(
                x_array=x, y_array=y, batch_size=BATCH_SIZE
            )
            generated_data = next(tiledb_dataset.__iter__())
            for attr in range(num_of_attributes):
                assert generated_data[attr].layout == torch.sparse_coo
                assert (
                    generated_data[attr + num_of_attributes].layout == torch.sparse_coo
                )
                assert generated_data[attr].size() == (BATCH_SIZE, *input_shape)
                assert generated_data[attr + num_of_attributes].size() == (
                    BATCH_SIZE,
                    NUM_OF_CLASSES,
                )
