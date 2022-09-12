import numpy as np
import pytest
import scipy.sparse
import sparse
import torch

from tiledb.ml.readers import _pytorch_collators as pc

from .utils import get_tensor_kind


class TestNumpyArrayCollator:
    def test_convert_dense(self):
        value = np.random.rand(2, 5)
        tensor = pc.NumpyArrayCollator().convert(value)
        assert tensor.shape == (2, 5)
        assert get_tensor_kind(tensor) is pc.TensorKind.DENSE
        np.testing.assert_array_equal(tensor, value)

    def test_collate_dense(self):
        batch = [np.random.rand(2, 5) for _ in range(7)]
        tensor = pc.NumpyArrayCollator().collate(batch)
        assert tensor.shape == (7, 2, 5)
        assert get_tensor_kind(tensor) is pc.TensorKind.DENSE
        np.testing.assert_array_equal(tensor, np.stack(batch))

    def test_convert_nested(self):
        value = np.random.rand(10)
        tensor = pc.NumpyArrayCollator(to_nested=True).convert(value)
        assert tensor.shape == (10,)
        assert get_tensor_kind(tensor) is pc.TensorKind.DENSE
        np.testing.assert_array_equal(tensor, value)

    @pytest.mark.skipif(
        not hasattr(torch, "nested"), reason="Nested tensors not supported"
    )
    def test_collate_nested(self):
        batch = [np.random.rand(8), np.random.rand(3), np.random.rand(7)]
        tensor = pc.NumpyArrayCollator(to_nested=True).collate(batch)
        assert list(t.shape for t in tensor) == [(8,), (3,), (7,)]
        assert get_tensor_kind(tensor) is pc.TensorKind.RAGGED
        for tensor_row, batch_row in zip(tensor, batch):
            np.testing.assert_array_equal(tensor_row, batch_row)


class TestSparseCOOCollator:
    def test_convert_to_coo(self):
        value = sparse.random((2, 5), nnz=3)
        tensor = pc.SparseCOOCollator().convert(value)
        assert tensor.shape == (2, 5)
        assert get_tensor_kind(tensor) is pc.TensorKind.SPARSE_COO
        np.testing.assert_array_equal(tensor.to_dense(), value.todense())

    def test_collate_to_coo(self):
        batch = [sparse.random((2, 5), nnz=nnz) for nnz in np.random.randint(0, 11, 7)]
        tensor = pc.SparseCOOCollator().collate(batch)
        assert tensor.shape == (7, 2, 5)
        assert get_tensor_kind(tensor) is pc.TensorKind.SPARSE_COO
        np.testing.assert_array_equal(tensor.to_dense(), sparse.stack(batch).todense())

    def test_convert_to_csr(self):
        value = sparse.random((2, 5), nnz=3)
        tensor = pc.SparseCOOCollator(to_csr=True).convert(value)
        assert tensor.shape == (2, 5)
        assert get_tensor_kind(tensor) is pc.TensorKind.SPARSE_CSR
        np.testing.assert_array_equal(tensor.to_dense(), value.todense())

    def test_collate_to_csr(self):
        batch = [sparse.random(10, nnz=nnz) for nnz in np.random.randint(0, 11, 7)]
        tensor = pc.SparseCOOCollator(to_csr=True).collate(batch)
        assert tensor.shape == (7, 10)
        assert get_tensor_kind(tensor) is pc.TensorKind.SPARSE_CSR
        np.testing.assert_array_equal(tensor.to_dense(), sparse.stack(batch).todense())


class TestScipySparseCSRCollator:
    def test_convert_to_coo(self):
        value = scipy.sparse.random(2, 5, density=0.3, format="csr")
        tensor = pc.ScipySparseCSRCollator().convert(value)
        assert tensor.shape == (2, 5)
        assert get_tensor_kind(tensor) is pc.TensorKind.SPARSE_COO
        np.testing.assert_array_equal(tensor.to_dense(), value.toarray())

    def test_convert_to_coo_single_row(self):
        value = scipy.sparse.random(1, 10, density=0.3, format="csr")
        tensor = pc.ScipySparseCSRCollator().convert(value)
        assert tensor.shape == (10,)
        assert get_tensor_kind(tensor) is pc.TensorKind.SPARSE_COO
        np.testing.assert_array_equal(tensor.to_dense(), value.toarray()[0])

    def test_collate_to_coo(self):
        batch = [
            scipy.sparse.random(1, 10, density=nnz / 10, format="csr")
            for nnz in np.random.randint(0, 11, 7)
        ]
        tensor = pc.ScipySparseCSRCollator().collate(batch)
        assert tensor.shape == (7, 10)
        assert get_tensor_kind(tensor) is pc.TensorKind.SPARSE_COO
        np.testing.assert_array_equal(
            tensor.to_dense(), scipy.sparse.vstack(batch).toarray()
        )

    def test_convert_to_csr(self):
        value = scipy.sparse.random(2, 5, density=0.3, format="csr")
        tensor = pc.ScipySparseCSRCollator(to_csr=True).convert(value)
        assert tensor.shape == (2, 5)
        assert get_tensor_kind(tensor) is pc.TensorKind.SPARSE_CSR
        np.testing.assert_array_equal(tensor.to_dense(), value.toarray())

    def test_collate_to_csr(self):
        batch = [
            scipy.sparse.random(1, 10, density=nnz / 10, format="csr")
            for nnz in np.random.randint(0, 11, 7)
        ]
        tensor = pc.ScipySparseCSRCollator(to_csr=True).collate(batch)
        assert tensor.shape == (7, 10)
        assert get_tensor_kind(tensor) is pc.TensorKind.SPARSE_CSR
        np.testing.assert_array_equal(
            tensor.to_dense(), scipy.sparse.vstack(batch).toarray()
        )


class TestRowCollator:
    collator = pc.RowCollator(
        [
            pc.NumpyArrayCollator(),
            pc.SparseCOOCollator(),
            pc.ScipySparseCSRCollator(to_csr=True),
        ]
    )

    def test_convert(self):
        row = self._get_row()
        tensors = self.collator.convert(row)

        assert isinstance(tensors, tuple)
        assert len(tensors) == 3

        assert tensors[0].shape == (10,)
        assert get_tensor_kind(tensors[0]) is pc.TensorKind.DENSE
        np.testing.assert_array_equal(tensors[0], row[0])

        assert tensors[1].shape == (3, 4, 2)
        assert get_tensor_kind(tensors[1]) is pc.TensorKind.SPARSE_COO
        np.testing.assert_array_equal(tensors[1].to_dense(), row[1].todense())

        assert tensors[2].shape == (1, 5)
        assert get_tensor_kind(tensors[2]) is pc.TensorKind.SPARSE_CSR
        np.testing.assert_array_equal(tensors[2].to_dense(), row[2].toarray())

    def test_collate(self):
        rows = [self._get_row() for _ in range(7)]
        tensors = self.collator.collate(rows)
        columns = list(zip(*rows))

        assert isinstance(tensors, tuple)
        assert len(tensors) == 3

        assert tensors[0].shape == (7, 10)
        assert get_tensor_kind(tensors[0]) is pc.TensorKind.DENSE
        np.testing.assert_array_equal(tensors[0], columns[0])

        assert tensors[1].shape == (7, 3, 4, 2)
        assert get_tensor_kind(tensors[1]) is pc.TensorKind.SPARSE_COO
        np.testing.assert_array_equal(
            tensors[1].to_dense(), sparse.stack(columns[1]).todense()
        )

        assert tensors[2].shape == (7, 5)
        assert get_tensor_kind(tensors[2]) is pc.TensorKind.SPARSE_CSR
        np.testing.assert_array_equal(
            tensors[2].to_dense(), scipy.sparse.vstack(columns[2]).toarray()
        )

    @staticmethod
    def _get_row():
        return (
            np.random.rand(10),
            sparse.random((3, 4, 2), nnz=np.random.randint(24)),
            scipy.sparse.random(1, 5, density=np.random.randint(10) / 10, format="csr"),
        )


class Test_Collator_from_schemas:
    @pytest.mark.parametrize(
        "tensor_kind,to_nested",
        [(pc.TensorKind.DENSE, False), (pc.TensorKind.RAGGED, True)],
    )
    def test_numpy(self, mocker, tensor_kind, to_nested):
        schema = mocker.Mock(kind=tensor_kind, num_fields=1)
        collator = pc.Collator.from_schemas(schema)
        assert isinstance(collator, pc.NumpyArrayCollator)
        assert collator.to_nested is to_nested
        self._test_multiple_fields(schema, collator)

    @pytest.mark.parametrize(
        "tensor_kind,shape,collator_cls,to_csr",
        [
            (pc.TensorKind.SPARSE_COO, (3, 4, 2), pc.SparseCOOCollator, False),
            (pc.TensorKind.SPARSE_CSR, (3, 4, 2), pc.SparseCOOCollator, True),
            (pc.TensorKind.SPARSE_COO, (3, 4), pc.ScipySparseCSRCollator, False),
            (pc.TensorKind.SPARSE_CSR, (3, 4), pc.ScipySparseCSRCollator, True),
        ],
    )
    def test_sparse(self, mocker, tensor_kind, shape, collator_cls, to_csr):
        schema = mocker.Mock(kind=tensor_kind, shape=shape, num_fields=1)
        collator = pc.Collator.from_schemas(schema)
        assert isinstance(collator, collator_cls)
        assert collator.to_csr is to_csr
        self._test_multiple_fields(schema, collator)

    def test_multiple(self, mocker):
        schemas = (
            mocker.Mock(kind=pc.TensorKind.DENSE, num_fields=1),
            mocker.Mock(kind=pc.TensorKind.SPARSE_CSR, shape=(3, 4, 2), num_fields=2),
            mocker.Mock(kind=pc.TensorKind.SPARSE_COO, shape=(3, 4), num_fields=3),
        )

        multi_collator = pc.Collator.from_schemas(*schemas)
        assert isinstance(multi_collator, pc.RowCollator)
        assert len(multi_collator.column_collators) == 3

        collator = multi_collator.column_collators[0]
        assert isinstance(collator, pc.NumpyArrayCollator)
        assert collator.to_nested is False

        collator = multi_collator.column_collators[1]
        assert isinstance(collator, pc.RowCollator)
        assert len(collator.column_collators) == 2
        for sub_collator in collator.column_collators:
            assert isinstance(sub_collator, pc.SparseCOOCollator)
            assert sub_collator.to_csr is True

        collator = multi_collator.column_collators[2]
        assert isinstance(collator, pc.RowCollator)
        assert len(collator.column_collators) == 3
        for sub_collator in collator.column_collators:
            assert isinstance(sub_collator, pc.ScipySparseCSRCollator)
            assert sub_collator.to_csr is False

    @staticmethod
    def _test_multiple_fields(schema, collator):
        schema.num_fields = 2
        row_collator = pc.Collator.from_schemas(schema)
        assert isinstance(row_collator, pc.RowCollator)
        assert row_collator.column_collators == (collator, collator)
