from typing import Any, Callable, Mapping, Tuple

from ..types import TensorKind
from .base import TensorSchema
from .dense import DenseTensorSchema
from .mapped import MappedTensorSchema
from .ragged import RaggedArray, RaggedTensorSchema
from .sparse import SparseArray, SparseTensorSchema
from .sparse_to_dense import SparseToDenseTensorSchema

# mapping (is_sparse, tensor_kind): TensorSchema factory
TensorSchemaFactories: Mapping[
    Tuple[bool, TensorKind], Callable[..., TensorSchema[Any]]
] = {
    (False, TensorKind.DENSE): DenseTensorSchema,
    (True, TensorKind.DENSE): SparseToDenseTensorSchema,
    (True, TensorKind.RAGGED): RaggedTensorSchema,
    (True, TensorKind.SPARSE_COO): SparseTensorSchema,
    (True, TensorKind.SPARSE_CSR): SparseTensorSchema,
}
