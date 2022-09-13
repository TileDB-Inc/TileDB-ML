from functools import singledispatch
from operator import methodcaller
from typing import Any

import numpy as np
import scipy.sparse
import sparse

from .base import TensorSchema
from .mapped import MappedTensorSchema
from .sparse import SparseTensorSchema


def SparseToDenseTensorSchema(**kwargs: Any) -> TensorSchema[np.ndarray]:
    """
    Return a TensorSchema for reading sparse TileDB arrays as (dense) Numpy arrays.
    """
    return MappedTensorSchema(SparseTensorSchema(**kwargs), to_dense)


@singledispatch
def to_dense(sa: Any) -> np.ndarray:
    """Create a Numpy array from a sparse array"""
    raise NotImplementedError


to_dense.register(sparse.COO)(methodcaller("todense"))  # type: ignore
to_dense.register(scipy.sparse.csr_matrix)(methodcaller("toarray"))  # type: ignore
