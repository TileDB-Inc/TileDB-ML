from typing import Any, Callable, Generic, Iterable, Sequence, TypeVar, Union

import wrapt

from .base import Tensor, TensorSchema
from .ranges import InclusiveRange

MappedTensor = TypeVar("MappedTensor")


class MappedTensorSchema(wrapt.ObjectProxy, Generic[Tensor, MappedTensor]):
    """
    Proxy class that wraps a TensorSchema and applies a mapping function to each tensor
    yielded by `iter_tensors`.
    """

    def __init__(
        self,
        wrapped: TensorSchema[Tensor],
        map_tensor: Callable[[Tensor], MappedTensor],
    ):
        super().__init__(wrapped)
        self._self_map_tensor = map_tensor

    def iter_tensors(
        self, key_ranges: Iterable[InclusiveRange[Any, int]]
    ) -> Union[Iterable[MappedTensor], Iterable[Sequence[MappedTensor]]]:
        wrapped_iter_tensors = self.__wrapped__.iter_tensors(key_ranges)
        if self.num_fields == 1:
            return map(self._self_map_tensor, wrapped_iter_tensors)
        else:
            return (
                tuple(map(self._self_map_tensor, tensors))
                for tensors in wrapped_iter_tensors
            )

    def __reduce_ex__(self, protocol):  # type: ignore
        return MappedTensorSchema, (self.__wrapped__, self._self_map_tensor)
