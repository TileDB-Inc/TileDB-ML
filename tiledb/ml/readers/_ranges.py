from __future__ import annotations

import itertools as it
from abc import ABC, abstractmethod
from collections import Counter, abc
from dataclasses import dataclass
from functools import singledispatch
from typing import Any, Generic, Iterable, Mapping, NoReturn, TypeVar, Union, cast

import numpy as np

__all__ = ["InclusiveRange"]

V = TypeVar("V")
W = TypeVar("W")
VDtype = TypeVar("VDtype", bound=np.generic)
WDtype = TypeVar("WDtype", bound=np.number)


class InclusiveRange(ABC, Generic[V, W]):
    """
    Base abstract class for finite ranges that include both ends.

    An InclusiveRange `R` is a sorted set of comparable elements of type `V`. Every member
    `m` of `R` satisfies the inequality `R.min <= m <= R.max`.

    As a set, an InclusiveRange does not contain duplicates. However, range members may
    be associated with numeric weights of type `W`. This way duplicate members can be
    represented as unique members with weight equal to the number of occurrences.
    """

    __slots__: Iterable[str] = ()

    @property
    @abstractmethod
    def min(self) -> V:
        """Lower bound of this range."""

    @property
    @abstractmethod
    def max(self) -> V:
        """Upper bound of this range."""

    @property
    @abstractmethod
    def weight(self) -> W:
        """Total weight of the members of this range."""

    @property
    @abstractmethod
    def values(self) -> np.ndarray[V, Any]:
        """Unique sorted values of this range."""

    @abstractmethod
    def indices(self, values: np.ndarray[V, Any]) -> np.ndarray[np.int_, Any]:
        """Get the (0-based) indices of the given values in this range.

        :raises ValueError: If any of the given values is not in this range.
        """

    @abstractmethod
    def __len__(self) -> int:
        """Number of unique members in this range."""

    @abstractmethod
    def __eq__(self, other: Any) -> bool:
        """Check if this range equals to another object.

        Two ranges are equal if they consist of the same values with the same weights.
        """

    @abstractmethod
    def equal_values(self, other: InclusiveRange[V, W]) -> bool:
        """Check if two ranges consist of the same values."""

    @abstractmethod
    def partition_by_count(self, k: int) -> Iterable[InclusiveRange[V, W]]:
        """Partition this range into `k` subranges of approximately equal weight."""

    @abstractmethod
    def partition_by_weight(self, max_weight: W) -> Iterable[InclusiveRange[V, W]]:
        """
        Partition this range into the minimum number of subranges so that each subrange
        weight is at most `max_weight`.
        """

    @singledispatch
    @staticmethod
    def factory(values: Any) -> InclusiveRange[Any, Any]:
        """Create an inclusive range of comparable values.

        :param values: Can be:
        - an integer `range`. The weight of each value is 1.
        - an iterable of (potentially non-unique) values. The weight of each unique value
          is its cardinality.
        - a mapping from (unique) values to weights.
        """

    @staticmethod
    @factory.register(range)
    def _from_range(values: range) -> Union[IntRange, WeightedRange[np.int_, np.int_]]:
        if values.step < 0:
            values = range(values.stop + 1, values.start + 1, -values.step)

        if values.step == 1:
            return IntRange(values.start, values.stop - 1)
        else:
            return WeightedRange(
                np.arange(values.start, values.stop, values.step),
                np.ones(len(values), dtype=int),
            )

    @staticmethod
    @factory.register(abc.Iterable)
    def _from_iterable(values: Iterable[V]) -> Union[IntRange, InclusiveRange[V, int]]:
        return InclusiveRange._from_mapping(Counter(values))

    @staticmethod
    @factory.register(abc.Mapping)
    def _from_mapping(mapping: Mapping[V, W]) -> Union[IntRange, InclusiveRange[V, W]]:
        unique_sorted, weights = zip(*sorted(mapping.items()))
        return InclusiveRange._from_ndarrays(np.array(unique_sorted), np.array(weights))

    @staticmethod
    @factory.register(np.ndarray)
    def _from_ndarray(
        values: np.ndarray[VDtype, Any],
    ) -> Union[IntRange, WeightedRange[VDtype, np.int_]]:
        unique_sorted, counts = np.unique(values, return_counts=True)
        return InclusiveRange._from_ndarrays(unique_sorted, counts)

    @staticmethod
    def _from_ndarrays(
        unique_sorted: np.ndarray[VDtype, Any], weights: np.ndarray[WDtype, Any]
    ) -> Union[IntRange, WeightedRange[VDtype, WDtype]]:
        if (
            np.issubsctype(unique_sorted, np.integer)
            and unique_sorted[-1] - unique_sorted[0] + 1 == len(unique_sorted)
            and np.all(weights == 1)
        ):
            return IntRange(unique_sorted[0].item(), unique_sorted[-1].item())
        else:
            return WeightedRange(unique_sorted, weights)

    @staticmethod
    @factory.register(object)
    def _from_object(values: object) -> NoReturn:
        raise TypeError(f"Cannot create inclusive range from {values!r}")

    def __getstate__(self) -> Mapping[str, Any]:
        return {slot: getattr(self, slot) for slot in self.__slots__}

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        for slot, value in state.items():
            object.__setattr__(self, slot, value)


@dataclass(frozen=True)
class IntRange(InclusiveRange[int, int]):
    """
    An inclusive range of consecutive integers.
    """

    __slots__ = ("min", "max")
    min: int
    max: int

    def __post_init__(self) -> None:
        assert self.min <= self.max

    @property
    def weight(self) -> int:
        return len(self)

    @property
    def values(self) -> np.ndarray[int, Any]:
        return np.arange(self.min, self.max + 1)

    def indices(self, values: np.ndarray[int, Any]) -> np.ndarray[np.int_, Any]:
        indices = values - self.min
        if not (
            np.issubsctype(indices, np.integer)
            and np.all(indices >= 0)
            and np.all(indices < len(self))
        ):
            raise ValueError(f"Values not in the range {self}")
        return indices

    def __len__(self) -> int:
        return self.max - self.min + 1

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, IntRange):
            return NotImplemented
        return self.min == other.min and self.max == other.max

    def equal_values(self, other: InclusiveRange[V, W]) -> bool:
        return self == other

    def partition_by_count(self, k: int) -> Iterable[IntRange]:
        n = len(self)
        if not (1 <= k <= n):
            raise ValueError(
                f"Cannot partition range of {n} members into {k} partitions"
            )
        d, m = divmod(n, k)
        # the first m partitions have length d+1 and the rest have length d
        lengths = it.chain(it.repeat(d + 1, m), it.repeat(d, k - m))
        yield from self._partition_by_lengths(lengths)

    def partition_by_weight(self, max_weight: int) -> Iterable[IntRange]:
        d, m = divmod(len(self), max_weight)
        # all partitions have length max_weight, with the possible exception of the last
        # partition that has length m (if m > 0)
        lengths = it.chain(it.repeat(max_weight, d), it.repeat(m, m > 0))
        yield from self._partition_by_lengths(lengths)

    def _partition_by_lengths(self, lengths: Iterable[int]) -> Iterable[IntRange]:
        start = self.min
        for length in lengths:
            next_start = start + length
            yield IntRange(start, next_start - 1)
            start = next_start


@dataclass(frozen=True)
class WeightedRange(InclusiveRange[VDtype, WDtype]):
    """An inclusive range of weighted comparable values."""

    __slots__ = ("values", "weights")
    values: np.ndarray[VDtype, Any]
    weights: np.ndarray[WDtype, Any]

    def __post_init__(self) -> None:
        assert self.values.ndim == self.weights.ndim == 1
        assert len(self.values) == len(self.weights)

    @property
    def min(self) -> VDtype:
        return cast(VDtype, self.values[0])

    @property
    def max(self) -> VDtype:
        return cast(VDtype, self.values[-1])

    @property
    def weight(self) -> WDtype:
        return cast(WDtype, np.sum(self.weights))

    def __len__(self) -> int:
        return len(self.values)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, WeightedRange):
            return NotImplemented
        return bool(
            len(self) == len(other)
            and np.all(self.values == other.values)
            and np.all(self.weights == other.weights)
        )

    def equal_values(self, other: InclusiveRange[V, W]) -> bool:
        return bool(len(self) == len(other) and np.all(self.values == other.values))

    def indices(self, values: np.ndarray[VDtype, Any]) -> np.ndarray[np.int_, Any]:
        members = self.values
        indices = np.searchsorted(members, values)
        if not (np.max(indices) < len(members) and np.all(members[indices] == values)):
            raise ValueError(f"Values not in the range {self}")
        return indices

    def partition_by_count(self, k: int) -> Iterable[WeightedRange[VDtype, WDtype]]:
        n = len(self)
        if not (1 <= k <= n):
            raise ValueError(
                f"Cannot partition range of {n} members into {k} partitions"
            )

        values, weights = self.values, self.weights
        if k == n:
            # edge case: one partition per value
            for i in range(n):
                yield WeightedRange(values[i : i + 1], weights[i : i + 1])
            return

        total_weight = self.weight
        # target_weight: sum of
        # (a) weights of the partitions yielded so far and
        # (b) the average of the remaining total weight over the remaining partitions
        target_weight = total_weight / k
        acc_weights = np.cumsum(weights)
        start = 0
        for i in range(k - 1, 0, -1):
            # find the index of target_weight in the accumulated weights
            stop = int(np.searchsorted(acc_weights, target_weight))
            # check if it's after the start index; if not, move it right after start
            if stop <= start < n:
                stop = start + 1
            # otherwise the following inequality holds:
            # acc_weights[stop - 1] < target_weight <= acc_weights[stop]
            elif (
                acc_weights[stop] - target_weight
                < target_weight - acc_weights[stop - 1]
            ):
                # increment stop if acc_weights[stop] is closer to target_weight than
                # acc_weights[stop - 1]
                stop += 1

            # yield partition
            yield WeightedRange(values[start:stop], weights[start:stop])

            # update target_weight and start index
            target_weight = acc_weights[stop - 1]
            target_weight += (total_weight - target_weight) / i
            start = stop

        # yield last partition
        yield WeightedRange(values[start:], weights[start:])

    def partition_by_weight(
        self, max_weight: WDtype
    ) -> Iterable[WeightedRange[VDtype, WDtype]]:
        values, weights = self.values, self.weights
        if max_weight < np.max(weights):
            raise ValueError(
                f"Cannot partition range with max weight={max_weight}: "
                f"max value weight={np.max(weights)}"
            )
        # target_weight: sum of weights of the partitions yielded so far plus max_weight
        target_weight = max_weight
        acc_weights = np.cumsum(weights)
        start = 0
        n = len(acc_weights)
        while start < n:
            # find the index of target_weight in the accumulated weights
            stop = int(np.searchsorted(acc_weights, target_weight, side="right"))
            # yield partition
            yield WeightedRange(values[start:stop], weights[start:stop])
            # update target_weight and start index
            target_weight = acc_weights[stop - 1] + max_weight
            start = stop
