import pickle
from collections import Counter
from datetime import timedelta

import numpy as np
import pytest

from tiledb.ml.readers._tensor_schema.ranges import (
    ConstrainedPartitionsIntRange,
    IntRange,
    WeightedRange,
)


class TestIntRange:
    r = IntRange(10, 19)

    def test_basic(self):
        assert self.r.min == 10
        assert self.r.max == 19
        assert self.r.weight == 10
        assert len(self.r) == 10

    def test_equal(self):
        assert_equal_ranges(self.r, IntRange(10, 19))
        assert self.r != IntRange(0, 9)
        assert self.r != IntRange(10, 20)
        assert self.r != IntRange(11, 19)
        assert self.r != WeightedRange.from_mapping(dict.fromkeys(self.r.values, 2))

    def test_equal_values(self):
        assert self.r.equal_values(IntRange(10, 19))
        assert not self.r.equal_values(IntRange(10, 20))
        assert not self.r.equal_values(IntRange(11, 19))
        assert self.r.equal_values(
            WeightedRange.from_mapping(dict.fromkeys(self.r.values, 2))
        )

    def test_indices(self):
        np.testing.assert_array_equal(
            self.r.indices(np.array([10, 16, 16, 11, 19, 11])),
            np.array([0, 6, 6, 1, 9, 1]),
        )

    @pytest.mark.parametrize(
        "values",
        [
            [10, 16, 16, 11, 20],
            [10, 16, 16, 11, 9],
            [10, 16, 16, 11, 15.5],
        ],
    )
    def test_indices_error(self, values):
        with pytest.raises(ValueError) as excinfo:
            self.r.indices(np.array(values))
        assert "Values not in the range" in str(excinfo.value)

    @pytest.mark.parametrize(
        "k,expected_bounds",
        [
            (1, [(10, 19)]),
            (2, [(10, 14), (15, 19)]),
            (3, [(10, 13), (14, 16), (17, 19)]),
            (4, [(10, 12), (13, 15), (16, 17), (18, 19)]),
            (5, [(10, 11), (12, 13), (14, 15), (16, 17), (18, 19)]),
            (6, [(10, 11), (12, 13), (14, 15), (16, 17), (18, 18), (19, 19)]),
            (7, [(10, 11), (12, 13), (14, 15)] + [(i, i) for i in range(16, 20)]),
            (8, [(10, 11), (12, 13)] + [(i, i) for i in range(14, 20)]),
            (9, [(10, 11)] + [(i, i) for i in range(12, 20)]),
            (10, [(i, i) for i in range(10, 20)]),
        ],
    )
    def test_partition_by_count(self, k, expected_bounds):
        ranges = list(self.r.partition_by_count(k))
        assert len(ranges) == k
        expected_ranges = [IntRange(*bounds) for bounds in expected_bounds]
        assert ranges == expected_ranges

    def test_partition_by_count_error(self):
        for k in range(11, 20):
            with pytest.raises(ValueError) as excinfo:
                list(self.r.partition_by_count(k))
            assert "Cannot partition range" in str(excinfo.value)

    @pytest.mark.parametrize(
        "max_weight,expected_bounds",
        [
            (1, [(i, i) for i in range(10, 20)]),
            (2, [(10, 11), (12, 13), (14, 15), (16, 17), (18, 19)]),
            (3, [(10, 12), (13, 15), (16, 18), (19, 19)]),
            (4, [(10, 13), (14, 17), (18, 19)]),
            (5, [(10, 14), (15, 19)]),
            (6, [(10, 15), (16, 19)]),
            (7, [(10, 16), (17, 19)]),
            (8, [(10, 17), (18, 19)]),
            (9, [(10, 18), (19, 19)]),
            (10, [(10, 19)]),
            (11, [(10, 19)]),
        ],
    )
    def test_partition_by_weight(self, max_weight, expected_bounds):
        ranges = list(self.r.partition_by_weight(max_weight))
        assert max(r.weight for r in ranges) <= max_weight
        expected_ranges = [IntRange(*bounds) for bounds in expected_bounds]
        assert ranges == expected_ranges

    def test_pickle(self):
        assert pickle.loads(pickle.dumps(self.r)) == self.r


class TestConstrainedPartitionsIntRange:
    r = ConstrainedPartitionsIntRange(10, 29, range(1, 101, 4))

    @pytest.mark.parametrize(
        "k,expected_bounds",
        [
            (1, [(10, 29)]),
            (2, [(10, 20), (21, 29)]),
            (3, [(10, 16), (17, 24), (25, 29)]),
            (4, [(10, 16), (17, 20), (21, 24), (25, 29)]),
            (5, [(10, 12), (13, 16), (17, 20), (21, 24), (25, 29)]),
            (6, [(10, 12), (13, 16), (17, 20), (21, 24), (25, 28), (29, 29)]),
        ],
    )
    def test_partition_by_count(self, k, expected_bounds):
        ranges = list(self.r.partition_by_count(k))
        assert len(ranges) == k
        # all partitions after the first must start at a start_offset
        start_offsets = self.r.start_offsets
        assert all(r.min in start_offsets for r in ranges[1:])
        bounds = [(r.min, r.max) for r in ranges]
        assert bounds == expected_bounds

    @pytest.mark.parametrize("k", [7, 8, 9, 10])
    def test_partition_by_count_error(self, k):
        with pytest.raises(ValueError) as excinfo:
            list(self.r.partition_by_count(k))
        assert "Cannot partition range" in str(excinfo.value)

    @pytest.mark.parametrize(
        "max_weight,expected_bounds",
        [
            (4, [(10, 12), (13, 16), (17, 20), (21, 24), (25, 28), (29, 29)]),
            (5, [(10, 12), (13, 16), (17, 20), (21, 24), (25, 29)]),
            (6, [(10, 12), (13, 16), (17, 20), (21, 24), (25, 29)]),
            (7, [(10, 16), (17, 20), (21, 24), (25, 29)]),
            (8, [(10, 16), (17, 24), (25, 29)]),
            (9, [(10, 16), (17, 24), (25, 29)]),
            (10, [(10, 16), (17, 24), (25, 29)]),
            (11, [(10, 20), (21, 29)]),
        ],
    )
    def test_partition_by_weight(self, max_weight, expected_bounds):
        ranges = list(self.r.partition_by_weight(max_weight))
        assert max(r.weight for r in ranges) <= max_weight
        # all partitions after the first must start at a start_offset
        start_offsets = self.r.start_offsets
        assert all(r.min in start_offsets for r in ranges[1:])
        bounds = [(r.min, r.max) for r in ranges]
        assert bounds == expected_bounds

    @pytest.mark.parametrize("max_weight", [1, 2, 3])
    def test_partition_by_weight_error(self, max_weight):
        with pytest.raises(ValueError) as excinfo:
            list(self.r.partition_by_weight(max_weight))
        assert "Cannot partition range" in str(excinfo.value)


class TestWeightedRange:
    values = ("e", "f", "a", "d", "a", "c", "d", "a", "f", "c", "f", "f", "b", "d")
    r = WeightedRange.from_mapping(Counter(values))
    r2 = WeightedRange.from_mapping(
        {v: timedelta(c) for v, c in Counter(values).items()}
    )

    @pytest.mark.parametrize("r", [r, r2])
    def test_basic(self, r):
        assert r.min == "a"
        assert r.max == "f"
        assert len(r) == 6
        assert r.weight == 14 if r is self.r else timedelta(14)

    def test_equal(self):
        assert_equal_ranges(self.r, WeightedRange.from_mapping(Counter(self.values)))
        assert self.r != WeightedRange.from_mapping(Counter(set(self.values)))
        assert self.r != IntRange(0, len(set(self.values)) - 1)

    def test_equal_values(self):
        assert self.r.equal_values(
            WeightedRange.from_mapping(Counter(set(self.values)))
        )
        r = WeightedRange.from_mapping(Counter([1, 2, 3, 3, 4, 5]))
        assert r.equal_values(IntRange(1, 5))
        assert not r.equal_values(IntRange(1, 6))
        assert not r.equal_values(IntRange(2, 5))

    def test_indices(self):
        np.testing.assert_array_equal(
            self.r.indices(np.array(["a", "e", "e", "d", "a", "f", "c"])),
            np.array([0, 4, 4, 3, 0, 5, 2]),
        )

    @pytest.mark.parametrize(
        "values",
        [
            ["a", "e", "e", "d", "_"],
            ["a", "e", "e", "d", "z"],
            ["a", "e", "e", "d", "aa"],
        ],
    )
    def test_indices_error(self, values):
        with pytest.raises(ValueError) as excinfo:
            self.r.indices(np.array(values))
        assert "Values not in the range" in str(excinfo.value)

    parametrize_by_count = pytest.mark.parametrize(
        "k,expected_mappings",
        [
            (1, [{"e": 1, "f": 4, "a": 3, "d": 3, "c": 2, "b": 1}]),
            (2, [{"b": 1, "c": 2, "a": 3}, {"e": 1, "f": 4, "d": 3}]),
            (3, [{"b": 1, "a": 3}, {"c": 2, "d": 3}, {"e": 1, "f": 4}]),
            (4, [{"a": 3}, {"b": 1, "c": 2}, {"d": 3, "e": 1}, {"f": 4}]),
            (5, [{"a": 3}, {"b": 1, "c": 2}, {"d": 3}, {"e": 1}, {"f": 4}]),
            (6, [{"a": 3}, {"b": 1}, {"c": 2}, {"d": 3}, {"e": 1}, {"f": 4}]),
        ],
    )

    @parametrize_by_count
    def test_partition_by_count(self, k, expected_mappings):
        ranges = list(self.r.partition_by_count(k))
        assert len(ranges) == k
        expected_ranges = list(map(WeightedRange.from_mapping, expected_mappings))
        assert ranges == expected_ranges

    @parametrize_by_count
    def test_partition_by_count2(self, k, expected_mappings):
        ranges = list(self.r2.partition_by_count(k))
        assert len(ranges) == k
        expected_ranges = [
            WeightedRange.from_mapping({v: timedelta(w) for v, w in mapping.items()})
            for mapping in expected_mappings
        ]
        assert ranges == expected_ranges

    @pytest.mark.parametrize("r", [r, r2])
    def test_partition_by_count_error(self, r):
        for k in range(7, 20):
            with pytest.raises(ValueError) as excinfo:
                list(r.partition_by_count(k))
            assert "Cannot partition range" in str(excinfo.value)

    parametrize_by_max_weight = pytest.mark.parametrize(
        "max_weight,expected_mappings",
        [
            (4, [{"b": 1, "a": 3}, {"c": 2}, {"d": 3, "e": 1}, {"f": 4}]),
            (5, [{"b": 1, "a": 3}, {"c": 2, "d": 3}, {"e": 1, "f": 4}]),
            (6, [{"b": 1, "a": 3, "c": 2}, {"d": 3, "e": 1}, {"f": 4}]),
            (7, [{"b": 1, "a": 3, "c": 2}, {"d": 3, "e": 1}, {"f": 4}]),
            (8, [{"b": 1, "a": 3, "c": 2}, {"d": 3, "e": 1, "f": 4}]),
            (9, [{"b": 1, "a": 3, "c": 2, "d": 3}, {"e": 1, "f": 4}]),
            (10, [{"b": 1, "a": 3, "c": 2, "d": 3, "e": 1}, {"f": 4}]),
            (11, [{"b": 1, "a": 3, "c": 2, "d": 3, "e": 1}, {"f": 4}]),
            (12, [{"b": 1, "a": 3, "c": 2, "d": 3, "e": 1}, {"f": 4}]),
            (13, [{"b": 1, "a": 3, "c": 2, "d": 3, "e": 1}, {"f": 4}]),
            (14, [{"b": 1, "a": 3, "c": 2, "d": 3, "e": 1, "f": 4}]),
            (15, [{"b": 1, "a": 3, "c": 2, "d": 3, "e": 1, "f": 4}]),
        ],
    )

    @parametrize_by_max_weight
    def test_partition_by_weight(self, max_weight, expected_mappings):
        ranges = list(self.r.partition_by_weight(max_weight))
        assert max(r.weight for r in ranges) <= max_weight
        expected_ranges = list(map(WeightedRange.from_mapping, expected_mappings))
        assert ranges == expected_ranges

    @parametrize_by_max_weight
    def test_partition_by_weight2(self, max_weight, expected_mappings):
        max_weight = timedelta(max_weight)
        ranges = list(self.r2.partition_by_weight(max_weight))
        assert max(r.weight for r in ranges) <= max_weight
        expected_ranges = [
            WeightedRange.from_mapping({v: timedelta(w) for v, w in mapping.items()})
            for mapping in expected_mappings
        ]
        assert ranges == expected_ranges

    @pytest.mark.parametrize(
        "r,max_weights", [(r, range(1, 4)), (r2, map(timedelta, range(1, 4)))]
    )
    def test_partition_by_weight_error(self, r, max_weights):
        for max_weight in max_weights:
            with pytest.raises(ValueError) as excinfo:
                list(r.partition_by_weight(max_weight))
            assert "Cannot partition range" in str(excinfo.value)

    def test_pickle(self):
        assert pickle.loads(pickle.dumps(self.r)) == self.r
        assert pickle.loads(pickle.dumps(self.r2)) == self.r2


def assert_equal_ranges(r1, r2):
    assert r1.min == r2.min
    assert r1.max == r2.max
    assert r1.weight == r2.weight
    assert len(r1) == len(r2)
    assert r1 == r2
    assert r1.equal_values(r2)
