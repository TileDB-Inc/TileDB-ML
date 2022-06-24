import pickle
from collections import Counter
from datetime import timedelta

import numpy as np
import pytest

from tiledb.ml.readers._ranges import InclusiveRange, IntRange, WeightedRange


@pytest.mark.parametrize("values", [None, 42, 3.14])
def test_inclusive_range_factory_type_error(values):
    with pytest.raises(TypeError) as excinfo:
        InclusiveRange.factory(values)
    assert "Cannot create inclusive range" in str(excinfo.value)


class TestIntRange:
    values = range(10, 20)
    r = InclusiveRange.factory(values)

    def test_basic(self):
        assert self.r.min == 10
        assert self.r.max == 19
        assert self.r.weight == 10
        assert len(self.r) == 10

    @pytest.mark.parametrize(
        "values",
        [
            values,
            list(values),
            set(values),
            iter(values),
            reversed(values),
            Counter(values),
            np.array(values),
            range(19, 9, -1),
            np.arange(19, 9, -1),
        ],
    )
    def test_equal(self, values):
        assert_equal_ranges(self.r, InclusiveRange.factory(values), IntRange)

    @pytest.mark.parametrize(
        "values",
        [
            np.array(values, dtype=object),
            range(0, 10),
            range(10, 21),
            range(11, 20),
            range(10, 20, 2),
        ],
    )
    def test_not_equal(self, values):
        r = InclusiveRange.factory(values)
        assert self.r != r
        assert not self.r.equal_values(r)

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
            (1, [(10, 20)]),
            (2, [(10, 15), (15, 20)]),
            (3, [(10, 14), (14, 17), (17, 20)]),
            (4, [(10, 13), (13, 16), (16, 18), (18, 20)]),
            (5, [(10, 12), (12, 14), (14, 16), (16, 18), (18, 20)]),
            (6, [(10, 12), (12, 14), (14, 16), (16, 18), (18, 19), (19, 20)]),
            (7, [(10, 12), (12, 14), (14, 16)] + [(i, i + 1) for i in range(16, 20)]),
            (8, [(10, 12), (12, 14)] + [(i, i + 1) for i in range(14, 20)]),
            (9, [(10, 12)] + [(i, i + 1) for i in range(12, 20)]),
            (10, [(i, i + 1) for i in range(10, 20)]),
        ],
    )
    def test_partition_by_count(self, k, expected_bounds):
        ranges = list(self.r.partition_by_count(k))
        assert len(ranges) == k
        expected_ranges = [InclusiveRange.factory(range(*bs)) for bs in expected_bounds]
        assert ranges == expected_ranges

    def test_partition_by_count_error(self):
        for k in range(11, 20):
            with pytest.raises(ValueError) as excinfo:
                list(self.r.partition_by_count(k))
            assert "Cannot partition range" in str(excinfo.value)

    @pytest.mark.parametrize(
        "max_weight,expected_bounds",
        [
            (1, [(i, i + 1) for i in range(10, 20)]),
            (2, [(10, 12), (12, 14), (14, 16), (16, 18), (18, 20)]),
            (3, [(10, 13), (13, 16), (16, 19), (19, 20)]),
            (4, [(10, 14), (14, 18), (18, 20)]),
            (5, [(10, 15), (15, 20)]),
            (6, [(10, 16), (16, 20)]),
            (7, [(10, 17), (17, 20)]),
            (8, [(10, 18), (18, 20)]),
            (9, [(10, 19), (19, 20)]),
            (10, [(10, 20)]),
            (11, [(10, 20)]),
        ],
    )
    def test_partition_by_weight(self, max_weight, expected_bounds):
        ranges = list(self.r.partition_by_weight(max_weight))
        assert max(r.weight for r in ranges) <= max_weight
        expected_ranges = [InclusiveRange.factory(range(*bs)) for bs in expected_bounds]
        assert ranges == expected_ranges

    def test_pickle(self):
        assert pickle.loads(pickle.dumps(self.r)) == self.r


class TestWeightedRange:
    values = ("e", "f", "a", "d", "a", "c", "d", "a", "f", "c", "f", "f", "b", "d")
    r = InclusiveRange.factory(values)
    r2 = InclusiveRange.factory({v: timedelta(c) for v, c in Counter(values).items()})

    @pytest.mark.parametrize("r", [r, r2])
    def test_basic(self, r):
        assert r.min == "a"
        assert r.max == "f"
        assert len(r) == 6
        assert r.weight == 14 if r is self.r else timedelta(14)

    @pytest.mark.parametrize(
        "values",
        [
            values,
            list(values),
            iter(values),
            reversed(values),
            Counter(values),
            np.array(values),
            np.array(values, dtype=object),
        ],
    )
    def test_equal(self, values):
        assert_equal_ranges(self.r, InclusiveRange.factory(values), WeightedRange)

    def test_not_equal(self):
        assert self.r != InclusiveRange.factory(set(self.values))
        assert self.r != InclusiveRange.factory(range(len(set(self.values))))

    def test_equal_values(self):
        assert self.r.equal_values(InclusiveRange.factory(set(self.values)))

        r = InclusiveRange.factory([1, 2, 3, 3, 4, 5])
        assert r.equal_values(InclusiveRange.factory(range(1, 6)))
        assert not r.equal_values(InclusiveRange.factory(range(1, 7)))
        assert not r.equal_values(InclusiveRange.factory(range(2, 7)))

    def test_strided_range(self):
        assert_equal_ranges(
            InclusiveRange.factory(range(10, 20, 3)),
            InclusiveRange.factory([10, 13, 16, 19]),
            WeightedRange,
        )
        assert_equal_ranges(
            InclusiveRange.factory(range(20, 10, -3)),
            InclusiveRange.factory([11, 14, 17, 20]),
            WeightedRange,
        )

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
        expected_ranges = list(map(InclusiveRange.factory, expected_mappings))
        assert ranges == expected_ranges

    @parametrize_by_count
    def test_partition_by_count2(self, k, expected_mappings):
        ranges = list(self.r2.partition_by_count(k))
        assert len(ranges) == k
        expected_ranges = [
            InclusiveRange.factory({v: timedelta(w) for v, w in mapping.items()})
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
        expected_ranges = list(map(InclusiveRange.factory, expected_mappings))
        assert ranges == expected_ranges

    @parametrize_by_max_weight
    def test_partition_by_weight2(self, max_weight, expected_mappings):
        max_weight = timedelta(max_weight)
        ranges = list(self.r2.partition_by_weight(max_weight))
        assert max(r.weight for r in ranges) <= max_weight
        expected_ranges = [
            InclusiveRange.factory({v: timedelta(w) for v, w in mapping.items()})
            for mapping in expected_mappings
        ]
        assert ranges == expected_ranges

    @pytest.mark.parametrize(
        "r,max_weights", [(r, range(1, 4)), (r2, map(timedelta, range(1, 4)))]
    )
    def test_partition_by_weight_error(self, r, max_weights):
        for max_weight in max_weights:
            with pytest.raises(ValueError):
                list(r.partition_by_weight(max_weight))

    def test_pickle(self):
        assert pickle.loads(pickle.dumps(self.r)) == self.r
        assert pickle.loads(pickle.dumps(self.r2)) == self.r2


def assert_equal_ranges(r1, r2, cls):
    assert isinstance(r1, cls)
    assert isinstance(r2, cls)
    assert r1.min == r2.min
    assert r1.max == r2.max
    assert r1.weight == r2.weight
    assert len(r1) == len(r2)
    assert r1 == r2
    assert r1.equal_values(r2)
