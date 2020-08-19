# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import awkward1.highlevel
import awkward1.nplike
import awkward1.operations.convert
import awkward1.operations.reducers
import awkward1._util


np = awkward1.nplike.NumpyMetadata.instance()


class CategoricalBehavior(awkward1.highlevel.Array):
    __name__ = "Array"


awkward1.behavior["categorical"] = CategoricalBehavior


class _HashableDict(object):
    def __init__(self, obj):
        self.keys = tuple(sorted(obj))
        self.values = tuple(_hashable(obj[k]) for k in self.keys)
        self.hash = hash((_HashableDict,) + self.keys, self.values)

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        return (
            isinstance(other, _HashableDict)
            and self.keys == other.keys
            and self.values == other.values
        )


class _HashableList(object):
    def __init__(self, obj):
        self.values = tuple(obj)
        self.hash = hash((_HashableList,) + self.values)

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        return (
            isinstance(other, _HashableList)
            and self.values == other.values
        )


def _hashable(obj):
    if isinstance(obj, dict):
        return _HashableDict(obj)
    elif isinstance(obj, tuple):
        return tuple(_hashable(x) for x in obj)
    elif isinstance(obj, list):
        return _HashableList(obj)
    else:
        return obj


_all_indexedtypes = (
    awkward1.layout.IndexedOptionArray32,
    awkward1.layout.IndexedOptionArray64,
) + awkward1._util.indexedtypes


def _categorical_equal(one, two):
    nplike = awkward1.nplike.of(one, two)
    behavior = awkward1._util.behaviorof(one, two)

    one, two = one.layout, two.layout

    assert isinstance(one, _all_indexedtypes)
    assert isinstance(two, _all_indexedtypes)
    assert one.parameter("__array__") == "categorical"
    assert two.parameter("__array__") == "categorical"

    one_index = nplike.asarray(one.index)
    two_index = nplike.asarray(two.index)
    one_content = awkward1._util.wrap(one.content, behavior)
    two_content = awkward1._util.wrap(two.content, behavior)

    if (
        len(one_content) == len(two_content) and
        awkward1.operations.reducers.all(one_content == two_content, axis=None)
    ):
        one_mapped = one_index

    else:
        one_list = awkward1.operations.convert.to_list(one_content)
        two_list = awkward1.operations.convert.to_list(two_content)
        one_hashable = [_hashable(x) for x in one_list]
        two_hashable = [_hashable(x) for x in two_list]
        two_lookup = {x: i for i, x in enumerate(two_hashable)}

        one_to_two = nplike.empty(len(one_hashable) + 1, dtype=np.int64)
        for i, x in enumerate(one_hashable):
            one_to_two[i] = two_lookup.get(x, len(two_hashable))
        one_to_two[-1] = -1

        one_mapped = one_to_two[one_index]

    out = one_mapped == two_index
    return awkward1.highlevel.Array(awkward1.layout.NumpyArray(out))


awkward1.behavior[awkward1.nplike.numpy.equal, "categorical", "categorical"] = _categorical_equal
