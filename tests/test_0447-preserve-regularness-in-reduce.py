# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest

import numpy
import awkward1


def test_nokeepdims():
    nparray = numpy.arange(2*3*5).reshape(2, 3, 5)
    content = awkward1.layout.NumpyArray(numpy.arange(2*3*5))
    regular = awkward1.layout.RegularArray(content, 5)
    listoffset = regular.toListOffsetArray64(False)
    regular_regular = awkward1.layout.RegularArray(regular, 3)
    listoffset_regular = regular_regular.toListOffsetArray64(False)
    regular_listoffset = awkward1.layout.RegularArray(listoffset, 3)
    listoffset_listoffset = regular_listoffset.toListOffsetArray64(False)

    assert awkward1.to_list(regular_regular) == [[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]], [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]]]
    assert awkward1.to_list(listoffset_regular) == [[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]], [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]]]
    assert awkward1.to_list(regular_listoffset) == [[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]], [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]]]
    assert awkward1.to_list(listoffset_listoffset) == [[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]], [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]]]

    assert str(awkward1.type(awkward1.Array(listoffset_listoffset))) == "2 * var * var * int64"
    axis1 = awkward1.sum(listoffset_listoffset, axis=-1)
    axis2 = awkward1.sum(listoffset_listoffset, axis=-2)
    axis3 = awkward1.sum(listoffset_listoffset, axis=-3)
    assert awkward1.to_list(axis1) == numpy.sum(nparray, axis=-1).tolist()
    assert awkward1.to_list(axis2) == numpy.sum(nparray, axis=-2).tolist()
    assert awkward1.to_list(axis3) == numpy.sum(nparray, axis=-3).tolist()
    assert str(awkward1.type(awkward1.Array(axis1))) == "2 * var * int64"
    assert str(awkward1.type(awkward1.Array(axis2))) == "2 * var * int64"
    assert str(awkward1.type(awkward1.Array(axis3))) == "3 * var * int64"

    assert str(awkward1.type(awkward1.Array(listoffset_regular))) == "2 * var * 5 * int64"
    axis1 = awkward1.sum(listoffset_regular, axis=-1)
    axis2 = awkward1.sum(listoffset_regular, axis=-2)
    axis3 = awkward1.sum(listoffset_regular, axis=-3)
    assert awkward1.to_list(axis1) == numpy.sum(nparray, axis=-1).tolist()
    assert awkward1.to_list(axis2) == numpy.sum(nparray, axis=-2).tolist()
    assert awkward1.to_list(axis3) == numpy.sum(nparray, axis=-3).tolist()
    assert str(awkward1.type(awkward1.Array(axis1))) == "2 * var * int64"
    assert str(awkward1.type(awkward1.Array(axis2))) == "2 * 5 * int64"
    assert str(awkward1.type(awkward1.Array(axis3))) == "3 * 5 * int64"

    assert str(awkward1.type(awkward1.Array(regular_listoffset))) == "2 * 3 * var * int64"
    axis1 = awkward1.sum(regular_listoffset, axis=-1)
    axis2 = awkward1.sum(regular_listoffset, axis=-2)
    axis3 = awkward1.sum(regular_listoffset, axis=-3)
    assert awkward1.to_list(axis1) == numpy.sum(nparray, axis=-1).tolist()
    assert awkward1.to_list(axis2) == numpy.sum(nparray, axis=-2).tolist()
    assert awkward1.to_list(axis3) == numpy.sum(nparray, axis=-3).tolist()
    assert str(awkward1.type(awkward1.Array(axis1))) == "2 * 3 * int64"
    assert str(awkward1.type(awkward1.Array(axis2))) == "2 * var * int64"
    assert str(awkward1.type(awkward1.Array(axis3))) == "3 * var * int64"

    assert str(awkward1.type(awkward1.Array(regular_regular))) == "2 * 3 * 5 * int64"
    axis1 = awkward1.sum(regular_regular, axis=-1)
    axis2 = awkward1.sum(regular_regular, axis=-2)
    axis3 = awkward1.sum(regular_regular, axis=-3)
    assert awkward1.to_list(axis1) == numpy.sum(nparray, axis=-1).tolist()
    assert awkward1.to_list(axis2) == numpy.sum(nparray, axis=-2).tolist()
    assert awkward1.to_list(axis3) == numpy.sum(nparray, axis=-3).tolist()
    assert str(awkward1.type(awkward1.Array(axis1))) == "2 * 3 * int64"
    assert str(awkward1.type(awkward1.Array(axis2))) == "2 * 5 * int64"
    assert str(awkward1.type(awkward1.Array(axis3))) == "3 * 5 * int64"

def test_keepdims():
    nparray = numpy.arange(2*3*5).reshape(2, 3, 5)
    content = awkward1.layout.NumpyArray(numpy.arange(2*3*5))
    regular = awkward1.layout.RegularArray(content, 5)
    listoffset = regular.toListOffsetArray64(False)
    regular_regular = awkward1.layout.RegularArray(regular, 3)
    listoffset_regular = regular_regular.toListOffsetArray64(False)
    regular_listoffset = awkward1.layout.RegularArray(listoffset, 3)
    listoffset_listoffset = regular_listoffset.toListOffsetArray64(False)

    assert str(awkward1.type(awkward1.Array(listoffset_listoffset))) == "2 * var * var * int64"
    axis1 = awkward1.sum(listoffset_listoffset, axis=-1, keepdims=True)
    axis2 = awkward1.sum(listoffset_listoffset, axis=-2, keepdims=True)
    axis3 = awkward1.sum(listoffset_listoffset, axis=-3, keepdims=True)
    assert awkward1.to_list(axis1) == numpy.sum(nparray, axis=-1, keepdims=True).tolist()
    assert awkward1.to_list(axis2) == numpy.sum(nparray, axis=-2, keepdims=True).tolist()
    assert awkward1.to_list(axis3) == numpy.sum(nparray, axis=-3, keepdims=True).tolist()
    assert str(awkward1.type(awkward1.Array(axis1))) == "2 * var * var * int64"
    assert str(awkward1.type(awkward1.Array(axis2))) == "2 * var * var * int64"
    assert str(awkward1.type(awkward1.Array(axis3))) == "1 * var * var * int64"

    assert str(awkward1.type(awkward1.Array(listoffset_regular))) == "2 * var * 5 * int64"
    axis1 = awkward1.sum(listoffset_regular, axis=-1, keepdims=True)
    axis2 = awkward1.sum(listoffset_regular, axis=-2, keepdims=True)
    axis3 = awkward1.sum(listoffset_regular, axis=-3, keepdims=True)
    assert awkward1.to_list(axis1) == numpy.sum(nparray, axis=-1, keepdims=True).tolist()
    assert awkward1.to_list(axis2) == numpy.sum(nparray, axis=-2, keepdims=True).tolist()
    assert awkward1.to_list(axis3) == numpy.sum(nparray, axis=-3, keepdims=True).tolist()
    assert str(awkward1.type(awkward1.Array(axis1))) == "2 * var * 1 * int64"
    assert str(awkward1.type(awkward1.Array(axis2))) == "2 * var * 5 * int64"
    assert str(awkward1.type(awkward1.Array(axis3))) == "1 * var * 5 * int64"

    assert str(awkward1.type(awkward1.Array(regular_listoffset))) == "2 * 3 * var * int64"
    axis1 = awkward1.sum(regular_listoffset, axis=-1, keepdims=True)
    axis2 = awkward1.sum(regular_listoffset, axis=-2, keepdims=True)
    axis3 = awkward1.sum(regular_listoffset, axis=-3, keepdims=True)
    assert awkward1.to_list(axis1) == numpy.sum(nparray, axis=-1, keepdims=True).tolist()
    assert awkward1.to_list(axis2) == numpy.sum(nparray, axis=-2, keepdims=True).tolist()
    assert awkward1.to_list(axis3) == numpy.sum(nparray, axis=-3, keepdims=True).tolist()
    assert str(awkward1.type(awkward1.Array(axis1))) == "2 * 3 * var * int64"
    assert str(awkward1.type(awkward1.Array(axis2))) == "2 * 1 * var * int64"
    assert str(awkward1.type(awkward1.Array(axis3))) == "1 * 3 * var * int64"

    assert str(awkward1.type(awkward1.Array(regular_regular))) == "2 * 3 * 5 * int64"
    axis1 = awkward1.sum(regular_regular, axis=-1, keepdims=True)
    axis2 = awkward1.sum(regular_regular, axis=-2, keepdims=True)
    axis3 = awkward1.sum(regular_regular, axis=-3, keepdims=True)
    assert awkward1.to_list(axis1) == numpy.sum(nparray, axis=-1, keepdims=True).tolist()
    assert awkward1.to_list(axis2) == numpy.sum(nparray, axis=-2, keepdims=True).tolist()
    assert awkward1.to_list(axis3) == numpy.sum(nparray, axis=-3, keepdims=True).tolist()
    assert str(awkward1.type(awkward1.Array(axis1))) == "2 * 3 * 1 * int64"
    assert str(awkward1.type(awkward1.Array(axis2))) == "2 * 1 * 5 * int64"
    assert str(awkward1.type(awkward1.Array(axis3))) == "1 * 3 * 5 * int64"

    # print(listoffset_listoffset)
    # print(repr(awkward1.Array(listoffset_listoffset)))
    # print()
    # print(repr(awkward1.sum(listoffset_listoffset, axis=-1, keepdims=True)))
    # print()
    # print(repr(awkward1.sum(listoffset_listoffset, axis=-2, keepdims=True)))
    # print()
    # print(repr(awkward1.sum(listoffset_listoffset, axis=-3, keepdims=True)))

    # print(regular_regular)
    # print(repr(awkward1.Array(regular_regular)))
    # print()
    # print(repr(awkward1.sum(regular_regular, axis=-1, keepdims=True)))
    # print()
    # print(repr(awkward1.sum(regular_regular, axis=-2, keepdims=True)))
    # print()
    # print(repr(awkward1.sum(regular_regular, axis=-3, keepdims=True)))

    # print(regular_listoffset)
    # print(repr(awkward1.Array(regular_listoffset)))
    # print()
    # print(repr(awkward1.sum(regular_listoffset, axis=-1, keepdims=True)))
    # print()
    # print(repr(awkward1.sum(regular_listoffset, axis=-2, keepdims=True)))
    # print()
    # print(repr(awkward1.sum(regular_listoffset, axis=-3, keepdims=True)))

    # print(listoffset_regular)
    # print(repr(awkward1.Array(listoffset_regular)))
    # print()
    # print(repr(awkward1.sum(listoffset_regular, axis=-1, keepdims=True)))
    # print()
    # print(repr(awkward1.sum(listoffset_regular, axis=-2, keepdims=True)))
    # print()
    # print(repr(awkward1.sum(listoffset_regular, axis=-3, keepdims=True)))

    # raise Exception("STOPPY")
