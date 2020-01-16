# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys

import pytest
import numpy

import awkward1

def test_basic():
    content0 = awkward1.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]]).layout
    content1 = awkward1.Array(["one", "two", "three", "four", "five"]).layout
    tags = awkward1.layout.IndexU8(numpy.array([1, 1, 0, 0, 1, 0, 1, 1], dtype=numpy.uint8))
    index = awkward1.layout.Index32(numpy.array([0, 1, 0, 1, 2, 2, 4, 3], dtype=numpy.int32))
    array = awkward1.layout.UnionArrayU8_32(tags, index, [content0, content1])
    assert numpy.asarray(array.tags).tolist() == [1, 1, 0, 0, 1, 0, 1, 1]
    assert numpy.asarray(array.tags).dtype == numpy.dtype(numpy.uint8)
    assert numpy.asarray(array.index).tolist() == [0, 1, 0, 1, 2, 2, 4, 3]
    assert numpy.asarray(array.index).dtype == numpy.dtype(numpy.int32)
    assert type(array.contents) is list
    assert [awkward1.tolist(x) for x in array.contents] == [[[1.1, 2.2, 3.3], [], [4.4, 5.5]], ["one", "two", "three", "four", "five"]]
    assert array.numcontents == 2
    assert awkward1.tolist(array.content(0)) == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
    assert awkward1.tolist(array.content(1)) == ["one", "two", "three", "four", "five"]
    assert awkward1.tolist(array.project(0)) == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
    assert awkward1.tolist(array.project(1)) == ["one", "two", "three", "five", "four"]
    repr(array)

    assert awkward1.tolist(array[0]) == "one"
    assert awkward1.tolist(array[1]) == "two"
    assert awkward1.tolist(array[2]) == [1.1, 2.2, 3.3]
    assert awkward1.tolist(array[3]) == []
    assert awkward1.tolist(array[4]) == "three"
    assert awkward1.tolist(array[5]) == [4.4, 5.5]
    assert awkward1.tolist(array[6]) == "five"
    assert awkward1.tolist(array[7]) == "four"

    assert awkward1.tolist(array) == ["one", "two", [1.1, 2.2, 3.3], [], "three", [4.4, 5.5], "five", "four"]
    assert awkward1.tolist(array[1:-1]) == ["two", [1.1, 2.2, 3.3], [], "three", [4.4, 5.5], "five"]
    assert awkward1.tolist(array[2:-2]) == [[1.1, 2.2, 3.3], [], "three", [4.4, 5.5]]
    # assert awkward1.tolist(array[::2]) == ["one", [1.1, 2.2, 3.3], "three", "five"]
