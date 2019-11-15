# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys
import os
import json

import pytest
import numpy

import awkward1

content = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]));
offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 10, 10]))
listoffsetarray = awkward1.layout.ListOffsetArray64(offsets, content)
regulararray = awkward1.layout.RegularArray(listoffsetarray, 2)

def test_type():
    assert str(awkward1.typeof(regulararray)) == "3 * 2 * var * float64"

def test_getitem():
    assert awkward1.tolist(regulararray[0]) == [[0.0, 1.1, 2.2], []]
    assert awkward1.tolist(regulararray[1]) == [[3.3, 4.4], [5.5]]
    assert awkward1.tolist(regulararray[2]) == [[6.6, 7.7, 8.8, 9.9], []]
