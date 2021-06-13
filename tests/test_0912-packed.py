# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_indexed_numpy_array():
    index = ak.layout.Index64(np.array([0, 1, 2, 3, 6, 7, 8]))
    content = ak.layout.NumpyArray(np.arange(10))
    layout = ak.layout.IndexedArray64(index, content)

    packed = ak.packed(layout, highlevel=False)
    assert ak.to_list(packed) == ak.to_list(layout)

    assert isinstance(packed, ak.layout.NumpyArray)
    assert len(packed) == len(index)


def test_empty_array():
    layout = ak.layout.EmptyArray()
    assert ak.packed(layout, highlevel=False) is layout


def test_virtual_array():
    n_called = [0]

    def generate():
        n_called[0] += 1
        return ak.layout.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5]))

    generator = ak.layout.ArrayGenerator(
        generate, form=ak.forms.NumpyForm([], 8, "d"), length=5
    )
    layout = ak.layout.VirtualArray(generator)
    assert n_called[0] == 0
    packed = ak.packed(layout, highlevel=False)
    assert n_called[0] == 1

    assert isinstance(packed, ak.layout.NumpyArray)
    assert ak.to_list(packed) == ak.to_list(layout)
