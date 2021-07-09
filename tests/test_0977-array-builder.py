# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_array_builder():
    builder = ak.ArrayBuilder()

    builder.real(1.1)
    builder.real(2.2)
    builder.real(3.3)

    assert ak.to_list(builder.snapshot()) == [1.1, 2.2, 3.3]


def test_array_bool_builder():
    builder = ak.ArrayBuilder()

    builder.boolean(True)
    builder.boolean(False)
    builder.boolean(True)

    assert ak.to_list(builder.snapshot()) == [True, False, True]
