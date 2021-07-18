# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    a = ak.Array([{"this": 100}])
    b = ak.Array([{"this": 90, "that": 100}])
    c = ak.concatenate((a, b))

    with pytest.raises(ValueError):
        ak.unzip(c)
