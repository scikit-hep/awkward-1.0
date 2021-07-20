# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    x = ak.layout.ListOffsetArray64(
        ak.layout.Index64(np.array([0, 2, 2, 4, 6])),
        ak.layout.NumpyArray(np.arange(8 * 8).reshape(8, -1)),
    )
    y = ak.layout.ListOffsetArray64(
        ak.layout.Index64(np.array([0, 2, 2, 4, 6])), ak.layout.NumpyArray(np.arange(8))
    )
    u, v = ak.broadcast_arrays(x, y)
    assert u.ndim == v.ndim
