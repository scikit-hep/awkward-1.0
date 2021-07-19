# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import distutils.version

# NumPy 1.13.1 introduced NEP13, without which Awkward ufuncs won't work, which
# would be worse than lacking a feature: it would cause unexpected output.
# NumPy 1.17.0 introduced NEP18, which is optional (use ak.* instead of np.*).
import numpy

if distutils.version.LooseVersion(numpy.__version__) < distutils.version.LooseVersion(
    "1.13.1"
):
    raise ImportError("NumPy 1.13.1 or later required")

# NumPy-like alternatives
import awkward.nplike

# shims for C++ (now everything is compiled into one 'awkward._ext' module)
import awkward.layout
import awkward.types
import awkward.forms
import awkward.partition

# internal
import awkward._v2
import awkward._cpu_kernels
import awkward._libawkward
import awkward._util

# third-party connectors
import awkward._connect._numpy
import awkward._connect._numba
import awkward._connect._numexpr
import awkward._connect._autograd

# high-level interface
behavior = {}
from awkward.highlevel import Array
from awkward.highlevel import Record
from awkward.highlevel import ArrayBuilder

# third-party jax connectors
import awkward._connect._jax

# behaviors
from awkward.behaviors.mixins import *
from awkward.behaviors.string import *
from awkward.behaviors.categorical import *

# operations
from awkward.operations.convert import *
from awkward.operations.describe import *
from awkward.operations.structure import *
from awkward.operations.reducers import *

# version
__version__ = awkward._ext.__version__

# call C++ startup function
awkward._ext.startup()

__all__ = [
    x
    for x in list(globals())
    if not x.startswith("_") and x not in ("distutils", "numpy")
]


def __dir__():
    return __all__
