# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import distutils.version
import types

import awkward as ak

checked_version = False


def register_and_check():
    global checked_version
    try:
        import jax
    except ImportError:
        raise ImportError(
            """install the 'jax' package with:

    pip install jax jaxlib --upgrade

or

    conda install jax jaxlib"""
        )
    else:
        if not checked_version and distutils.version.LooseVersion(
            jax.__version__
        ) < distutils.version.LooseVersion("0.2.7"):
            raise ImportError(
                "Awkward Array can only work with jax 0.2.7 or later "
                "(you have version {0})".format(jax.__version__)
            )
        checked_version = True
        register()


def register():
    import awkward._connect._jax.jax_utils


ak.jax = types.ModuleType("jax")
ak.jax.register = register_and_check
