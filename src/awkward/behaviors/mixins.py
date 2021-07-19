# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import sys

import awkward as ak


def mixin_class(registry):
    """
    Args:
        registry (dict): The destination behavior mapping registry. Typically,
            this would be the global registry #ak.behavior, but one may wish
            to register methods in an alternative way.

    This decorator can be used to register a behavior mixin class.

    Any inherited behaviors will automatically be made available to the decorated
    class.

    See the "Mixin decorators" section of #ak.behavior for further details.
    """

    def register(cls):
        name = cls.__name__
        record = type(
            name + "Record", (cls, ak.highlevel.Record), {"__module__": cls.__module__}
        )
        setattr(sys.modules[cls.__module__], name + "Record", record)
        registry[name] = record
        array = type(
            name + "Array", (cls, ak.highlevel.Array), {"__module__": cls.__module__}
        )
        setattr(sys.modules[cls.__module__], name + "Array", array)
        registry["*", name] = array
        for basecls in cls.mro():
            for method in basecls.__dict__.values():
                if hasattr(method, "_awkward_mixin"):
                    ufunc, rhs, transpose = method._awkward_mixin
                    if rhs is None:
                        registry.setdefault((ufunc, name), method)
                        continue
                    for rhs_name in list(rhs) + [name]:
                        registry.setdefault((ufunc, name, rhs_name), method)
                        if transpose is not None and rhs_name != name:
                            registry.setdefault((ufunc, rhs_name, name), transpose)
                    if basecls.__name__ in rhs:
                        rhs.add(name)
        return cls

    return register


def mixin_class_method(ufunc, rhs=None, transpose=True):
    """
    Args:
        ufunc (numpy.ufunc): A universal function (or NEP18 callable) that is
            hooked in Awkward Array, i.e. it can be the first argument of a behavior.
        rhs (Set[type] or None): Set of right-hand side argument types, optional
            if wrapping a unary function. The left-hand side is expected to
            always be `self` of the parent class.
        transpose (bool): If true, automatically create a transpose signature
            (only makes sense for binary ufuncs).

    This decorator can be used to register a mixin class method.

    Using this decorator ensures that derived classes that are declared with the
    #ak.mixin_class decorator will also have the behaviors that this class has.
    """

    def register(method):
        if not isinstance(rhs, (set, type(None))):
            raise ValueError(
                "expected a set of right-hand-side argument types"
                + ak._util.exception_suffix(__file__)
            )
        if transpose and rhs is not None:

            def transposed(left, right):
                return method(right, left)

            # make a copy of rhs, we will edit it later
            method._awkward_mixin = (ufunc, set(rhs), transposed)
        else:
            method._awkward_mixin = (ufunc, rhs, None)
        return method

    return register


__all__ = [
    x for x in list(globals()) if not x.startswith("_") and x not in ("sys", "ak")
]


def __dir__():
    return __all__
