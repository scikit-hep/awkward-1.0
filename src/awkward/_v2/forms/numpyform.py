# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

from awkward._v2.forms.form import Form

import awkward as ak

np = ak.nplike.NumPyMetadata.instance()


def from_dtype(dtype, parameters=None):
    if dtype.subdtype is None:
        inner_shape = ()
    else:
        inner_shape = dtype.shape
        dtype = dtype.subdtype[0]

    if issubclass(dtype.type, (np.datetime64, np.timedelta64)):
        unit, step = np.datetime_data(dtype)
        if unit != "generic":
            unitstr = ("" if step == 1 else str(step)) + unit
            if parameters is None:
                parameters = {}
            else:
                parameters = parameters.copy()
            parameters["__unit__"] = unitstr
            dtype = np.dtype(dtype.type)

    primitive = ak._v2.types.numpytype._dtype_to_primitive.get(dtype)
    if primitive is None:
        raise TypeError(
            "dtype {0} is not supported; must be one of {1}".format(
                repr(dtype),
                ", ".join(repr(x) for x in ak._v2.types.numpytype._dtype_to_primitive),
            )
        )

    return NumpyForm(
        primitive=primitive,
        parameters=parameters,
        inner_shape=inner_shape,
    )


class NumpyForm(Form):
    def __init__(
        self,
        primitive,
        inner_shape=(),
        has_identifier=False,
        parameters=None,
        form_key=None,
    ):
        if primitive not in ak._v2.types.numpytype._primitive_to_dtype:
            raise TypeError(
                "{0} 'primitive' must be one of {1}, not {2}".format(
                    type(self).__name__,
                    ", ".join(
                        repr(x) for x in ak._v2.types.numpytype._primitive_to_dtype
                    ),
                    repr(primitive),
                )
            )
        if not isinstance(inner_shape, Iterable):
            raise TypeError(
                "{0} 'inner_shape' must be iterable, not {1}".format(
                    type(self).__name__, repr(inner_shape)
                )
            )

        self._primitive = primitive
        self._inner_shape = tuple(inner_shape)
        self._init(has_identifier, parameters, form_key)

    @property
    def primitive(self):
        return self._primitive

    @property
    def inner_shape(self):
        return self._inner_shape

    def __repr__(self):
        args = [repr(self._primitive)]
        if len(self._inner_shape) > 0:
            args.append("inner_shape=" + repr(self._inner_shape))
        args += self._repr_args()
        return "{0}({1})".format(type(self).__name__, ", ".join(args))

    def _tolist_part(self, verbose, toplevel):
        if (
            not verbose
            and not toplevel
            and len(self._inner_shape) == 0
            and not self._has_identifier
            and len(self._parameters) == 0
            and self._form_key is None
        ):
            return self._primitive

        else:
            out = {
                "class": "NumpyArray",
                "primitive": self._primitive,
            }
            if verbose or len(self._inner_shape) > 0:
                out["inner_shape"] = list(self._inner_shape)
            return self._tolist_extra(out, verbose)
