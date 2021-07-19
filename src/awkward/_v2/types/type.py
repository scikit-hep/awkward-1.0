# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import json

import awkward as ak

np = ak.nplike.NumPyMetadata.instance()


class Type(object):
    @property
    def parameters(self):
        return self._parameters

    def parameter(self, key):
        if self._parameters is None:
            return None
        else:
            return self._parameters.get(key)

    @property
    def typestr(self):
        return self._typestr

    _str_parameters_exclude = ("__categorical__",)

    def _str_categorical_begin(self):
        if self.parameter("__categorical__") is not None:
            return "categorical[type="
        else:
            return ""

    def _str_categorical_end(self):
        if self.parameter("__categorical__") is not None:
            return "]"
        else:
            return ""

    def _str_parameters(self):
        out = []
        if self._parameters is not None:
            for k, v in self._parameters.items():
                if k not in self._str_parameters_exclude:
                    out.append(json.dumps(k) + ": " + json.dumps(v))

        if len(out) == 0:
            return None
        else:
            return "parameters={" + ", ".join(out) + "}"

    def _repr_args(self):
        out = []

        if self._parameters is not None and len(self._parameters) > 0:
            out.append("parameters=" + repr(self._parameters))

        if self._typestr is not None:
            out.append("typestr=" + repr(self._typestr))

        return out
