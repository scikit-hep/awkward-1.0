# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import awkward as ak
from awkward._v2.contents.content import Content


np = ak.nplike.NumPyMetadata.instance()


class Record(object):
    def __init__(self, array, at):
        if not isinstance(array, ak._v2.contents.recordarray.RecordArray):
            raise TypeError(
                "Record 'array' must be a RecordArray, not {0}".format(repr(array))
            )
        if not ak._util.isint(at):
            raise TypeError(
                "Record 'at' must be an integer, not {0}".format(repr(array))
            )
        if 0 <= at < len(array):
            self._array = array
            self._at = at
        else:
            raise ValueError(
                "Record 'at' must be >= 0 and < len(array) == {0}, not {1}".format(
                    len(array), at
                )
            )

    @property
    def array(self):
        return self._array

    @property
    def at(self):
        return self._at

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<Record at="]
        out.append(repr(str(self._at)))
        out.append(">\n")
        out.append(self._array._repr(indent + "    ", "<array>", "</array>\n"))
        out.append(indent)
        out.append("</Record>")
        out.append(post)
        return "".join(out)

    def __getitem__(self, where):
        if ak._util.isint(where):
            raise IndexError("scalar Record cannot be sliced by an integer")

        elif isinstance(where, slice):
            raise IndexError("scalar Record cannot be sliced by a range slice (`:`)")

        elif ak._util.isstr(where):
            return self._getitem_field(where)

        elif where is np.newaxis:
            raise IndexError("scalar Record cannot be sliced by np.newaxis (`None`)")

        elif where is Ellipsis:
            raise IndexError("scalar Record cannot be sliced by an ellipsis (`...`)")

        elif isinstance(where, tuple):
            raise NotImplementedError("needs _getitem_next")

        elif isinstance(where, ak.highlevel.Array):
            raise IndexError("scalar Record cannot be sliced by an array")

        elif isinstance(where, Content):
            raise IndexError("scalar Record cannot be sliced by an array")

        elif isinstance(where, Iterable) and all(ak._util.isstr(x) for x in where):
            return self._getitem_fields(where)

        elif isinstance(where, Iterable):
            raise IndexError("scalar Record cannot be sliced by an array")

        else:
            raise TypeError(
                "only field name (str) or names (non-tuple iterable of str) "
                "are valid indices for slicing a scalar record, not\n\n    "
                + repr(where)
            )

    def _getitem_field(self, where):
        return self._array._getitem_field(where)._getitem_at(self._at)

    def _getitem_fields(self, where):
        return self._array._getitem_fields(where)._getitem_at(self._at)
