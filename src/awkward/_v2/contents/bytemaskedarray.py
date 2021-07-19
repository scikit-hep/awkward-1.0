# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak
from awkward._v2.contents.content import Content
from awkward._v2.index import Index

np = ak.nplike.NumPyMetadata.instance()


class ByteMaskedArray(Content):
    def __init__(self, mask, content, valid_when, identifier=None, parameters=None):
        if not (isinstance(mask, Index) and mask.dtype == np.dtype(np.int8)):
            raise TypeError(
                "{0} 'mask' must be an Index with dtype=int8, not {1}".format(
                    type(self).__name__, repr(mask)
                )
            )
        if not isinstance(content, Content):
            raise TypeError(
                "{0} 'content' must be a Content subtype, not {1}".format(
                    type(self).__name__, repr(content)
                )
            )
        if not isinstance(valid_when, bool):
            raise TypeError(
                "{0} 'valid_when' must be boolean, not {1}".format(
                    type(self).__name__, repr(valid_when)
                )
            )
        if not len(mask) <= len(content):
            raise ValueError(
                "{0} len(mask) ({1}) must be <= len(content) ({2})".format(
                    type(self).__name__, len(mask), len(content)
                )
            )

        self._mask = mask
        self._content = content
        self._valid_when = valid_when
        self._init(identifier, parameters)

    @property
    def mask(self):
        return self._mask

    @property
    def content(self):
        return self._content

    @property
    def valid_when(self):
        return self._valid_when

    @property
    def form(self):
        return ak._v2.forms.ByteMaskedForm(
            self._mask.form,
            self._content.form,
            self._valid_when,
            has_identifier=self._identifier is not None,
            parameters=self._parameters,
            form_key=None,
        )

    def __len__(self):
        return len(self._mask)

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<ByteMaskedArray len="]
        out.append(repr(str(len(self))))
        out.append(" valid_when=")
        out.append(repr(str(self._valid_when)))
        out.append(">\n")
        out.append(self._mask._repr(indent + "    ", "<mask>", "</mask>\n"))
        out.append(self._content._repr(indent + "    ", "<content>", "</content>\n"))
        out.append(indent)
        out.append("</ByteMaskedArray>")
        out.append(post)
        return "".join(out)

    def _getitem_at(self, where):
        if where < 0:
            where += len(self)
        if 0 > where or where >= len(self):
            raise IndexError("array index out of bounds")
        if self._mask[where] == self._valid_when:
            return self._content[where]
        else:
            return None

    def _getitem_range(self, where):
        start, stop, step = where.indices(len(self))
        return ByteMaskedArray(
            Index(self._mask[start:stop]),
            self._content[start:stop],
            valid_when=self._valid_when,
        )

    def _getitem_field(self, where):
        return ByteMaskedArray(
            self._mask, self._content[where], valid_when=self._valid_when
        )

    def _getitem_fields(self, where):
        return ByteMaskedArray(
            self._mask, self._content[where], valid_when=self._valid_when
        )
