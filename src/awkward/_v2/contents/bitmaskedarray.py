# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak
from awkward._v2.contents.content import Content
from awkward._v2.index import Index
from awkward._v2.contents.bytemaskedarray import ByteMaskedArray

np = ak.nplike.NumPyMetadata.instance()


class BitMaskedArray(Content):
    def __init__(
        self,
        mask,
        content,
        valid_when,
        length,
        lsb_order,
        identifier=None,
        parameters=None,
    ):
        if not (isinstance(mask, Index) and mask.dtype == np.dtype(np.uint8)):
            raise TypeError(
                "{0} 'mask' must be an Index with dtype=uint8, not {1}".format(
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
        if not ak._util.isint(length):
            raise TypeError(
                "{0} 'length' must be an integer, not {1}".format(
                    type(self).__name__, repr(length)
                )
            )
        if not isinstance(lsb_order, bool):
            raise TypeError(
                "{0} 'lsb_order' must be boolean, not {1}".format(
                    type(self).__name__, repr(lsb_order)
                )
            )
        if not length <= len(mask) * 8:
            raise ValueError(
                "{0} 'length' ({1}) must be <= len(mask) * 8 ({2})".format(
                    type(self).__name__, length, len(mask) * 8
                )
            )
        if not length <= len(content):
            raise ValueError(
                "{0} 'length' ({1}) must be <= len(content) ({2})".format(
                    type(self).__name__, length, len(content)
                )
            )

        self._mask = mask
        self._content = content
        self._valid_when = valid_when
        self._length = length
        self._lsb_order = lsb_order
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
    def lsb_order(self):
        return self._lsb_order

    @property
    def form(self):
        return ak._v2.forms.BitMaskedForm(
            self._mask.form,
            self._content.form,
            self._valid_when,
            self._lsb_order,
            has_identifier=self._identifier is not None,
            parameters=self._parameters,
            form_key=None,
        )

    def __len__(self):
        return self._length

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<BitMaskedArray len="]
        out.append(repr(str(len(self))))
        out.append(" valid_when=")
        out.append(repr(str(self._valid_when)))
        out.append(" lsb_order=")
        out.append(repr(str(self._lsb_order)))
        out.append(">\n")
        out.append(self._mask._repr(indent + "    ", "<mask>", "</mask>\n"))
        out.append(self._content._repr(indent + "    ", "<content>", "</content>\n"))
        out.append(indent)
        out.append("</BitMaskedArray>")
        out.append(post)
        return "".join(out)

    def _getitem_at(self, where):
        if where < 0:
            where += len(self)
        if 0 > where or where >= len(self):
            raise IndexError("array index out of bounds")
        if self._lsb_order:
            bit = bool(self._mask[where // 8] & (1 << (where % 8)))
        else:
            bit = bool(self._mask[where // 8] & (128 >> (where % 8)))
        if bit == self._valid_when:
            return self._content[where]
        else:
            return None

    def _getitem_range(self, where):
        # In general, slices must convert BitMaskedArray to ByteMaskedArray.
        if self._lsb_order:
            bytemask = (
                self._mask.nplike.unpackbits(self._mask)
                .reshape(-1, 8)[:, ::-1]
                .reshape(-1)
                .view(np.int8)
            )
        else:
            bytemask = self._mask.nplike.unpackbits(self._mask).view(np.int8)
        start, stop, step = where.indices(len(self))
        return ByteMaskedArray(
            Index(bytemask[start:stop]),
            self._content[start:stop],
            valid_when=self._valid_when,
        )

    def _getitem_field(self, where):
        return BitMaskedArray(
            self._mask,
            self._content[where],
            valid_when=self._valid_when,
            length=self._length,
            lsb_order=self._lsb_order,
        )

    def _getitem_fields(self, where):
        return BitMaskedArray(
            self._mask,
            self._content[where],
            valid_when=self._valid_when,
            length=self._length,
            lsb_order=self._lsb_order,
        )
