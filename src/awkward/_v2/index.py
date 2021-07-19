# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumPyMetadata.instance()

_dtype_to_form = {
    np.dtype(np.int8): "i8",
    np.dtype(np.uint8): "u8",
    np.dtype(np.int32): "i32",
    np.dtype(np.uint32): "u32",
    np.dtype(np.int64): "i64",
}


class Index(object):
    _expected_dtype = None

    def __init__(self, data):
        self._nplike = ak.nplike.of(data)

        self._data = self._nplike.asarray(data, dtype=self._expected_dtype, order="C")
        if len(self._data.shape) != 1:
            raise TypeError("Index data must be one-dimensional")

        if self._data.dtype == np.dtype(np.int8):
            self.__class__ = Index8
        elif self._data.dtype == np.dtype(np.uint8):
            self.__class__ = IndexU8
        elif self._data.dtype == np.dtype(np.int32):
            self.__class__ = Index32
        elif self._data.dtype == np.dtype(np.uint32):
            self.__class__ = IndexU32
        elif self._data.dtype == np.dtype(np.int64):
            self.__class__ = Index64
        else:
            raise TypeError(
                "Index data must be int8, uint8, int32, uint32, int64, not {0}".format(
                    repr(self._data.dtype)
                )
            )

    @classmethod
    def zeros(cls, length, nplike, dtype):
        return Index(nplike.zeros(length, dtype=dtype))

    @classmethod
    def empty(cls, length, nplike, dtype):
        return Index(nplike.empty(length, dtype=dtype))

    @property
    def data(self):
        return self._data

    @property
    def nplike(self):
        return self._nplike

    @property
    def dtype(self):
        return self._data.dtype

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<Index dtype="]
        out.append(repr(str(self.dtype)))
        out.append(" len=")
        out.append(repr(str(len(self))))

        arraystr_lines = self._nplike.array_str(self._data, max_line_width=30).split(
            "\n"
        )
        if len(arraystr_lines) > 1:
            arraystr_lines = self._nplike.array_str(
                self._data, max_line_width=max(80 - len(indent) - 4, 40)
            ).split("\n")
            if len(arraystr_lines) > 5:
                arraystr_lines = arraystr_lines[:2] + [" ..."] + arraystr_lines[-2:]
            out.append(">\n" + indent + "    ")
            out.append(("\n" + indent + "    ").join(arraystr_lines))
            out.append("\n" + indent + "</Index>")
        else:
            if len(arraystr_lines) > 5:
                arraystr_lines = arraystr_lines[:2] + [" ..."] + arraystr_lines[-2:]
            out.append(">")
            out.append(arraystr_lines[0])
            out.append("</Index>")

        out.append(post)
        return "".join(out)

    @property
    def form(self):
        return _dtype_to_form[self._data.dtype]

    def __getitem__(self, where):
        return self._data[where]

    def __setitem__(self, where, what):
        self._data[where] = what

    def to64(self):
        return Index(self._data.astype(np.int64))

    def iscontiguous(self):
        return self._data.strides == (self._data.itemsize,)

    def __copy__(self):
        return Index(self._data.copy())

    def convert_to(self, nplike):
        return Index(nplike.asarray(self._data))


class Index8(Index):
    _expected_dtype = np.dtype(np.int8)


class IndexU8(Index):
    _expected_dtype = np.dtype(np.uint8)


class Index32(Index):
    _expected_dtype = np.dtype(np.int32)


class IndexU32(Index):
    _expected_dtype = np.dtype(np.uint32)


class Index64(Index):
    _expected_dtype = np.dtype(np.int64)
