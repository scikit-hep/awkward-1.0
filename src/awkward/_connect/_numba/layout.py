# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import json
import ctypes

import numba

import awkward as ak

np = ak.nplike.NumPyMetadata.instance()
numpy = ak.nplike.NumPy.instance()


@numba.extending.typeof_impl.register(ak.layout.NumpyArray)
@numba.extending.typeof_impl.register(ak.layout.RegularArray)
@numba.extending.typeof_impl.register(ak.layout.ListArray32)
@numba.extending.typeof_impl.register(ak.layout.ListArrayU32)
@numba.extending.typeof_impl.register(ak.layout.ListArray64)
@numba.extending.typeof_impl.register(ak.layout.ListOffsetArray32)
@numba.extending.typeof_impl.register(ak.layout.ListOffsetArrayU32)
@numba.extending.typeof_impl.register(ak.layout.ListOffsetArray64)
@numba.extending.typeof_impl.register(ak.layout.IndexedArray32)
@numba.extending.typeof_impl.register(ak.layout.IndexedArrayU32)
@numba.extending.typeof_impl.register(ak.layout.IndexedArray64)
@numba.extending.typeof_impl.register(ak.layout.IndexedOptionArray32)
@numba.extending.typeof_impl.register(ak.layout.IndexedOptionArray64)
@numba.extending.typeof_impl.register(ak.layout.ByteMaskedArray)
@numba.extending.typeof_impl.register(ak.layout.BitMaskedArray)
@numba.extending.typeof_impl.register(ak.layout.UnmaskedArray)
@numba.extending.typeof_impl.register(ak.layout.RecordArray)
@numba.extending.typeof_impl.register(ak.layout.UnionArray8_32)
@numba.extending.typeof_impl.register(ak.layout.UnionArray8_U32)
@numba.extending.typeof_impl.register(ak.layout.UnionArray8_64)
@numba.extending.typeof_impl.register(ak.layout.VirtualArray)
@numba.extending.typeof_impl.register(ak.layout.Index8)
@numba.extending.typeof_impl.register(ak.layout.IndexU8)
@numba.extending.typeof_impl.register(ak.layout.Index32)
@numba.extending.typeof_impl.register(ak.layout.IndexU32)
@numba.extending.typeof_impl.register(ak.layout.Index64)
@numba.extending.typeof_impl.register(ak.layout.Record)
def fake_typeof(obj, c):
    raise TypeError(
        "{0} objects cannot be passed directly into Numba-compiled functions; "
        "construct a high-level ak.Array or ak.Record instead".format(
            type(obj).__name__
        )
    )


def typeof(obj):
    if isinstance(obj, ak.layout.NumpyArray):
        return typeof_NumpyArray(obj)

    elif isinstance(obj, ak.layout.RegularArray):
        return typeof_RegularArray(obj)

    elif isinstance(
        obj,
        (
            ak.layout.ListArray32,
            ak.layout.ListArrayU32,
            ak.layout.ListArray64,
            ak.layout.ListOffsetArray32,
            ak.layout.ListOffsetArrayU32,
            ak.layout.ListOffsetArray64,
        ),
    ):
        return typeof_ListArray(obj)

    elif isinstance(
        obj,
        (
            ak.layout.IndexedArray32,
            ak.layout.IndexedArrayU32,
            ak.layout.IndexedArray64,
        ),
    ):
        return typeof_IndexedArray(obj)

    elif isinstance(
        obj,
        (
            ak.layout.IndexedOptionArray32,
            ak.layout.IndexedOptionArray64,
        ),
    ):
        return typeof_IndexedOptionArray(obj)

    elif isinstance(obj, ak.layout.ByteMaskedArray):
        return typeof_ByteMaskedArray(obj)

    elif isinstance(obj, ak.layout.BitMaskedArray):
        return typeof_BitMaskedArray(obj)

    elif isinstance(obj, ak.layout.UnmaskedArray):
        return typeof_UnmaskedArray(obj)

    elif isinstance(obj, ak.layout.RecordArray):
        return typeof_RecordArray(obj)

    elif isinstance(
        obj,
        (
            ak.layout.UnionArray8_32,
            ak.layout.UnionArray8_U32,
            ak.layout.UnionArray8_64,
        ),
    ):
        return typeof_UnionArray(obj)

    elif isinstance(obj, ak.layout.VirtualArray):
        return typeof_VirtualArray(obj)

    elif isinstance(
        obj,
        (
            ak.layout.Identities32,
            ak.layout.Identities64,
        ),
    ):
        raise NotImplementedError(
            "Awkward Identities are not yet supported for functions compiled by Numba"
        )

    elif isinstance(
        obj,
        (
            ak.layout.Index8,
            ak.layout.IndexU8,
            ak.layout.Index32,
            ak.layout.IndexU32,
            ak.layout.Index64,
        ),
    ):
        raise RuntimeError(
            "Awkward Indexes should not be used directly in functions compiled by Numba"
        )

    else:
        return numba.typeof(obj)


def typeof_NumpyArray(obj):
    t = numba.typeof(ak.nplike.of(obj).asarray(obj))
    return NumpyArrayType(
        numba.types.Array(t.dtype, t.ndim, "A"),
        typeof(obj.identities),
        obj.parameters,
    )


def typeof_RegularArray(obj):
    return RegularArrayType(
        typeof(obj.content),
        obj.size,
        typeof(obj.identities),
        obj.parameters,
    )


def typeof_ListArray(obj):
    return ListArrayType(
        numba.typeof(ak.nplike.of(obj.starts).asarray(obj.starts)),
        typeof(obj.content),
        typeof(obj.identities),
        obj.parameters,
    )


def typeof_IndexedArray(obj):
    return IndexedArrayType(
        numba.typeof(ak.nplike.of(obj.index).asarray(obj.index)),
        typeof(obj.content),
        typeof(obj.identities),
        obj.parameters,
    )


def typeof_IndexedOptionArray(obj):
    return IndexedOptionArrayType(
        numba.typeof(ak.nplike.of(obj.index).asarray(obj.index)),
        typeof(obj.content),
        typeof(obj.identities),
        obj.parameters,
    )


def typeof_ByteMaskedArray(obj):
    return ByteMaskedArrayType(
        numba.typeof(ak.nplike.of(obj.mask).asarray(obj.mask)),
        typeof(obj.content),
        obj.valid_when,
        typeof(obj.identities),
        obj.parameters,
    )


def typeof_BitMaskedArray(obj):
    return BitMaskedArrayType(
        numba.typeof(ak.nplike.of(obj.mask).asarray(obj.mask)),
        typeof(obj.content),
        obj.valid_when,
        obj.lsb_order,
        typeof(obj.identities),
        obj.parameters,
    )


def typeof_UnmaskedArray(obj):
    return UnmaskedArrayType(
        typeof(obj.content), typeof(obj.identities), obj.parameters
    )


def typeof_RecordArray(obj):
    return RecordArrayType(
        tuple(typeof(x) for x in obj.contents),
        obj.recordlookup,
        typeof(obj.identities),
        obj.parameters,
    )


def typeof_UnionArray(obj):
    return UnionArrayType(
        numba.typeof(ak.nplike.of(obj.tags).asarray(obj.tags)),
        numba.typeof(ak.nplike.of(obj.index).asarray(obj.index)),
        tuple(typeof(x) for x in obj.contents),
        typeof(obj.identities),
        obj.parameters,
    )


def typeof_VirtualArray(obj):
    if obj.form.form is None:
        raise ValueError(
            "VirtualArrays without a known 'form' can't be used in Numba"
            + ak._util.exception_suffix(__file__)
        )
    if obj.form.has_identities:
        raise NotImplementedError(
            "TODO: identities in VirtualArray" + ak._util.exception_suffix(__file__)
        )
    return VirtualArrayType(obj.form.form, numba.none, obj.parameters)


class ContentType(numba.types.Type):
    @classmethod
    def tolookup_identities(cls, layout, positions, sharedptrs, arrays):
        if layout.identities is None:
            positions.append(-1)
            sharedptrs.append(None)
        else:
            arrays.append(ak.nplike.of(layout.identities).asarray(layout.identities))
            positions.append(arrays[-1])
            sharedptrs.append(None)

    @classmethod
    def form_tolookup_identities(cls, form, positions, sharedptrs, arrays):
        if not form.has_identities:
            positions.append(-1)
            sharedptrs.append(None)
        else:
            arrays.append(None)
            positions.append(0)
            sharedptrs.append(None)

    @classmethod
    def from_form_identities(cls, form):
        if not form.has_identities:
            return numba.none
        else:
            raise NotImplementedError(
                "TODO: identities in VirtualArray" + ak._util.exception_suffix(__file__)
            )

    @classmethod
    def from_form_index(cls, index_string):
        if index_string == "i8":
            return numba.types.Array(numba.int8, 1, "C")
        elif index_string == "u8":
            return numba.types.Array(numba.uint8, 1, "C")
        elif index_string == "i32":
            return numba.types.Array(numba.int32, 1, "C")
        elif index_string == "u32":
            return numba.types.Array(numba.uint32, 1, "C")
        elif index_string == "i64":
            return numba.types.Array(numba.int64, 1, "C")
        else:
            raise AssertionError(
                "unrecognized Form index type: {0}".format(index_string)
                + ak._util.exception_suffix(__file__)
            )

    def form_fill_identities(self, pos, layout, lookup):
        identities = layout.identities
        if identities is not None:
            lookup.arrayptr[pos + self.IDENTITIES] = (
                ak.nplike.of(identities).asarray(identities).ctypes.data
            )

    def IndexOf(self, arraytype):
        if arraytype.dtype.bitwidth == 8 and arraytype.dtype.signed:
            return ak.layout.Index8
        elif arraytype.dtype.bitwidth == 8:
            return ak.layout.IndexU8
        elif arraytype.dtype.bitwidth == 32 and arraytype.dtype.signed:
            return ak.layout.Index32
        elif arraytype.dtype.bitwidth == 32:
            return ak.layout.IndexU32
        elif arraytype.dtype.bitwidth == 64 and arraytype.dtype.signed:
            return ak.layout.Index64
        else:
            raise AssertionError(
                "no Index* type for array: {0}".format(arraytype)
                + ak._util.exception_suffix(__file__)
            )

    def getitem_at_check(self, viewtype):
        typer = ak._util.numba_array_typer(viewtype.type, viewtype.behavior)
        if typer is None:
            return self.getitem_at(viewtype)
        else:
            return typer(viewtype)

    def getitem_range(self, viewtype):
        return ak._connect._numba.arrayview.wrap(self, viewtype, None)

    def getitem_field(self, viewtype, key):
        if self.hasfield(key):
            return ak._connect._numba.arrayview.wrap(
                self, viewtype, viewtype.fields + (key,)
            )
        else:
            raise TypeError(
                "array does not have a field with key {0}".format(repr(key))
                + ak._util.exception_suffix(__file__)
            )

    def lower_getitem_at_check(
        self,
        context,
        builder,
        rettype,
        viewtype,
        viewval,
        viewproxy,
        attype,
        atval,
        wrapneg,
        checkbounds,
    ):
        lower = ak._util.numba_array_lower(viewtype.type, viewtype.behavior)
        if lower is not None:
            atval = regularize_atval(
                context, builder, viewproxy, attype, atval, wrapneg, checkbounds
            )
            return lower(
                context, builder, rettype, viewtype, viewval, viewproxy, attype, atval
            )
        else:
            return self.lower_getitem_at(
                context,
                builder,
                rettype,
                viewtype,
                viewval,
                viewproxy,
                attype,
                atval,
                wrapneg,
                checkbounds,
            )

    def lower_getitem_range(
        self,
        context,
        builder,
        rettype,
        viewtype,
        viewval,
        viewproxy,
        start,
        stop,
        wrapneg,
    ):
        length = builder.sub(viewproxy.stop, viewproxy.start)

        regular_start = numba.core.cgutils.alloca_once_value(builder, start)
        regular_stop = numba.core.cgutils.alloca_once_value(builder, stop)

        if wrapneg:
            with builder.if_then(
                builder.icmp_signed("<", start, context.get_constant(numba.intp, 0))
            ):
                builder.store(builder.add(start, length), regular_start)
            with builder.if_then(
                builder.icmp_signed("<", stop, context.get_constant(numba.intp, 0))
            ):
                builder.store(builder.add(stop, length), regular_stop)

        with builder.if_then(
            builder.icmp_signed(
                "<", builder.load(regular_start), context.get_constant(numba.intp, 0)
            )
        ):
            builder.store(context.get_constant(numba.intp, 0), regular_start)
        with builder.if_then(
            builder.icmp_signed(">", builder.load(regular_start), length)
        ):
            builder.store(length, regular_start)

        with builder.if_then(
            builder.icmp_signed(
                "<", builder.load(regular_stop), builder.load(regular_start)
            )
        ):
            builder.store(builder.load(regular_start), regular_stop)
        with builder.if_then(
            builder.icmp_signed(">", builder.load(regular_stop), length)
        ):
            builder.store(length, regular_stop)

        proxyout = context.make_helper(builder, rettype)
        proxyout.pos = viewproxy.pos
        proxyout.start = builder.add(viewproxy.start, builder.load(regular_start))
        proxyout.stop = builder.add(viewproxy.start, builder.load(regular_stop))
        proxyout.arrayptrs = viewproxy.arrayptrs
        proxyout.sharedptrs = viewproxy.sharedptrs
        proxyout.pylookup = viewproxy.pylookup
        return proxyout._getvalue()

    def lower_getitem_field(self, context, builder, viewtype, viewval, key):
        return viewval


def type_bitwidth(numbatype):
    if isinstance(numbatype, numba.types.Boolean):
        return 8
    else:
        return numbatype.bitwidth


def posat(context, builder, pos, offset):
    return builder.add(pos, context.get_constant(numba.intp, offset))


def getat(context, builder, baseptr, offset, rettype=None):
    ptrtype = None
    if rettype is not None:
        ptrtype = context.get_value_type(numba.types.CPointer(rettype))
        bitwidth = type_bitwidth(rettype)
    else:
        bitwidth = numba.intp.bitwidth
    byteoffset = builder.mul(offset, context.get_constant(numba.intp, bitwidth // 8))
    out = builder.load(
        numba.core.cgutils.pointer_add(builder, baseptr, byteoffset, ptrtype)
    )
    if rettype is not None and isinstance(rettype, numba.types.Boolean):
        return builder.icmp_signed(
            "!=",
            out,
            context.get_constant(numba.int8, 0),
        )
    else:
        return out


def regularize_atval(context, builder, viewproxy, attype, atval, wrapneg, checkbounds):
    atval = ak._connect._numba.castint(context, builder, attype, numba.intp, atval)

    if not attype.signed:
        wrapneg = False

    if wrapneg or checkbounds:
        length = builder.sub(viewproxy.stop, viewproxy.start)

        if wrapneg:
            regular_atval = numba.core.cgutils.alloca_once_value(builder, atval)
            with builder.if_then(
                builder.icmp_signed("<", atval, context.get_constant(numba.intp, 0))
            ):
                builder.store(builder.add(atval, length), regular_atval)
            atval = builder.load(regular_atval)

        if checkbounds:
            with builder.if_then(
                builder.or_(
                    builder.icmp_signed(
                        "<", atval, context.get_constant(numba.intp, 0)
                    ),
                    builder.icmp_signed(">=", atval, length),
                )
            ):
                context.call_conv.return_user_exc(
                    builder, ValueError, ("slice index out of bounds",)
                )

    return ak._connect._numba.castint(context, builder, atval.type, numba.intp, atval)


class NumpyArrayType(ContentType):
    IDENTITIES = 0
    ARRAY = 1

    @classmethod
    def tolookup(cls, layout, positions, sharedptrs, arrays):
        array = ak.nplike.of(layout).ascontiguousarray(layout)
        assert len(array.shape) == 1
        pos = len(positions)
        cls.tolookup_identities(layout, positions, sharedptrs, arrays)
        sharedptrs[-1] = layout._persistent_shared_ptr
        positions.append(array)
        sharedptrs.append(None)
        arrays.append(array)
        return pos

    @classmethod
    def form_tolookup(cls, form, positions, sharedptrs, arrays):
        if len(form.inner_shape) != 0:
            raise NotImplementedError(
                "NumpyForm is multidimensional; TODO: convert to RegularForm,"
                " just as NumpyArrays are converted to RegularArrays"
                + ak._util.exception_suffix(__file__)
            )
        pos = len(positions)
        cls.form_tolookup_identities(form, positions, sharedptrs, arrays)
        sharedptrs[-1] = 0
        positions.append(0)
        sharedptrs.append(None)
        arrays.append(0)
        return pos

    @classmethod
    def from_form(cls, form):
        if len(form.inner_shape) != 0:
            raise NotImplementedError(
                "NumpyForm is multidimensional; TODO: convert to RegularForm,"
                " just as NumpyArrays are converted to RegularArrays"
                + ak._util.exception_suffix(__file__)
            )
        if form.primitive == "float64":
            arraytype = numba.types.Array(numba.float64, 1, "A")
        elif form.primitive == "float32":
            arraytype = numba.types.Array(numba.float32, 1, "A")
        elif form.primitive == "int64":
            arraytype = numba.types.Array(numba.int64, 1, "A")
        elif form.primitive == "uint64":
            arraytype = numba.types.Array(numba.uint64, 1, "A")
        elif form.primitive == "int32":
            arraytype = numba.types.Array(numba.int32, 1, "A")
        elif form.primitive == "uint32":
            arraytype = numba.types.Array(numba.uint32, 1, "A")
        elif form.primitive == "int16":
            arraytype = numba.types.Array(numba.int16, 1, "A")
        elif form.primitive == "uint16":
            arraytype = numba.types.Array(numba.uint16, 1, "A")
        elif form.primitive == "int8":
            arraytype = numba.types.Array(numba.int8, 1, "A")
        elif form.primitive == "uint8":
            arraytype = numba.types.Array(numba.uint8, 1, "A")
        elif form.primitive == "bool":
            arraytype = numba.types.Array(numba.boolean, 1, "A")
        else:
            raise ValueError(
                "unrecognized NumpyForm.primitive type: {0}".format(form.primitive)
                + ak._util.exception_suffix(__file__)
            )
        return NumpyArrayType(
            arraytype, cls.from_form_identities(form), form.parameters
        )

    def __init__(self, arraytype, identitiestype, parameters):
        super(NumpyArrayType, self).__init__(
            name="ak.NumpyArrayType({0}, {1}, {2})".format(
                arraytype.name, identitiestype.name, json.dumps(parameters)
            )
        )
        self.arraytype = arraytype
        self.identitiestype = identitiestype
        self.parameters = parameters

    def form_fill(self, pos, layout, lookup):
        lookup.sharedptrs_hold[pos] = layout._persistent_shared_ptr
        lookup.sharedptrs[pos] = lookup.sharedptrs_hold[pos].ptr()
        self.form_fill_identities(pos, layout, lookup)

        lookup.original_positions[pos + self.ARRAY] = ak.nplike.of(layout).asarray(
            layout
        )
        lookup.arrayptrs[pos + self.ARRAY] = lookup.original_positions[
            pos + self.ARRAY
        ].ctypes.data

    def tolayout(self, lookup, pos, fields):
        assert fields == ()
        return ak.layout.NumpyArray(
            lookup.original_positions[pos + self.ARRAY], parameters=self.parameters
        )

    def hasfield(self, key):
        return False

    def getitem_at(self, viewtype):
        return self.arraytype.dtype

    def lower_getitem_at(
        self,
        context,
        builder,
        rettype,
        viewtype,
        viewval,
        viewproxy,
        attype,
        atval,
        wrapneg,
        checkbounds,
    ):
        whichpos = posat(context, builder, viewproxy.pos, self.ARRAY)
        arrayptr = getat(context, builder, viewproxy.arrayptrs, whichpos)
        atval = regularize_atval(
            context, builder, viewproxy, attype, atval, wrapneg, checkbounds
        )
        arraypos = builder.add(viewproxy.start, atval)
        return getat(context, builder, arrayptr, arraypos, rettype=rettype)

    @property
    def ndim(self):
        return self.arraytype.ndim

    @property
    def inner_dtype(self):
        return self.arraytype.dtype

    @property
    def is_optiontype(self):
        return False

    @property
    def is_recordtype(self):
        return False


class RegularArrayType(ContentType):
    IDENTITIES = 0
    CONTENT = 1

    @classmethod
    def tolookup(cls, layout, positions, sharedptrs, arrays):
        pos = len(positions)
        cls.tolookup_identities(layout, positions, sharedptrs, arrays)
        sharedptrs[-1] = layout._persistent_shared_ptr
        positions.append(None)
        sharedptrs.append(None)
        positions[pos + cls.CONTENT] = ak._connect._numba.arrayview.tolookup(
            layout.content, positions, sharedptrs, arrays
        )
        return pos

    @classmethod
    def form_tolookup(cls, form, positions, sharedptrs, arrays):
        pos = len(positions)
        cls.form_tolookup_identities(form, positions, sharedptrs, arrays)
        sharedptrs[-1] = 0
        positions.append(None)
        sharedptrs.append(None)
        positions[pos + cls.CONTENT] = ak._connect._numba.arrayview.tolookup(
            form.content, positions, sharedptrs, arrays
        )
        return pos

    @classmethod
    def from_form(cls, form):
        return RegularArrayType(
            ak._connect._numba.arrayview.tonumbatype(form.content),
            form.size,
            cls.from_form_identities(form),
            form.parameters,
        )

    def __init__(self, contenttype, size, identitiestype, parameters):
        super(RegularArrayType, self).__init__(
            name="ak.RegularArrayType({0}, {1}, {2}, {3})".format(
                contenttype.name, size, identitiestype.name, json.dumps(parameters)
            )
        )
        self.contenttype = contenttype
        self.size = size
        self.identitiestype = identitiestype
        self.parameters = parameters

    def form_fill(self, pos, layout, lookup):
        lookup.sharedptrs_hold[pos] = layout._persistent_shared_ptr
        lookup.sharedptrs[pos] = lookup.sharedptrs_hold[pos].ptr()
        self.form_fill_identities(pos, layout, lookup)

        self.contenttype.form_fill(
            lookup.arrayptrs[pos + self.CONTENT], layout.content, lookup
        )

    def tolayout(self, lookup, pos, fields):
        content = self.contenttype.tolayout(
            lookup, lookup.positions[pos + self.CONTENT], fields
        )
        return ak.layout.RegularArray(content, self.size, 0, parameters=self.parameters)

    def hasfield(self, key):
        return self.contenttype.hasfield(key)

    def getitem_at(self, viewtype):
        return ak._connect._numba.arrayview.wrap(self.contenttype, viewtype, None)

    def lower_getitem_at(
        self,
        context,
        builder,
        rettype,
        viewtype,
        viewval,
        viewproxy,
        attype,
        atval,
        wrapneg,
        checkbounds,
    ):
        whichpos = posat(context, builder, viewproxy.pos, self.CONTENT)
        nextpos = getat(context, builder, viewproxy.arrayptrs, whichpos)

        atval = regularize_atval(
            context, builder, viewproxy, attype, atval, wrapneg, checkbounds
        )

        size = context.get_constant(numba.intp, self.size)
        start = builder.mul(builder.add(viewproxy.start, atval), size)
        stop = builder.add(start, size)

        proxyout = context.make_helper(builder, rettype)
        proxyout.pos = nextpos
        proxyout.start = start
        proxyout.stop = stop
        proxyout.arrayptrs = viewproxy.arrayptrs
        proxyout.sharedptrs = viewproxy.sharedptrs
        proxyout.pylookup = viewproxy.pylookup
        return proxyout._getvalue()

    @property
    def ndim(self):
        return 1 + self.contenttype.ndim

    @property
    def inner_dtype(self):
        return self.contenttype.inner_dtype

    @property
    def is_optiontype(self):
        return False

    @property
    def is_recordtype(self):
        return False


class ListArrayType(ContentType):
    IDENTITIES = 0
    STARTS = 1
    STOPS = 2
    CONTENT = 3

    @classmethod
    def tolookup(cls, layout, positions, sharedptrs, arrays):
        if isinstance(
            layout,
            (
                ak.layout.ListArray32,
                ak.layout.ListArrayU32,
                ak.layout.ListArray64,
            ),
        ):
            starts = ak.nplike.of(layout.starts).asarray(layout.starts)
            stops = ak.nplike.of(layout.stops).asarray(layout.stops)
        elif isinstance(
            layout,
            (
                ak.layout.ListOffsetArray32,
                ak.layout.ListOffsetArrayU32,
                ak.layout.ListOffsetArray64,
            ),
        ):
            offsets = ak.nplike.of(layout.offsets).asarray(layout.offsets)
            starts = offsets[:-1]
            stops = offsets[1:]

        pos = len(positions)
        cls.tolookup_identities(layout, positions, sharedptrs, arrays)
        sharedptrs[-1] = layout._persistent_shared_ptr
        positions.append(starts)
        sharedptrs.append(None)
        arrays.append(starts)
        positions.append(stops)
        sharedptrs.append(None)
        arrays.append(stops)
        positions.append(None)
        sharedptrs.append(None)
        positions[pos + cls.CONTENT] = ak._connect._numba.arrayview.tolookup(
            layout.content, positions, sharedptrs, arrays
        )
        return pos

    @classmethod
    def form_tolookup(cls, form, positions, sharedptrs, arrays):
        pos = len(positions)
        cls.form_tolookup_identities(form, positions, sharedptrs, arrays)
        sharedptrs[-1] = 0
        positions.append(0)
        sharedptrs.append(None)
        arrays.append(0)
        positions.append(0)
        sharedptrs.append(None)
        arrays.append(0)
        positions.append(None)
        sharedptrs.append(None)
        positions[pos + cls.CONTENT] = ak._connect._numba.arrayview.tolookup(
            form.content, positions, sharedptrs, arrays
        )
        return pos

    @classmethod
    def from_form(cls, form):
        return ListArrayType(
            cls.from_form_index(
                form.starts if isinstance(form, ak.forms.ListForm) else form.offsets
            ),
            ak._connect._numba.arrayview.tonumbatype(form.content),
            cls.from_form_identities(form),
            form.parameters,
        )

    def __init__(self, indextype, contenttype, identitiestype, parameters):
        super(ListArrayType, self).__init__(
            name="ak.ListArrayType({0}, {1}, {2}, {3})".format(
                indextype.name,
                contenttype.name,
                identitiestype.name,
                json.dumps(parameters),
            )
        )
        self.indextype = indextype
        self.contenttype = contenttype
        self.identitiestype = identitiestype
        self.parameters = parameters

    def form_fill(self, pos, layout, lookup):
        lookup.sharedptrs_hold[pos] = layout._persistent_shared_ptr
        lookup.sharedptrs[pos] = lookup.sharedptrs_hold[pos].ptr()
        self.form_fill_identities(pos, layout, lookup)

        if isinstance(
            layout,
            (
                ak.layout.ListArray32,
                ak.layout.ListArrayU32,
                ak.layout.ListArray64,
            ),
        ):
            starts = ak.nplike.of(layout.starts).asarray(layout.starts)
            stops = ak.nplike.of(layout.stops).asarray(layout.stops)
        elif isinstance(
            layout,
            (
                ak.layout.ListOffsetArray32,
                ak.layout.ListOffsetArrayU32,
                ak.layout.ListOffsetArray64,
            ),
        ):
            offsets = ak.nplike.of(layout.offsets).asarray(layout.offsets)
            starts = offsets[:-1]
            stops = offsets[1:]

        lookup.original_positions[pos + self.STARTS] = starts
        lookup.original_positions[pos + self.STOPS] = stops
        lookup.arrayptrs[pos + self.STARTS] = starts.ctypes.data
        lookup.arrayptrs[pos + self.STOPS] = stops.ctypes.data

        self.contenttype.form_fill(
            lookup.arrayptrs[pos + self.CONTENT], layout.content, lookup
        )

    def ListArrayOf(self):
        if self.indextype.dtype.bitwidth == 32 and self.indextype.dtype.signed:
            return ak.layout.ListArray32
        elif self.indextype.dtype.bitwidth == 32:
            return ak.layout.ListArrayU32
        elif self.indextype.dtype.bitwidth == 64 and self.indextype.dtype.signed:
            return ak.layout.ListArray64
        else:
            raise AssertionError(
                "no ListArray* type for array: {0}".format(self.indextype)
                + ak._util.exception_suffix(__file__)
            )

    def tolayout(self, lookup, pos, fields):
        starts = self.IndexOf(self.indextype)(
            lookup.original_positions[pos + self.STARTS]
        )
        stops = self.IndexOf(self.indextype)(
            lookup.original_positions[pos + self.STOPS]
        )
        content = self.contenttype.tolayout(
            lookup, lookup.positions[pos + self.CONTENT], fields
        )
        return self.ListArrayOf()(starts, stops, content, parameters=self.parameters)

    def hasfield(self, key):
        return self.contenttype.hasfield(key)

    def getitem_at(self, viewtype):
        return ak._connect._numba.arrayview.wrap(self.contenttype, viewtype, None)

    def lower_getitem_at(
        self,
        context,
        builder,
        rettype,
        viewtype,
        viewval,
        viewproxy,
        attype,
        atval,
        wrapneg,
        checkbounds,
    ):
        whichpos = posat(context, builder, viewproxy.pos, self.CONTENT)
        nextpos = getat(context, builder, viewproxy.arrayptrs, whichpos)

        atval = regularize_atval(
            context, builder, viewproxy, attype, atval, wrapneg, checkbounds
        )

        startspos = posat(context, builder, viewproxy.pos, self.STARTS)
        startsptr = getat(context, builder, viewproxy.arrayptrs, startspos)
        startsarraypos = builder.add(viewproxy.start, atval)
        start = getat(
            context, builder, startsptr, startsarraypos, rettype=self.indextype.dtype
        )

        stopspos = posat(context, builder, viewproxy.pos, self.STOPS)
        stopsptr = getat(context, builder, viewproxy.arrayptrs, stopspos)
        stopsarraypos = builder.add(viewproxy.start, atval)
        stop = getat(
            context, builder, stopsptr, stopsarraypos, rettype=self.indextype.dtype
        )

        proxyout = context.make_helper(builder, rettype)
        proxyout.pos = nextpos
        proxyout.start = ak._connect._numba.castint(
            context, builder, self.indextype.dtype, numba.intp, start
        )
        proxyout.stop = ak._connect._numba.castint(
            context, builder, self.indextype.dtype, numba.intp, stop
        )
        proxyout.arrayptrs = viewproxy.arrayptrs
        proxyout.sharedptrs = viewproxy.sharedptrs
        proxyout.pylookup = viewproxy.pylookup
        return proxyout._getvalue()

    @property
    def ndim(self):
        return 1 + self.contenttype.ndim

    @property
    def inner_dtype(self):
        return self.contenttype.inner_dtype

    @property
    def is_optiontype(self):
        return False

    @property
    def is_recordtype(self):
        return False


class IndexedArrayType(ContentType):
    IDENTITIES = 0
    INDEX = 1
    CONTENT = 2

    @classmethod
    def tolookup(cls, layout, positions, sharedptrs, arrays):
        pos = len(positions)
        cls.tolookup_identities(layout, positions, sharedptrs, arrays)
        sharedptrs[-1] = layout._persistent_shared_ptr
        arrays.append(ak.nplike.of(layout.index).asarray(layout.index))
        positions.append(arrays[-1])
        sharedptrs.append(None)
        positions.append(None)
        sharedptrs.append(None)
        positions[pos + cls.CONTENT] = ak._connect._numba.arrayview.tolookup(
            layout.content, positions, sharedptrs, arrays
        )
        return pos

    @classmethod
    def form_tolookup(cls, form, positions, sharedptrs, arrays):
        pos = len(positions)
        cls.form_tolookup_identities(form, positions, sharedptrs, arrays)
        sharedptrs[-1] = 0
        arrays.append(0)
        positions.append(0)
        sharedptrs.append(None)
        positions.append(None)
        sharedptrs.append(None)
        positions[pos + cls.CONTENT] = ak._connect._numba.arrayview.tolookup(
            form.content, positions, sharedptrs, arrays
        )
        return pos

    @classmethod
    def from_form(cls, form):
        return IndexedArrayType(
            cls.from_form_index(form.index),
            ak._connect._numba.arrayview.tonumbatype(form.content),
            cls.from_form_identities(form),
            form.parameters,
        )

    def __init__(self, indextype, contenttype, identitiestype, parameters):
        super(IndexedArrayType, self).__init__(
            name="ak.IndexedArrayType({0}, {1}, {2}, {3})".format(
                indextype.name,
                contenttype.name,
                identitiestype.name,
                json.dumps(parameters),
            )
        )
        self.indextype = indextype
        self.contenttype = contenttype
        self.identitiestype = identitiestype
        self.parameters = parameters

    def form_fill(self, pos, layout, lookup):
        lookup.sharedptrs_hold[pos] = layout._persistent_shared_ptr
        lookup.sharedptrs[pos] = lookup.sharedptrs_hold[pos].ptr()
        self.form_fill_identities(pos, layout, lookup)

        index = ak.nplike.of(layout.index).asarray(layout.index)
        lookup.original_positions[pos + self.INDEX] = index
        lookup.arrayptrs[pos + self.INDEX] = index.ctypes.data

        self.contenttype.form_fill(
            lookup.arrayptrs[pos + self.CONTENT], layout.content, lookup
        )

    def IndexedArrayOf(self):
        if self.indextype.dtype.bitwidth == 32 and self.indextype.dtype.signed:
            return ak.layout.IndexedArray32
        elif self.indextype.dtype.bitwidth == 32:
            return ak.layout.IndexedArrayU32
        elif self.indextype.dtype.bitwidth == 64 and self.indextype.dtype.signed:
            return ak.layout.IndexedArray64
        else:
            raise AssertionError(
                "no IndexedArray* type for array: {0}".format(self.indextype)
                + ak._util.exception_suffix(__file__)
            )

    def tolayout(self, lookup, pos, fields):
        index = self.IndexOf(self.indextype)(
            lookup.original_positions[pos + self.INDEX]
        )
        content = self.contenttype.tolayout(
            lookup, lookup.positions[pos + self.CONTENT], fields
        )
        return self.IndexedArrayOf()(index, content, parameters=self.parameters)

    def hasfield(self, key):
        return self.contenttype.hasfield(key)

    def getitem_at(self, viewtype):
        viewtype = ak._connect._numba.arrayview.wrap(self.contenttype, viewtype, None)
        return self.contenttype.getitem_at_check(viewtype)

    def lower_getitem_at(
        self,
        context,
        builder,
        rettype,
        viewtype,
        viewval,
        viewproxy,
        attype,
        atval,
        wrapneg,
        checkbounds,
    ):
        whichpos = posat(context, builder, viewproxy.pos, self.CONTENT)
        nextpos = getat(context, builder, viewproxy.arrayptrs, whichpos)

        atval = regularize_atval(
            context, builder, viewproxy, attype, atval, wrapneg, checkbounds
        )

        indexpos = posat(context, builder, viewproxy.pos, self.INDEX)
        indexptr = getat(context, builder, viewproxy.arrayptrs, indexpos)
        indexarraypos = builder.add(viewproxy.start, atval)
        nextat = getat(
            context, builder, indexptr, indexarraypos, rettype=self.indextype.dtype
        )

        nextviewtype = ak._connect._numba.arrayview.wrap(
            self.contenttype, viewtype, None
        )
        proxynext = context.make_helper(builder, nextviewtype)
        proxynext.pos = nextpos
        proxynext.start = context.get_constant(numba.intp, 0)
        proxynext.stop = builder.add(
            ak._connect._numba.castint(
                context, builder, self.indextype.dtype, numba.intp, nextat
            ),
            context.get_constant(numba.intp, 1),
        )
        proxynext.arrayptrs = viewproxy.arrayptrs
        proxynext.sharedptrs = viewproxy.sharedptrs
        proxynext.pylookup = viewproxy.pylookup

        return self.contenttype.lower_getitem_at_check(
            context,
            builder,
            rettype,
            nextviewtype,
            proxynext._getvalue(),
            proxynext,
            numba.intp,
            nextat,
            False,
            False,
        )

    @property
    def ndim(self):
        return self.contenttype.ndim

    @property
    def inner_dtype(self):
        return self.contenttype.inner_dtype

    @property
    def is_optiontype(self):
        return False

    @property
    def is_recordtype(self):
        return self.contenttype.is_recordtype


class IndexedOptionArrayType(ContentType):
    IDENTITIES = 0
    INDEX = 1
    CONTENT = 2

    @classmethod
    def tolookup(cls, layout, positions, sharedptrs, arrays):
        pos = len(positions)
        cls.tolookup_identities(layout, positions, sharedptrs, arrays)
        sharedptrs[-1] = layout._persistent_shared_ptr
        arrays.append(ak.nplike.of(layout.index).asarray(layout.index))
        positions.append(arrays[-1])
        sharedptrs.append(None)
        positions.append(None)
        sharedptrs.append(None)
        positions[pos + cls.CONTENT] = ak._connect._numba.arrayview.tolookup(
            layout.content, positions, sharedptrs, arrays
        )
        return pos

    @classmethod
    def form_tolookup(cls, form, positions, sharedptrs, arrays):
        pos = len(positions)
        cls.form_tolookup_identities(form, positions, sharedptrs, arrays)
        sharedptrs[-1] = 0
        arrays.append(0)
        positions.append(0)
        sharedptrs.append(None)
        positions.append(None)
        sharedptrs.append(None)
        positions[pos + cls.CONTENT] = ak._connect._numba.arrayview.tolookup(
            form.content, positions, sharedptrs, arrays
        )
        return pos

    @classmethod
    def from_form(cls, form):
        return IndexedOptionArrayType(
            cls.from_form_index(form.index),
            ak._connect._numba.arrayview.tonumbatype(form.content),
            cls.from_form_identities(form),
            form.parameters,
        )

    def __init__(self, indextype, contenttype, identitiestype, parameters):
        super(IndexedOptionArrayType, self).__init__(
            name="ak.IndexedOptionArrayType({0}, {1}, {2}, {3})".format(
                indextype.name,
                contenttype.name,
                identitiestype.name,
                json.dumps(parameters),
            )
        )
        self.indextype = indextype
        self.contenttype = contenttype
        self.identitiestype = identitiestype
        self.parameters = parameters

    def form_fill(self, pos, layout, lookup):
        lookup.sharedptrs_hold[pos] = layout._persistent_shared_ptr
        lookup.sharedptrs[pos] = lookup.sharedptrs_hold[pos].ptr()
        self.form_fill_identities(pos, layout, lookup)

        index = ak.nplike.of(layout.index).asarray(layout.index)
        lookup.original_positions[pos + self.INDEX] = index
        lookup.arrayptrs[pos + self.INDEX] = index.ctypes.data

        self.contenttype.form_fill(
            lookup.arrayptrs[pos + self.CONTENT], layout.content, lookup
        )

    def IndexedOptionArrayOf(self):
        if self.indextype.dtype.bitwidth == 32 and self.indextype.dtype.signed:
            return ak.layout.IndexedOptionArray32
        elif self.indextype.dtype.bitwidth == 64 and self.indextype.dtype.signed:
            return ak.layout.IndexedOptionArray64
        else:
            raise AssertionError(
                "no IndexedOptionArray* type for array: {0}".format(self.indextype)
                + ak._util.exception_suffix(__file__)
            )

    def tolayout(self, lookup, pos, fields):
        index = self.IndexOf(self.indextype)(
            lookup.original_positions[pos + self.INDEX]
        )
        content = self.contenttype.tolayout(
            lookup, lookup.positions[pos + self.CONTENT], fields
        )
        return self.IndexedOptionArrayOf()(index, content, parameters=self.parameters)

    def hasfield(self, key):
        return self.contenttype.hasfield(key)

    def getitem_at(self, viewtype):
        viewtype = ak._connect._numba.arrayview.wrap(self.contenttype, viewtype, None)
        return numba.types.optional(self.contenttype.getitem_at_check(viewtype))

    def lower_getitem_at(
        self,
        context,
        builder,
        rettype,
        viewtype,
        viewval,
        viewproxy,
        attype,
        atval,
        wrapneg,
        checkbounds,
    ):
        whichpos = posat(context, builder, viewproxy.pos, self.CONTENT)
        nextpos = getat(context, builder, viewproxy.arrayptrs, whichpos)

        atval = regularize_atval(
            context, builder, viewproxy, attype, atval, wrapneg, checkbounds
        )

        indexpos = posat(context, builder, viewproxy.pos, self.INDEX)
        indexptr = getat(context, builder, viewproxy.arrayptrs, indexpos)
        indexarraypos = builder.add(viewproxy.start, atval)
        nextat = getat(
            context, builder, indexptr, indexarraypos, rettype=self.indextype.dtype
        )

        output = context.make_helper(builder, rettype)

        with builder.if_else(
            builder.icmp_signed(
                "<", nextat, context.get_constant(self.indextype.dtype, 0)
            )
        ) as (isnone, isvalid):
            with isnone:
                output.valid = numba.core.cgutils.false_bit
                output.data = numba.core.cgutils.get_null_value(output.data.type)

            with isvalid:
                nextviewtype = ak._connect._numba.arrayview.wrap(
                    self.contenttype, viewtype, None
                )
                proxynext = context.make_helper(builder, nextviewtype)
                proxynext.pos = nextpos
                proxynext.start = context.get_constant(numba.intp, 0)
                proxynext.stop = builder.add(
                    ak._connect._numba.castint(
                        context, builder, self.indextype.dtype, numba.intp, nextat
                    ),
                    context.get_constant(numba.intp, 1),
                )
                proxynext.arrayptrs = viewproxy.arrayptrs
                proxynext.sharedptrs = viewproxy.sharedptrs
                proxynext.pylookup = viewproxy.pylookup

                outdata = self.contenttype.lower_getitem_at_check(
                    context,
                    builder,
                    rettype.type,
                    nextviewtype,
                    proxynext._getvalue(),
                    proxynext,
                    numba.intp,
                    nextat,
                    False,
                    False,
                )

                output.valid = numba.core.cgutils.true_bit
                output.data = outdata

        return output._getvalue()

    @property
    def ndim(self):
        return self.contenttype.ndim

    @property
    def inner_dtype(self):
        return None

    @property
    def is_optiontype(self):
        return True

    @property
    def is_recordtype(self):
        return self.contenttype.is_recordtype


class ByteMaskedArrayType(ContentType):
    IDENTITIES = 0
    MASK = 1
    CONTENT = 2

    @classmethod
    def tolookup(cls, layout, positions, sharedptrs, arrays):
        pos = len(positions)
        cls.tolookup_identities(layout, positions, sharedptrs, arrays)
        sharedptrs[-1] = layout._persistent_shared_ptr
        arrays.append(ak.nplike.of(layout.mask).asarray(layout.mask))
        positions.append(arrays[-1])
        sharedptrs.append(None)
        positions.append(None)
        sharedptrs.append(None)
        positions[pos + cls.CONTENT] = ak._connect._numba.arrayview.tolookup(
            layout.content, positions, sharedptrs, arrays
        )
        return pos

    @classmethod
    def form_tolookup(cls, form, positions, sharedptrs, arrays):
        pos = len(positions)
        cls.form_tolookup_identities(form, positions, sharedptrs, arrays)
        sharedptrs[-1] = 0
        arrays.append(0)
        positions.append(0)
        sharedptrs.append(None)
        positions.append(None)
        sharedptrs.append(None)
        positions[pos + cls.CONTENT] = ak._connect._numba.arrayview.tolookup(
            form.content, positions, sharedptrs, arrays
        )
        return pos

    @classmethod
    def from_form(cls, form):
        return ByteMaskedArrayType(
            cls.from_form_index(form.mask),
            ak._connect._numba.arrayview.tonumbatype(form.content),
            form.valid_when,
            cls.from_form_identities(form),
            form.parameters,
        )

    def __init__(self, masktype, contenttype, valid_when, identitiestype, parameters):
        super(ByteMaskedArrayType, self).__init__(
            name="ak.ByteMaskedArrayType({0}, {1}, {2}, {3}, "
            "{4})".format(
                masktype.name,
                contenttype.name,
                valid_when,
                identitiestype.name,
                json.dumps(parameters),
            )
        )
        self.masktype = masktype
        self.contenttype = contenttype
        self.valid_when = valid_when
        self.identitiestype = identitiestype
        self.parameters = parameters

    def form_fill(self, pos, layout, lookup):
        lookup.sharedptrs_hold[pos] = layout._persistent_shared_ptr
        lookup.sharedptrs[pos] = lookup.sharedptrs_hold[pos].ptr()
        self.form_fill_identities(pos, layout, lookup)

        mask = ak.nplike.of(layout.mask).asarray(layout.mask)
        lookup.original_positions[pos + self.MASK] = mask
        lookup.arrayptrs[pos + self.MASK] = mask.ctypes.data

        self.contenttype.form_fill(
            lookup.arrayptrs[pos + self.CONTENT], layout.content, lookup
        )

    def tolayout(self, lookup, pos, fields):
        mask = self.IndexOf(self.masktype)(lookup.original_positions[pos + self.MASK])
        content = self.contenttype.tolayout(
            lookup, lookup.positions[pos + self.CONTENT], fields
        )
        return ak.layout.ByteMaskedArray(
            mask, content, self.valid_when, parameters=self.parameters
        )

    def hasfield(self, key):
        return self.contenttype.hasfield(key)

    def getitem_at(self, viewtype):
        viewtype = ak._connect._numba.arrayview.wrap(self.contenttype, viewtype, None)
        return numba.types.optional(self.contenttype.getitem_at_check(viewtype))

    def lower_getitem_at(
        self,
        context,
        builder,
        rettype,
        viewtype,
        viewval,
        viewproxy,
        attype,
        atval,
        wrapneg,
        checkbounds,
    ):
        whichpos = posat(context, builder, viewproxy.pos, self.CONTENT)
        nextpos = getat(context, builder, viewproxy.arrayptrs, whichpos)

        atval = regularize_atval(
            context, builder, viewproxy, attype, atval, wrapneg, checkbounds
        )

        maskpos = posat(context, builder, viewproxy.pos, self.MASK)
        maskptr = getat(context, builder, viewproxy.arrayptrs, maskpos)
        maskarraypos = builder.add(viewproxy.start, atval)
        byte = getat(
            context, builder, maskptr, maskarraypos, rettype=self.masktype.dtype
        )

        output = context.make_helper(builder, rettype)

        with builder.if_else(
            builder.icmp_signed(
                "==",
                builder.icmp_signed("!=", byte, context.get_constant(numba.int8, 0)),
                context.get_constant(numba.int8, int(self.valid_when)),
            )
        ) as (isvalid, isnone):
            with isvalid:
                nextviewtype = ak._connect._numba.arrayview.wrap(
                    self.contenttype, viewtype, None
                )
                proxynext = context.make_helper(builder, nextviewtype)
                proxynext.pos = nextpos
                proxynext.start = viewproxy.start
                proxynext.stop = viewproxy.stop
                proxynext.arrayptrs = viewproxy.arrayptrs
                proxynext.sharedptrs = viewproxy.sharedptrs
                proxynext.pylookup = viewproxy.pylookup

                outdata = self.contenttype.lower_getitem_at_check(
                    context,
                    builder,
                    rettype.type,
                    nextviewtype,
                    proxynext._getvalue(),
                    proxynext,
                    numba.intp,
                    atval,
                    False,
                    False,
                )

                output.valid = numba.core.cgutils.true_bit
                output.data = outdata

            with isnone:
                output.valid = numba.core.cgutils.false_bit
                output.data = numba.core.cgutils.get_null_value(output.data.type)

        return output._getvalue()

    @property
    def ndim(self):
        return self.contenttype.ndim

    @property
    def inner_dtype(self):
        return None

    @property
    def is_optiontype(self):
        return True

    @property
    def is_recordtype(self):
        return self.contenttype.is_recordtype


class BitMaskedArrayType(ContentType):
    IDENTITIES = 0
    MASK = 1
    CONTENT = 2

    @classmethod
    def tolookup(cls, layout, positions, sharedptrs, arrays):
        pos = len(positions)
        cls.tolookup_identities(layout, positions, sharedptrs, arrays)
        sharedptrs[-1] = layout._persistent_shared_ptr
        arrays.append(ak.nplike.of(layout.mask).asarray(layout.mask))
        positions.append(arrays[-1])
        sharedptrs.append(None)
        positions.append(None)
        sharedptrs.append(None)
        positions[pos + cls.CONTENT] = ak._connect._numba.arrayview.tolookup(
            layout.content, positions, sharedptrs, arrays
        )
        return pos

    @classmethod
    def form_tolookup(cls, form, positions, sharedptrs, arrays):
        pos = len(positions)
        cls.form_tolookup_identities(form, positions, sharedptrs, arrays)
        sharedptrs[-1] = 0
        arrays.append(0)
        positions.append(0)
        sharedptrs.append(None)
        positions.append(None)
        sharedptrs.append(None)
        positions[pos + cls.CONTENT] = ak._connect._numba.arrayview.tolookup(
            form.content, positions, sharedptrs, arrays
        )
        return pos

    @classmethod
    def from_form(cls, form):
        return BitMaskedArrayType(
            cls.from_form_index(form.mask),
            ak._connect._numba.arrayview.tonumbatype(form.content),
            form.valid_when,
            form.lsb_order,
            cls.from_form_identities(form),
            form.parameters,
        )

    def __init__(
        self, masktype, contenttype, valid_when, lsb_order, identitiestype, parameters
    ):
        super(BitMaskedArrayType, self).__init__(
            name="ak.BitMaskedArrayType({0}, {1}, {2}, {3}, {4}, "
            "{5})".format(
                masktype.name,
                contenttype.name,
                valid_when,
                lsb_order,
                identitiestype.name,
                json.dumps(parameters),
            )
        )
        self.masktype = masktype
        self.contenttype = contenttype
        self.valid_when = valid_when
        self.lsb_order = lsb_order
        self.identitiestype = identitiestype
        self.parameters = parameters

    def form_fill(self, pos, layout, lookup):
        lookup.sharedptrs_hold[pos] = layout._persistent_shared_ptr
        lookup.sharedptrs[pos] = lookup.sharedptrs_hold[pos].ptr()
        self.form_fill_identities(pos, layout, lookup)

        mask = ak.nplike.of(layout.mask).asarray(layout.mask)
        lookup.original_positions[pos + self.MASK] = mask
        lookup.arrayptrs[pos + self.MASK] = mask.ctypes.data

        self.contenttype.form_fill(
            lookup.arrayptrs[pos + self.CONTENT], layout.content, lookup
        )

    def tolayout(self, lookup, pos, fields):
        mask = self.IndexOf(self.masktype)(lookup.original_positions[pos + self.MASK])
        content = self.contenttype.tolayout(
            lookup, lookup.positions[pos + self.CONTENT], fields
        )
        return ak.layout.BitMaskedArray(
            mask,
            content,
            self.valid_when,
            len(content),
            self.lsb_order,
            parameters=self.parameters,
        )

    def hasfield(self, key):
        return self.contenttype.hasfield(key)

    def getitem_at(self, viewtype):
        viewtype = ak._connect._numba.arrayview.wrap(self.contenttype, viewtype, None)
        return numba.types.optional(self.contenttype.getitem_at_check(viewtype))

    def lower_getitem_at(
        self,
        context,
        builder,
        rettype,
        viewtype,
        viewval,
        viewproxy,
        attype,
        atval,
        wrapneg,
        checkbounds,
    ):
        whichpos = posat(context, builder, viewproxy.pos, self.CONTENT)
        nextpos = getat(context, builder, viewproxy.arrayptrs, whichpos)

        atval = regularize_atval(
            context, builder, viewproxy, attype, atval, wrapneg, checkbounds
        )
        bitatval = builder.sdiv(atval, context.get_constant(numba.intp, 8))
        shiftval = ak._connect._numba.castint(
            context,
            builder,
            numba.intp,
            numba.uint8,
            builder.srem(atval, context.get_constant(numba.intp, 8)),
        )

        maskpos = posat(context, builder, viewproxy.pos, self.MASK)
        maskptr = getat(context, builder, viewproxy.arrayptrs, maskpos)
        maskarraypos = builder.add(viewproxy.start, bitatval)
        byte = getat(
            context, builder, maskptr, maskarraypos, rettype=self.masktype.dtype
        )
        if self.lsb_order:
            # ((byte >> ((uint8_t)shift)) & ((uint8_t)1))
            asbool = builder.and_(
                builder.lshr(byte, shiftval), context.get_constant(numba.uint8, 1)
            )
        else:
            # ((byte << ((uint8_t)shift)) & ((uint8_t)128))
            asbool = builder.and_(
                builder.shl(byte, shiftval), context.get_constant(numba.uint8, 128)
            )

        output = context.make_helper(builder, rettype)

        with builder.if_else(
            builder.icmp_signed(
                "==",
                builder.icmp_signed("!=", asbool, context.get_constant(numba.uint8, 0)),
                context.get_constant(numba.uint8, int(self.valid_when)),
            )
        ) as (isvalid, isnone):
            with isvalid:
                nextviewtype = ak._connect._numba.arrayview.wrap(
                    self.contenttype, viewtype, None
                )
                proxynext = context.make_helper(builder, nextviewtype)
                proxynext.pos = nextpos
                proxynext.start = viewproxy.start
                proxynext.stop = viewproxy.stop
                proxynext.arrayptrs = viewproxy.arrayptrs
                proxynext.sharedptrs = viewproxy.sharedptrs
                proxynext.pylookup = viewproxy.pylookup

                outdata = self.contenttype.lower_getitem_at_check(
                    context,
                    builder,
                    rettype.type,
                    nextviewtype,
                    proxynext._getvalue(),
                    proxynext,
                    numba.intp,
                    atval,
                    False,
                    False,
                )

                output.valid = numba.core.cgutils.true_bit
                output.data = outdata

            with isnone:
                output.valid = numba.core.cgutils.false_bit
                output.data = numba.core.cgutils.get_null_value(output.data.type)

        return output._getvalue()

    @property
    def ndim(self):
        return self.contenttype.ndim

    @property
    def inner_dtype(self):
        return None

    @property
    def is_optiontype(self):
        return True

    @property
    def is_recordtype(self):
        return self.contenttype.is_recordtype


class UnmaskedArrayType(ContentType):
    IDENTITIES = 0
    CONTENT = 1

    @classmethod
    def tolookup(cls, layout, positions, sharedptrs, arrays):
        pos = len(positions)
        cls.tolookup_identities(layout, positions, sharedptrs, arrays)
        sharedptrs[-1] = layout._persistent_shared_ptr
        positions.append(None)
        sharedptrs.append(None)
        positions[pos + cls.CONTENT] = ak._connect._numba.arrayview.tolookup(
            layout.content, positions, sharedptrs, arrays
        )
        return pos

    @classmethod
    def form_tolookup(cls, form, positions, sharedptrs, arrays):
        pos = len(positions)
        cls.form_tolookup_identities(form, positions, sharedptrs, arrays)
        sharedptrs[-1] = 0
        positions.append(None)
        sharedptrs.append(None)
        positions[pos + cls.CONTENT] = ak._connect._numba.arrayview.tolookup(
            form.content, positions, sharedptrs, arrays
        )
        return pos

    @classmethod
    def from_form(cls, form):
        return UnmaskedArrayType(
            ak._connect._numba.arrayview.tonumbatype(form.content),
            cls.from_form_identities(form),
            form.parameters,
        )

    def __init__(self, contenttype, identitiestype, parameters):
        super(UnmaskedArrayType, self).__init__(
            name="ak.UnmaskedArrayType({0}, {1}, {2})".format(
                contenttype.name, identitiestype.name, json.dumps(parameters)
            )
        )
        self.contenttype = contenttype
        self.identitiestype = identitiestype
        self.parameters = parameters

    def form_fill(self, pos, layout, lookup):
        lookup.sharedptrs_hold[pos] = layout._persistent_shared_ptr
        lookup.sharedptrs[pos] = lookup.sharedptrs_hold[pos].ptr()
        self.form_fill_identities(pos, layout, lookup)

        self.contenttype.form_fill(
            lookup.arrayptrs[pos + self.CONTENT], layout.content, lookup
        )

    def tolayout(self, lookup, pos, fields):
        content = self.contenttype.tolayout(
            lookup, lookup.positions[pos + self.CONTENT], fields
        )
        return ak.layout.UnmaskedArray(content, parameters=self.parameters)

    def hasfield(self, key):
        return self.contenttype.hasfield(key)

    def getitem_at(self, viewtype):
        viewtype = ak._connect._numba.arrayview.wrap(self.contenttype, viewtype, None)
        return numba.types.optional(self.contenttype.getitem_at_check(viewtype))

    def lower_getitem_at(
        self,
        context,
        builder,
        rettype,
        viewtype,
        viewval,
        viewproxy,
        attype,
        atval,
        wrapneg,
        checkbounds,
    ):
        whichpos = posat(context, builder, viewproxy.pos, self.CONTENT)
        nextpos = getat(context, builder, viewproxy.arrayptrs, whichpos)

        atval = regularize_atval(
            context, builder, viewproxy, attype, atval, wrapneg, checkbounds
        )

        output = context.make_helper(builder, rettype)

        nextviewtype = ak._connect._numba.arrayview.wrap(
            self.contenttype, viewtype, None
        )
        proxynext = context.make_helper(builder, nextviewtype)
        proxynext.pos = nextpos
        proxynext.start = viewproxy.start
        proxynext.stop = viewproxy.stop
        proxynext.arrayptrs = viewproxy.arrayptrs
        proxynext.sharedptrs = viewproxy.sharedptrs
        proxynext.pylookup = viewproxy.pylookup

        outdata = self.contenttype.lower_getitem_at_check(
            context,
            builder,
            rettype.type,
            nextviewtype,
            proxynext._getvalue(),
            proxynext,
            numba.intp,
            atval,
            False,
            False,
        )

        output.valid = numba.core.cgutils.true_bit
        output.data = outdata

        return output._getvalue()

    @property
    def ndim(self):
        return self.contenttype.ndim

    @property
    def inner_dtype(self):
        return None

    @property
    def is_optiontype(self):
        return True

    @property
    def is_recordtype(self):
        return self.contenttype.is_recordtype


class RecordArrayType(ContentType):
    IDENTITIES = 0
    CONTENTS = 1

    @classmethod
    def tolookup(cls, layout, positions, sharedptrs, arrays):
        pos = len(positions)
        cls.tolookup_identities(layout, positions, sharedptrs, arrays)
        sharedptrs[-1] = layout._persistent_shared_ptr
        positions.extend([None] * layout.numfields)
        sharedptrs.extend([None] * layout.numfields)
        for i, content in enumerate(layout.contents):
            positions[pos + cls.CONTENTS + i] = ak._connect._numba.arrayview.tolookup(
                content, positions, sharedptrs, arrays
            )
        return pos

    @classmethod
    def form_tolookup(cls, form, positions, sharedptrs, arrays):
        pos = len(positions)
        cls.form_tolookup_identities(form, positions, sharedptrs, arrays)
        sharedptrs[-1] = 0
        positions.extend([None] * form.numfields)
        sharedptrs.extend([None] * form.numfields)
        if form.istuple:
            for i, (_, content) in enumerate(form.contents.items()):
                positions[
                    pos + cls.CONTENTS + i
                ] = ak._connect._numba.arrayview.tolookup(
                    content, positions, sharedptrs, arrays
                )
        else:
            for i, (_, content) in enumerate(form.contents.items()):
                positions[
                    pos + cls.CONTENTS + i
                ] = ak._connect._numba.arrayview.tolookup(
                    content, positions, sharedptrs, arrays
                )
        return pos

    @classmethod
    def from_form(cls, form):
        contents = []
        if form.istuple:
            recordlookup = None
            for x in form.contents.values():
                contents.append(ak._connect._numba.arrayview.tonumbatype(x))
        else:
            recordlookup = []
            for n, x in form.contents.items():
                contents.append(ak._connect._numba.arrayview.tonumbatype(x))
                recordlookup.append(n)

        return RecordArrayType(
            contents, recordlookup, cls.from_form_identities(form), form.parameters
        )

    def __init__(self, contenttypes, recordlookup, identitiestype, parameters):
        super(RecordArrayType, self).__init__(
            name="ak.RecordArrayType(({0}{1}), ({2}), {3}, {4})".format(
                ", ".join(x.name for x in contenttypes),
                "," if len(contenttypes) == 1 else "",
                "None" if recordlookup is None else repr(tuple(recordlookup)),
                identitiestype.name,
                json.dumps(parameters),
            )
        )
        self.contenttypes = contenttypes
        self.recordlookup = recordlookup
        self.identitiestype = identitiestype
        self.parameters = parameters

    def form_fill(self, pos, layout, lookup):
        lookup.sharedptrs_hold[pos] = layout._persistent_shared_ptr
        lookup.sharedptrs[pos] = lookup.sharedptrs_hold[pos].ptr()
        self.form_fill_identities(pos, layout, lookup)

        if self.recordlookup is None:
            for i, contenttype in enumerate(self.contenttypes):
                contenttype.form_fill(
                    lookup.arrayptrs[pos + self.CONTENTS + i], layout.field(i), lookup
                )
        else:
            for i, contenttype in enumerate(self.contenttypes):
                contenttype.form_fill(
                    lookup.arrayptrs[pos + self.CONTENTS + i],
                    layout.field(self.recordlookup[i]),
                    lookup,
                )

    def fieldindex(self, key):
        out = -1
        if self.recordlookup is not None:
            for i, x in enumerate(self.recordlookup):
                if x == key:
                    out = i
                    break
        if out == -1:
            try:
                out = int(key)
            except ValueError:
                return None
            if not 0 <= out < len(self.contenttypes):
                return None
        return out

    def tolayout(self, lookup, pos, fields):
        if len(fields) > 0:
            index = self.fieldindex(fields[0])
            assert index is not None
            return self.contenttypes[index].tolayout(
                lookup, lookup.positions[pos + self.CONTENTS + index], fields[1:]
            )
        else:
            contents = []
            for i, contenttype in enumerate(self.contenttypes):
                layout = contenttype.tolayout(
                    lookup, lookup.positions[pos + self.CONTENTS + i], fields
                )
                contents.append(layout)

            if len(contents) == 0:
                return ak.layout.RecordArray(
                    contents,
                    self.recordlookup,
                    np.iinfo(np.int64).max,
                    parameters=self.parameters,
                )
            else:
                return ak.layout.RecordArray(
                    contents, self.recordlookup, parameters=self.parameters
                )

    def hasfield(self, key):
        return self.fieldindex(key) is not None

    def getitem_at_check(self, viewtype):
        out = self.getitem_at(viewtype)
        if isinstance(out, ak._connect._numba.arrayview.RecordViewType):
            typer = ak._util.numba_record_typer(
                out.arrayviewtype.type, out.arrayviewtype.behavior
            )
            if typer is not None:
                return typer(out)
        return out

    def getitem_at(self, viewtype):
        if len(viewtype.fields) == 0:
            return ak._connect._numba.arrayview.RecordViewType(viewtype)
        else:
            key = viewtype.fields[0]
            index = self.fieldindex(key)
            if index is None:
                if self.recordlookup is None:
                    raise ValueError(
                        "no field {0} in tuples with {1} fields".format(
                            repr(key), len(self.contenttypes)
                        )
                        + ak._util.exception_suffix(__file__)
                    )
                else:
                    raise ValueError(
                        "no field {0} in records with "
                        "fields: [{1}]".format(
                            repr(key), ", ".join(repr(x) for x in self.recordlookup)
                        )
                        + ak._util.exception_suffix(__file__)
                    )
            contenttype = self.contenttypes[index]
            subviewtype = ak._connect._numba.arrayview.wrap(
                contenttype, viewtype, viewtype.fields[1:]
            )
            return contenttype.getitem_at_check(subviewtype)

    def getitem_field(self, viewtype, key):
        index = self.fieldindex(key)
        if index is None:
            if self.recordlookup is None:
                raise ValueError(
                    "no field {0} in tuples with {1} fields".format(
                        repr(key), len(self.contenttypes)
                    )
                    + ak._util.exception_suffix(__file__)
                )
            else:
                raise ValueError(
                    "no field {0} in records with fields: [{1}]".format(
                        repr(key), ", ".join(repr(x) for x in self.recordlookup)
                    )
                    + ak._util.exception_suffix(__file__)
                )
        contenttype = self.contenttypes[index]
        subviewtype = ak._connect._numba.arrayview.wrap(contenttype, viewtype, None)
        return contenttype.getitem_range(subviewtype)

    def getitem_field_record(self, recordviewtype, key):
        index = self.fieldindex(key)
        if index is None:
            if self.recordlookup is None:
                raise ValueError(
                    "no field {0} in tuple with {1} fields".format(
                        repr(key), len(self.contenttypes)
                    )
                    + ak._util.exception_suffix(__file__)
                )
            else:
                raise ValueError(
                    "no field {0} in record with fields: [{1}]".format(
                        repr(key), ", ".join(repr(x) for x in self.recordlookup)
                    )
                    + ak._util.exception_suffix(__file__)
                )
        contenttype = self.contenttypes[index]
        subviewtype = ak._connect._numba.arrayview.wrap(
            contenttype, recordviewtype, None
        )
        return contenttype.getitem_at_check(subviewtype)

    def lower_getitem_at_check(
        self,
        context,
        builder,
        rettype,
        viewtype,
        viewval,
        viewproxy,
        attype,
        atval,
        wrapneg,
        checkbounds,
    ):
        out = self.lower_getitem_at(
            context,
            builder,
            rettype,
            viewtype,
            viewval,
            viewproxy,
            attype,
            atval,
            wrapneg,
            checkbounds,
        )
        baretype = self.getitem_at(viewtype)
        if isinstance(baretype, ak._connect._numba.arrayview.RecordViewType):
            lower = ak._util.numba_record_lower(
                baretype.arrayviewtype.type, baretype.arrayviewtype.behavior
            )
            if lower is not None:
                return lower(context, builder, rettype(baretype), (out,))
        return out

    def lower_getitem_at(
        self,
        context,
        builder,
        rettype,
        viewtype,
        viewval,
        viewproxy,
        attype,
        atval,
        wrapneg,
        checkbounds,
    ):
        atval = regularize_atval(
            context, builder, viewproxy, attype, atval, wrapneg, checkbounds
        )

        if len(viewtype.fields) == 0:
            proxyout = context.make_helper(
                builder, ak._connect._numba.arrayview.RecordViewType(viewtype)
            )
            proxyout.arrayview = viewval
            proxyout.at = atval
            return proxyout._getvalue()

        else:
            index = self.fieldindex(viewtype.fields[0])
            contenttype = self.contenttypes[index]

            whichpos = posat(context, builder, viewproxy.pos, self.CONTENTS + index)
            nextpos = getat(context, builder, viewproxy.arrayptrs, whichpos)

            nextviewtype = ak._connect._numba.arrayview.wrap(
                contenttype, viewtype, viewtype.fields[1:]
            )
            proxynext = context.make_helper(builder, nextviewtype)
            proxynext.pos = nextpos
            proxynext.start = viewproxy.start
            proxynext.stop = builder.add(
                atval, builder.add(viewproxy.start, context.get_constant(numba.intp, 1))
            )
            proxynext.arrayptrs = viewproxy.arrayptrs
            proxynext.sharedptrs = viewproxy.sharedptrs
            proxynext.pylookup = viewproxy.pylookup

            return contenttype.lower_getitem_at_check(
                context,
                builder,
                rettype,
                nextviewtype,
                proxynext._getvalue(),
                proxynext,
                numba.intp,
                atval,
                False,
                False,
            )

    def lower_getitem_field(self, context, builder, viewtype, viewval, key):
        viewproxy = context.make_helper(builder, viewtype, viewval)

        index = self.fieldindex(key)
        contenttype = self.contenttypes[index]

        whichpos = posat(context, builder, viewproxy.pos, self.CONTENTS + index)
        nextpos = getat(context, builder, viewproxy.arrayptrs, whichpos)

        proxynext = context.make_helper(builder, contenttype.getitem_range(viewtype))
        proxynext.pos = nextpos
        proxynext.start = viewproxy.start
        proxynext.stop = viewproxy.stop
        proxynext.arrayptrs = viewproxy.arrayptrs
        proxynext.sharedptrs = viewproxy.sharedptrs
        proxynext.pylookup = viewproxy.pylookup

        return proxynext._getvalue()

    def lower_getitem_field_record(
        self, context, builder, recordviewtype, recordviewval, key
    ):
        arrayviewtype = recordviewtype.arrayviewtype
        recordviewproxy = context.make_helper(builder, recordviewtype, recordviewval)
        arrayviewval = recordviewproxy.arrayview
        arrayviewproxy = context.make_helper(builder, arrayviewtype, arrayviewval)

        index = self.fieldindex(key)
        contenttype = self.contenttypes[index]

        whichpos = posat(context, builder, arrayviewproxy.pos, self.CONTENTS + index)
        nextpos = getat(context, builder, arrayviewproxy.arrayptrs, whichpos)

        proxynext = context.make_helper(
            builder, contenttype.getitem_range(arrayviewtype)
        )
        proxynext.pos = nextpos
        proxynext.start = arrayviewproxy.start
        proxynext.stop = builder.add(
            recordviewproxy.at,
            builder.add(arrayviewproxy.start, context.get_constant(numba.intp, 1)),
        )
        proxynext.arrayptrs = arrayviewproxy.arrayptrs
        proxynext.sharedptrs = arrayviewproxy.sharedptrs
        proxynext.pylookup = arrayviewproxy.pylookup

        nextviewtype = ak._connect._numba.arrayview.wrap(
            contenttype, arrayviewtype, None
        )

        rettype = self.getitem_field_record(recordviewtype, key)

        return contenttype.lower_getitem_at_check(
            context,
            builder,
            rettype,
            nextviewtype,
            proxynext._getvalue(),
            proxynext,
            numba.intp,
            recordviewproxy.at,
            False,
            False,
        )

    @property
    def is_tuple(self):
        return self.recordlookup is None

    @property
    def ndim(self):
        return 1

    @property
    def inner_dtype(self):
        return None

    @property
    def is_optiontype(self):
        return False

    @property
    def is_recordtype(self):
        return True


class UnionArrayType(ContentType):
    IDENTITIES = 0
    TAGS = 1
    INDEX = 2
    CONTENTS = 3

    @classmethod
    def tolookup(cls, layout, positions, sharedptrs, arrays):
        pos = len(positions)
        cls.tolookup_identities(layout, positions, sharedptrs, arrays)
        sharedptrs[-1] = layout._persistent_shared_ptr
        arrays.append(ak.nplike.of(layout.tags).asarray(layout.tags))
        positions.append(arrays[-1])
        sharedptrs.append(None)
        arrays.append(ak.nplike.of(layout.index).asarray(layout.index))
        positions.append(arrays[-1])
        sharedptrs.append(None)
        positions.extend([None] * layout.numcontents)
        sharedptrs.extend([None] * layout.numcontents)
        for i, content in enumerate(layout.contents):
            positions[pos + cls.CONTENTS + i] = ak._connect._numba.arrayview.tolookup(
                content, positions, sharedptrs, arrays
            )
        return pos

    @classmethod
    def form_tolookup(cls, form, positions, sharedptrs, arrays):
        pos = len(positions)
        cls.form_tolookup_identities(form, positions, sharedptrs, arrays)
        sharedptrs[-1] = 0
        arrays.append(0)
        positions.append(0)
        sharedptrs.append(None)
        arrays.append(0)
        positions.append(0)
        sharedptrs.append(None)
        positions.extend([None] * form.numcontents)
        sharedptrs.extend([None] * form.numcontents)
        for i, content in enumerate(form.contents):
            positions[pos + cls.CONTENTS + i] = ak._connect._numba.arrayview.tolookup(
                content, positions, sharedptrs, arrays
            )
        return pos

    @classmethod
    def from_form(cls, form):
        contents = []
        for x in form.contents:
            contents.append(ak._connect._numba.arrayview.tonumbatype(x))

        return UnionArrayType(
            cls.from_form_index(form.tags),
            cls.from_form_index(form.index),
            contents,
            cls.from_form_identities(form),
            form.parameters,
        )

    def __init__(self, tagstype, indextype, contenttypes, identitiestype, parameters):
        super(UnionArrayType, self).__init__(
            name="ak.UnionArrayType({0}, {1}, ({2}{3}), {4}, "
            "{5})".format(
                tagstype.name,
                indextype.name,
                ", ".join(x.name for x in contenttypes),
                "," if len(contenttypes) == 1 else "",
                identitiestype.name,
                json.dumps(parameters),
            )
        )
        self.tagstype = tagstype
        self.indextype = indextype
        self.contenttypes = contenttypes
        self.identitiestype = identitiestype
        self.parameters = parameters

    def form_fill(self, pos, layout, lookup):
        lookup.sharedptrs_hold[pos] = layout._persistent_shared_ptr
        lookup.sharedptrs[pos] = lookup.sharedptrs_hold[pos].ptr()
        self.form_fill_identities(pos, layout, lookup)

        tags = ak.nplike.of(layout.tags).asarray(layout.tags)
        lookup.original_positions[pos + self.TAGS] = tags
        lookup.arrayptrs[pos + self.TAGS] = tags.ctypes.data

        index = ak.nplike.of(layout.index).asarray(layout.index)
        lookup.original_positions[pos + self.INDEX] = index
        lookup.arrayptrs[pos + self.INDEX] = index.ctypes.data

        for i, contenttype in enumerate(self.contenttypes):
            contenttype.form_fill(
                lookup.arrayptrs[pos + self.CONTENTS + i], layout.content(i), lookup
            )

    def UnionArrayOf(self):
        if self.tagstype.dtype.bitwidth == 8 and self.tagstype.dtype.signed:
            if self.indextype.dtype.bitwidth == 32 and self.indextype.dtype.signed:
                return ak.layout.UnionArray8_32
            elif self.indextype.dtype.bitwidth == 32:
                return ak.layout.UnionArray8_U32
            elif self.indextype.dtype.bitwidth == 64 and self.indextype.dtype.signed:
                return ak.layout.UnionArray8_64
            else:
                raise AssertionError(
                    "no UnionArray* type for index array: {0}".format(self.indextype)
                    + ak._util.exception_suffix(__file__)
                )
        else:
            raise AssertionError(
                "no UnionArray* type for tags array: {0}".format(self.tagstype)
                + ak._util.exception_suffix(__file__)
            )

    def tolayout(self, lookup, pos, fields):
        tags = self.IndexOf(self.tagstype)(lookup.original_positions[pos + self.TAGS])
        index = self.IndexOf(self.indextype)(
            lookup.original_positions[pos + self.INDEX]
        )
        contents = []
        for i, contenttype in enumerate(self.contenttypes):
            layout = contenttype.tolayout(
                lookup, lookup.positions[pos + self.CONTENTS + i], fields
            )
            contents.append(layout)
        return self.UnionArrayOf()(tags, index, contents, parameters=self.parameters)

    def hasfield(self, key):
        return any(x.hasfield(key) for x in self.contenttypes)

    def getitem_at(self, viewtype):
        if not all(isinstance(x, RecordArrayType) for x in self.contenttypes):
            raise TypeError(
                "union types cannot be accessed in Numba"
                + ak._util.exception_suffix(__file__)
            )

    def getitem_range(self, viewtype):
        if not all(isinstance(x, RecordArrayType) for x in self.contenttypes):
            raise TypeError(
                "union types cannot be accessed in Numba"
                + ak._util.exception_suffix(__file__)
            )

    def getitem_field(self, viewtype, key):
        if not all(isinstance(x, RecordArrayType) for x in self.contenttypes):
            raise TypeError(
                "union types cannot be accessed in Numba"
                + ak._util.exception_suffix(__file__)
            )

    def lower_getitem_at(
        self,
        context,
        builder,
        rettype,
        viewtype,
        viewval,
        viewproxy,
        attype,
        atval,
        wrapneg,
        checkbounds,
    ):
        raise NotImplementedError(
            type(self).__name__
            + ".lower_getitem_at not implemented"
            + ak._util.exception_suffix(__file__)
        )

    def lower_getitem_range(
        self,
        context,
        builder,
        rettype,
        viewtype,
        viewval,
        viewproxy,
        start,
        stop,
        wrapneg,
    ):
        raise NotImplementedError(
            type(self).__name__
            + ".lower_getitem_range not implemented"
            + ak._util.exception_suffix(__file__)
        )

    def lower_getitem_field(self, context, builder, viewtype, viewval, viewproxy, key):
        raise NotImplementedError(
            type(self).__name__
            + ".lower_getitem_field not implemented"
            + ak._util.exception_suffix(__file__)
        )

    @property
    def ndim(self):
        out = None
        for contenttype in self.contenttypes:
            if out is None:
                out = contenttype.ndim
            elif out != contenttype.ndim:
                return None
        return out

    @property
    def inner_dtype(self):
        context = numba.core.typing.Context()
        return context.unify_types(*[x.inner_dtype for x in self.contenttypes])

    @property
    def is_optiontype(self):
        return any(x.is_optiontype for x in self.contents)

    @property
    def is_recordtype(self):
        if all(x.is_recordtype for x in self.contents):
            return True
        elif all(not x.is_recordtype for x in self.contents):
            return False
        else:
            return None


class VirtualArrayType(ContentType):
    IDENTITIES = 0
    PYOBJECT = 1
    ARRAY = 2

    @classmethod
    def tolookup(cls, layout, positions, sharedptrs, arrays):
        pos = len(positions)
        cls.tolookup_identities(layout, positions, sharedptrs, arrays)
        sharedptrs[-1] = layout._persistent_shared_ptr
        if layout.form is None:
            raise ValueError(
                "VirtualArrays without a known 'form' can't be used in Numba"
                + ak._util.exception_suffix(__file__)
            )
        pyptr = ctypes.py_object(layout)
        ctypes.pythonapi.Py_IncRef(pyptr)
        voidptr = numpy.frombuffer(pyptr, dtype=np.intp).item()

        positions.append(voidptr)
        sharedptrs.append(None)
        positions.append(None)
        sharedptrs.append(None)
        positions[pos + cls.ARRAY] = ak._connect._numba.arrayview.tolookup(
            layout.form.form, positions, sharedptrs, arrays
        )
        return pos

    @classmethod
    def form_tolookup(cls, form, positions, sharedptrs, arrays):
        pos = len(positions)
        cls.form_tolookup_identities(form, positions, sharedptrs, arrays)
        sharedptrs[-1] = 0
        if form.form is None:
            raise ValueError(
                "VirtualArrays without a known 'form' can't be used in Numba"
                + ak._util.exception_suffix(__file__)
            )
        positions.append(0)
        sharedptrs.append(None)
        positions.append(None)
        sharedptrs.append(None)
        positions[pos + cls.ARRAY] = ak._connect._numba.arrayview.tolookup(
            form.form, positions, sharedptrs, arrays
        )
        return pos

    @classmethod
    def from_form(cls, form):
        if form.form is None:
            raise ValueError(
                "VirtualArrays without a known 'form' can't be used in Numba "
                "(including nested)" + ak._util.exception_suffix(__file__)
            )
        return VirtualArrayType(
            form.form, cls.from_form_identities(form), form.parameters
        )

    def __init__(self, generator_form, identitiestype, parameters):
        if generator_form is None:
            raise ValueError(
                "VirtualArrays without a known 'form' can't be used in Numba"
                + ak._util.exception_suffix(__file__)
            )
        super(VirtualArrayType, self).__init__(
            name="ak.VirtualArrayType({0}, {1}, {2})".format(
                generator_form.tojson(), identitiestype.name, json.dumps(parameters)
            )
        )
        self.generator_form = generator_form
        self.identitiestype = identitiestype
        self.parameters = parameters

    def form_fill(self, pos, layout, lookup):
        lookup.sharedptrs_hold[pos] = layout._persistent_shared_ptr
        lookup.sharedptrs[pos] = lookup.sharedptrs_hold[pos].ptr()
        self.form_fill_identities(pos, layout, lookup)

        pyptr = ctypes.py_object(layout)
        ctypes.pythonapi.Py_IncRef(pyptr)
        voidptr = numpy.frombuffer(pyptr, dtype=np.intp).item()

        lookup.original_positions[pos + self.PYOBJECT] = voidptr
        lookup.arrayptrs[pos + self.PYOBJECT] = voidptr

    def tolayout(self, lookup, pos, fields):
        voidptr = ctypes.c_void_p(int(lookup.arrayptrs[pos + self.PYOBJECT]))
        pyptr = ctypes.cast(voidptr, ctypes.py_object)
        ctypes.pythonapi.Py_IncRef(pyptr)
        virtualarray = pyptr.value
        return virtualarray

    def hasfield(self, key):
        return self.generator_form.haskey(key)

    def getitem_at(self, viewtype):
        def getitem_at(form):
            if isinstance(form, ak.forms.NumpyForm):
                assert len(form.inner_shape) == 0
                if form.primitive == "float64":
                    return numba.float64
                elif form.primitive == "float32":
                    return numba.float32
                elif form.primitive == "int64":
                    return numba.int64
                elif form.primitive == "uint64":
                    return numba.uint64
                elif form.primitive == "int32":
                    return numba.int32
                elif form.primitive == "uint32":
                    return numba.uint32
                elif form.primitive == "int16":
                    return numba.int16
                elif form.primitive == "uint16":
                    return numba.uint16
                elif form.primitive == "int8":
                    return numba.int8
                elif form.primitive == "uint8":
                    return numba.uint8
                elif form.primitive == "bool":
                    return numba.boolean
                else:
                    raise ValueError(
                        "unrecognized NumpyForm.primitive type: {0}".format(
                            form.primitive
                        )
                        + ak._util.exception_suffix(__file__)
                    )

            elif isinstance(
                form,
                (
                    ak.forms.RegularForm,
                    ak.forms.ListForm,
                    ak.forms.ListOffsetForm,
                ),
            ):
                return form.content

            elif isinstance(form, ak.forms.IndexedForm):
                return getitem_at(form.content)

            elif isinstance(
                form,
                (
                    ak.forms.IndexedOptionForm,
                    ak.forms.ByteMaskedForm,
                    ak.forms.BitMaskedForm,
                    ak.forms.UnmaskedForm,
                ),
            ):
                return numba.types.optional(wrap(getitem_at(form.content)))

            elif isinstance(form, ak.forms.RecordForm):
                arrayview = wrap(form)
                return arrayview.type.getitem_at(arrayview)

            elif isinstance(form, ak.forms.UnionForm):
                raise TypeError(
                    "union types cannot be accessed in Numba"
                    + ak._util.exception_suffix(__file__)
                )

            elif isinstance(form, ak.forms.VirtualForm):
                return getitem_at(form.form)

            else:
                raise AssertionError(
                    "unrecognized Form type: {0}".format(type(form))
                    + ak._util.exception_suffix(__file__)
                )

        def wrap(out):
            if isinstance(out, ak.forms.Form):
                numbatype = ak._connect._numba.arrayview.tonumbatype(out)
                return ak._connect._numba.arrayview.wrap(numbatype, viewtype, None)
            else:
                return out

        return wrap(getitem_at(self.generator_form))

    def lower_getitem_at(
        self,
        context,
        builder,
        rettype,
        viewtype,
        viewval,
        viewproxy,
        attype,
        atval,
        wrapneg,
        checkbounds,
    ):
        pyobjptr = getat(
            context,
            builder,
            viewproxy.arrayptrs,
            posat(context, builder, viewproxy.pos, self.PYOBJECT),
        )
        arraypos = getat(
            context,
            builder,
            viewproxy.arrayptrs,
            posat(context, builder, viewproxy.pos, self.ARRAY),
        )
        sharedptr = getat(context, builder, viewproxy.sharedptrs, arraypos)

        numbatype = ak._connect._numba.arrayview.tonumbatype(self.generator_form)

        with builder.if_then(
            builder.icmp_signed("==", sharedptr, context.get_constant(numba.intp, 0)),
            likely=False,
        ):
            # only rarely enter Python
            pyapi = context.get_python_api(builder)
            gil = pyapi.gil_ensure()

            # borrowed references
            virtualarray_obj = builder.inttoptr(
                pyobjptr, context.get_value_type(numba.types.pyobject)
            )
            lookup_obj = viewproxy.pylookup

            # new references
            numbatype_obj = pyapi.unserialize(pyapi.serialize_object(numbatype))
            fill_obj = pyapi.object_getattr_string(numbatype_obj, "form_fill")
            arraypos_obj = pyapi.long_from_ssize_t(arraypos)
            array_obj = pyapi.object_getattr_string(virtualarray_obj, "array")

            # FIXME: memory leak? what about putting this exception after the decrefs?
            with builder.if_then(
                builder.icmp_signed(
                    "!=",
                    pyapi.err_occurred(),
                    context.get_constant(numba.types.voidptr, 0),
                ),
                likely=False,
            ):
                context.call_conv.return_exc(builder)

            # add the materialized array to our Lookup
            pyapi.call_function_objargs(
                fill_obj,
                (
                    arraypos_obj,
                    array_obj,
                    lookup_obj,
                ),
            )

            # FIXME: memory leak? what about putting this exception after the decrefs?
            with builder.if_then(
                builder.icmp_signed(
                    "!=",
                    pyapi.err_occurred(),
                    context.get_constant(numba.types.voidptr, 0),
                ),
                likely=False,
            ):
                context.call_conv.return_exc(builder)

            # decref the new references
            pyapi.decref(array_obj)
            pyapi.decref(arraypos_obj)
            pyapi.decref(fill_obj)
            pyapi.decref(numbatype_obj)

            pyapi.gil_release(gil)

        # normally, we just pass on the request to the materialized array
        whichpos = posat(context, builder, viewproxy.pos, self.ARRAY)
        nextpos = getat(context, builder, viewproxy.arrayptrs, whichpos)

        nextviewtype = ak._connect._numba.arrayview.wrap(numbatype, viewtype, None)
        proxynext = context.make_helper(builder, nextviewtype)
        proxynext.pos = nextpos
        proxynext.start = viewproxy.start
        proxynext.stop = viewproxy.stop
        proxynext.arrayptrs = viewproxy.arrayptrs
        proxynext.sharedptrs = viewproxy.sharedptrs
        proxynext.pylookup = viewproxy.pylookup

        return numbatype.lower_getitem_at_check(
            context,
            builder,
            rettype,
            nextviewtype,
            proxynext._getvalue(),
            proxynext,
            numba.intp,
            atval,
            wrapneg,
            checkbounds,
        )

    @property
    def ndim(self):
        return self.generator_form.purelist_depth

    @property
    def inner_dtype(self):
        return inner_dtype_of_form(self.generator_form)

    @property
    def is_optiontype(self):
        return optiontype_of_form(self.generator_form)

    @property
    def is_recordtype(self):
        return recordtype_of_form(self.generator_form)


def inner_dtype_of_form(form):
    if form is None:
        return None

    elif isinstance(form, (ak.forms.NumpyForm,)):
        return numba.from_dtype(form.to_numpy())

    elif isinstance(form, (ak.forms.EmptyForm,)):
        return numba.types.float64

    elif isinstance(
        form,
        (
            ak.forms.RegularForm,
            ak.forms.ListForm,
            ak.forms.ListOffsetForm,
            ak.forms.IndexedForm,
        ),
    ):
        return inner_dtype_of_form(form.content)

    elif isinstance(
        form,
        (
            ak.forms.RecordForm,
            ak.forms.IndexedOptionForm,
            ak.forms.ByteMaskedForm,
            ak.forms.BitMaskedForm,
            ak.forms.UnmaskedForm,
        ),
    ):
        return None

    elif isinstance(form, ak.forms.UnionForm):
        context = numba.core.typing.Context()
        return context.unify_types(*[inner_dtype_of_form(x) for x in form.contents])

    elif isinstance(form, ak.forms.VirtualForm):
        return inner_dtype_of_form(form.form)

    else:
        raise AssertionError(
            "unrecognized Form type: {0}".format(type(form))
            + ak._util.exception_suffix(__file__)
        )


def optiontype_of_form(form):
    if form is None:
        return None

    elif isinstance(
        form,
        (
            ak.forms.NumpyForm,
            ak.forms.EmptyForm,
            ak.forms.RegularForm,
            ak.forms.ListForm,
            ak.forms.ListOffsetForm,
            ak.forms.IndexedForm,
            ak.forms.RecordForm,
        ),
    ):
        return False

    elif isinstance(
        form,
        (
            ak.forms.IndexedOptionForm,
            ak.forms.ByteMaskedForm,
            ak.forms.BitMaskedForm,
            ak.forms.UnmaskedForm,
        ),
    ):
        return False

    elif isinstance(form, ak.forms.UnionForm):
        return any(optiontype_of_form(x) for x in form.contents)

    elif isinstance(form, ak.forms.VirtualForm):
        return optiontype_of_form(form.form)

    else:
        raise AssertionError(
            "unrecognized Form type: {0}".format(type(form))
            + ak._util.exception_suffix(__file__)
        )


def recordtype_of_form(form):
    if form is None:
        return None

    elif isinstance(
        form,
        (
            ak.forms.NumpyForm,
            ak.forms.EmptyForm,
            ak.forms.RegularForm,
            ak.forms.ListForm,
            ak.forms.ListOffsetForm,
        ),
    ):
        return False

    elif isinstance(
        form,
        (
            ak.forms.IndexedForm,
            ak.forms.IndexedOptionForm,
            ak.forms.ByteMaskedForm,
            ak.forms.BitMaskedForm,
            ak.forms.UnmaskedForm,
        ),
    ):
        return recordtype_of_form(form.content)

    elif isinstance(form, (ak.forms.RecordForm,)):
        return True

    elif isinstance(form, ak.forms.UnionForm):
        return any(recordtype_of_form(x) for x in form.contents)

    elif isinstance(form, ak.forms.VirtualForm):
        return recordtype_of_form(form.form)

    else:
        raise AssertionError(
            "unrecognized Form type: {0}".format(type(form))
            + ak._util.exception_suffix(__file__)
        )
