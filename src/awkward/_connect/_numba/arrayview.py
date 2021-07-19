# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import operator

import numba
import numba.core.typing
import numba.core.typing.ctypes_utils

import awkward as ak

np = ak.nplike.NumPyMetadata.instance()

########## for code that's built up from strings


def code_to_function(code, function_name, externals=None, debug=False):
    if debug:
        print("################### " + function_name)  # noqa: T001
        print(code)  # noqa: T001
    namespace = {} if externals is None else dict(externals)
    exec(code, namespace)
    return namespace[function_name]


########## Lookup


class Lookup(object):
    def __init__(self, layout):
        positions = []
        sharedptrs = []
        arrays = []
        tolookup(layout, positions, sharedptrs, arrays)
        assert len(positions) == len(sharedptrs)

        def find(x):
            for i, array in enumerate(arrays):
                if x is array:
                    return i
            else:
                assert isinstance(x, int)
                return x

        self.original_positions = positions
        self.positions = [find(x) for x in positions]
        self.sharedptrs_hold = sharedptrs
        self.arrays = arrays

        def arrayptr(x):
            if isinstance(x, int):
                return x
            else:
                return x.ctypes.data

        def sharedptr(x):
            if x is None:
                return -1
            elif x == 0:
                return 0
            else:
                return x.ptr()

        self.nplike = ak.nplike.of(layout)

        self.arrayptrs = self.nplike.array(
            [arrayptr(x) for x in positions], dtype=np.intp
        )
        self.sharedptrs = self.nplike.array(
            [sharedptr(x) for x in sharedptrs], dtype=np.intp
        )

    def _view_as_array(self):
        return self.nplike.vstack(
            [self.nplike.arange(len(self.arrayptrs)), self.arrayptrs, self.sharedptrs]
        ).T


def tolookup(layout, positions, sharedptrs, arrays):
    if isinstance(layout, ak.layout.NumpyArray):
        return ak._connect._numba.layout.NumpyArrayType.tolookup(
            layout, positions, sharedptrs, arrays
        )

    elif isinstance(layout, ak.forms.NumpyForm):
        return ak._connect._numba.layout.NumpyArrayType.form_tolookup(
            layout, positions, sharedptrs, arrays
        )

    elif isinstance(layout, ak.layout.RegularArray):
        return ak._connect._numba.layout.RegularArrayType.tolookup(
            layout, positions, sharedptrs, arrays
        )

    elif isinstance(layout, ak.forms.RegularForm):
        return ak._connect._numba.layout.RegularArrayType.form_tolookup(
            layout, positions, sharedptrs, arrays
        )

    elif isinstance(
        layout,
        (
            ak.layout.ListArray32,
            ak.layout.ListArrayU32,
            ak.layout.ListArray64,
            ak.layout.ListOffsetArray32,
            ak.layout.ListOffsetArrayU32,
            ak.layout.ListOffsetArray64,
        ),
    ):
        return ak._connect._numba.layout.ListArrayType.tolookup(
            layout, positions, sharedptrs, arrays
        )

    elif isinstance(layout, (ak.forms.ListForm, ak.forms.ListOffsetForm)):
        return ak._connect._numba.layout.ListArrayType.form_tolookup(
            layout, positions, sharedptrs, arrays
        )

    elif isinstance(
        layout,
        (
            ak.layout.IndexedArray32,
            ak.layout.IndexedArrayU32,
            ak.layout.IndexedArray64,
        ),
    ):
        return ak._connect._numba.layout.IndexedArrayType.tolookup(
            layout, positions, sharedptrs, arrays
        )

    elif isinstance(layout, ak.forms.IndexedForm):
        return ak._connect._numba.layout.IndexedArrayType.form_tolookup(
            layout, positions, sharedptrs, arrays
        )

    elif isinstance(
        layout,
        (ak.layout.IndexedOptionArray32, ak.layout.IndexedOptionArray64),
    ):
        return ak._connect._numba.layout.IndexedOptionArrayType.tolookup(
            layout, positions, sharedptrs, arrays
        )

    elif isinstance(layout, ak.forms.IndexedOptionForm):
        return ak._connect._numba.layout.IndexedOptionArrayType.form_tolookup(
            layout, positions, sharedptrs, arrays
        )

    elif isinstance(layout, ak.layout.ByteMaskedArray):
        return ak._connect._numba.layout.ByteMaskedArrayType.tolookup(
            layout, positions, sharedptrs, arrays
        )

    elif isinstance(layout, ak.forms.ByteMaskedForm):
        return ak._connect._numba.layout.ByteMaskedArrayType.form_tolookup(
            layout, positions, sharedptrs, arrays
        )

    elif isinstance(layout, ak.layout.BitMaskedArray):
        return ak._connect._numba.layout.BitMaskedArrayType.tolookup(
            layout, positions, sharedptrs, arrays
        )

    elif isinstance(layout, ak.forms.BitMaskedForm):
        return ak._connect._numba.layout.BitMaskedArrayType.form_tolookup(
            layout, positions, sharedptrs, arrays
        )

    elif isinstance(layout, ak.layout.UnmaskedArray):
        return ak._connect._numba.layout.UnmaskedArrayType.tolookup(
            layout, positions, sharedptrs, arrays
        )

    elif isinstance(layout, ak.forms.UnmaskedForm):
        return ak._connect._numba.layout.UnmaskedArrayType.form_tolookup(
            layout, positions, sharedptrs, arrays
        )

    elif isinstance(layout, ak.layout.RecordArray):
        return ak._connect._numba.layout.RecordArrayType.tolookup(
            layout, positions, sharedptrs, arrays
        )

    elif isinstance(layout, ak.forms.RecordForm):
        return ak._connect._numba.layout.RecordArrayType.form_tolookup(
            layout, positions, sharedptrs, arrays
        )

    elif isinstance(layout, ak.layout.Record):
        return ak._connect._numba.layout.RecordType.tolookup(
            layout, positions, sharedptrs, arrays
        )

    elif isinstance(
        layout,
        (
            ak.layout.UnionArray8_32,
            ak.layout.UnionArray8_U32,
            ak.layout.UnionArray8_64,
        ),
    ):
        return ak._connect._numba.layout.UnionArrayType.tolookup(
            layout, positions, sharedptrs, arrays
        )

    elif isinstance(layout, ak.forms.UnionForm):
        return ak._connect._numba.layout.UnionArrayType.form_tolookup(
            layout, positions, sharedptrs, arrays
        )

    elif isinstance(layout, ak.layout.VirtualArray):
        return ak._connect._numba.layout.VirtualArrayType.tolookup(
            layout, positions, sharedptrs, arrays
        )

    elif isinstance(layout, ak.forms.VirtualForm):
        return ak._connect._numba.layout.VirtualArrayType.form_tolookup(
            layout, positions, sharedptrs, arrays
        )

    else:
        raise AssertionError(
            "unrecognized Content or Form type: {0}".format(type(layout))
            + ak._util.exception_suffix(__file__)
        )


def tonumbatype(form):
    if isinstance(form, ak.forms.NumpyForm):
        return ak._connect._numba.layout.NumpyArrayType.from_form(form)

    elif isinstance(form, ak.forms.RegularForm):
        return ak._connect._numba.layout.RegularArrayType.from_form(form)

    elif isinstance(form, (ak.forms.ListForm, ak.forms.ListOffsetForm)):
        return ak._connect._numba.layout.ListArrayType.from_form(form)

    elif isinstance(form, ak.forms.IndexedForm):
        return ak._connect._numba.layout.IndexedArrayType.from_form(form)

    elif isinstance(form, ak.forms.IndexedOptionForm):
        return ak._connect._numba.layout.IndexedOptionArrayType.from_form(form)

    elif isinstance(form, ak.forms.ByteMaskedForm):
        return ak._connect._numba.layout.ByteMaskedArrayType.from_form(form)

    elif isinstance(form, ak.forms.BitMaskedForm):
        return ak._connect._numba.layout.BitMaskedArrayType.from_form(form)

    elif isinstance(form, ak.forms.UnmaskedForm):
        return ak._connect._numba.layout.UnmaskedArrayType.from_form(form)

    elif isinstance(form, ak.forms.RecordForm):
        return ak._connect._numba.layout.RecordArrayType.from_form(form)

    elif isinstance(form, ak.forms.UnionForm):
        return ak._connect._numba.layout.UnionArrayType.from_form(form)

    elif isinstance(form, ak.forms.VirtualForm):
        return ak._connect._numba.layout.VirtualArrayType.from_form(form)

    else:
        raise AssertionError(
            "unrecognized Form type: {0}".format(type(form))
            + ak._util.exception_suffix(__file__)
        )


@numba.extending.typeof_impl.register(Lookup)
def typeof_Lookup(obj, c):
    return LookupType()


class LookupType(numba.types.Type):
    arraytype = numba.types.Array(numba.intp, 1, "C")

    def __init__(self):
        super(LookupType, self).__init__(name="ak.LookupType()")


@numba.extending.register_model(LookupType)
class LookupModel(numba.core.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("arrayptrs", fe_type.arraytype), ("sharedptrs", fe_type.arraytype)]
        super(LookupModel, self).__init__(dmm, fe_type, members)


@numba.extending.unbox(LookupType)
def unbox_Lookup(lookuptype, lookupobj, c):
    arrayptrs_obj = c.pyapi.object_getattr_string(lookupobj, "arrayptrs")
    sharedptrs_obj = c.pyapi.object_getattr_string(lookupobj, "sharedptrs")

    proxyout = c.context.make_helper(c.builder, lookuptype)
    proxyout.arrayptrs = c.pyapi.to_native_value(
        lookuptype.arraytype, arrayptrs_obj
    ).value
    proxyout.sharedptrs = c.pyapi.to_native_value(
        lookuptype.arraytype, sharedptrs_obj
    ).value

    c.pyapi.decref(arrayptrs_obj)
    c.pyapi.decref(sharedptrs_obj)

    is_error = numba.core.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(proxyout._getvalue(), is_error)


########## ArrayView


class ArrayView(object):
    @classmethod
    def fromarray(cls, array):
        behavior = ak._util.behaviorof(array)
        layout = ak.operations.convert.to_layout(
            array,
            allow_record=False,
            allow_other=False,
            numpytype=(np.number, bool, np.bool_),
        )
        while isinstance(layout, ak.layout.VirtualArray) and isinstance(
            layout.generator, ak.layout.SliceGenerator
        ):
            layout = layout.array
        layout = ak.operations.convert.regularize_numpyarray(
            layout, allow_empty=False, highlevel=False
        )

        if isinstance(layout, ak.partition.PartitionedArray):
            numba_type = None
            for part in layout.partitions:
                if numba_type is None:
                    numba_type = ak._connect._numba.layout.typeof(part)
                elif numba_type != ak._connect._numba.layout.typeof(part):
                    raise ValueError(
                        "partitioned arrays can only be used in Numba if all "
                        "partitions have the same numba_type"
                        + ak._util.exception_suffix(__file__)
                    )
            return PartitionedView(
                ak._connect._numba.layout.typeof(part),
                behavior,
                [Lookup(x) for x in layout.partitions],
                ak.nplike.of(layout).asarray(layout.stops, dtype=np.intp),
                0,
                len(layout),
                (),
            )

        else:
            return ArrayView(
                ak._connect._numba.layout.typeof(layout),
                behavior,
                Lookup(layout),
                0,
                0,
                len(layout),
                (),
            )

    def __init__(self, type, behavior, lookup, pos, start, stop, fields):
        self.type = type
        self.behavior = behavior
        self.lookup = lookup
        self.pos = pos
        self.start = start
        self.stop = stop
        self.fields = fields

    def toarray(self):
        layout = self.type.tolayout(self.lookup, self.pos, self.fields)
        sliced = layout.getitem_range_nowrap(self.start, self.stop)
        return ak._util.wrap(sliced, self.behavior)


@numba.extending.typeof_impl.register(ArrayView)
def typeof_ArrayView(obj, c):
    return ArrayViewType(obj.type, obj.behavior, obj.fields)


def wrap(type, viewtype, fields):
    if fields is None:
        return ArrayViewType(type, viewtype.behavior, viewtype.fields)
    else:
        return ArrayViewType(type, viewtype.behavior, fields)


class ArrayViewType(numba.types.IterableType, numba.types.Sized):
    def __init__(self, type, behavior, fields):
        super(ArrayViewType, self).__init__(
            name="ak.ArrayView({0}, {1}, {2})".format(
                type.name,
                ak._connect._numba.repr_behavior(behavior),
                repr(fields),
            )
        )
        self.type = type
        self.behavior = behavior
        self.fields = fields

    @property
    def iterator_type(self):
        return IteratorType(self)


@numba.extending.register_model(ArrayViewType)
class ArrayViewModel(numba.core.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("pos", numba.intp),
            ("start", numba.intp),
            ("stop", numba.intp),
            ("arrayptrs", numba.types.CPointer(numba.intp)),
            ("sharedptrs", numba.types.CPointer(numba.intp)),
            ("pylookup", numba.types.pyobject),
        ]
        super(ArrayViewModel, self).__init__(dmm, fe_type, members)


@numba.core.imputils.lower_constant(ArrayViewType)
def lower_const_Array(context, builder, viewtype, array):
    return lower_const_view(context, builder, viewtype, array._numbaview)


def lower_const_view(context, builder, viewtype, view):
    lookup = view.lookup
    arrayptrs = lookup.arrayptrs
    sharedptrs = lookup.sharedptrs
    pos = view.pos
    start = view.start
    stop = view.stop

    arrayptrs_val = context.make_constant_array(
        builder, numba.typeof(arrayptrs), arrayptrs
    )
    sharedptrs_val = context.make_constant_array(
        builder, numba.typeof(sharedptrs), sharedptrs
    )

    proxyout = context.make_helper(builder, viewtype)
    proxyout.pos = context.get_constant(numba.intp, pos)
    proxyout.start = context.get_constant(numba.intp, start)
    proxyout.stop = context.get_constant(numba.intp, stop)
    proxyout.arrayptrs = context.make_helper(
        builder, numba.typeof(arrayptrs), arrayptrs_val
    ).data
    proxyout.sharedptrs = context.make_helper(
        builder, numba.typeof(sharedptrs), sharedptrs_val
    ).data
    proxyout.pylookup = context.add_dynamic_addr(
        builder, id(lookup), info=str(type(lookup))
    )

    return proxyout._getvalue()


@numba.extending.unbox(ArrayViewType)
def unbox_Array(viewtype, arrayobj, c):
    view_obj = c.pyapi.object_getattr_string(arrayobj, "_numbaview")
    out = unbox_ArrayView(viewtype, view_obj, c)
    c.pyapi.decref(view_obj)
    return out


def unbox_ArrayView(viewtype, view_obj, c):
    lookup_obj = c.pyapi.object_getattr_string(view_obj, "lookup")
    pos_obj = c.pyapi.object_getattr_string(view_obj, "pos")
    start_obj = c.pyapi.object_getattr_string(view_obj, "start")
    stop_obj = c.pyapi.object_getattr_string(view_obj, "stop")

    lookup_val = c.pyapi.to_native_value(LookupType(), lookup_obj).value
    lookup_proxy = c.context.make_helper(c.builder, LookupType(), lookup_val)

    proxyout = c.context.make_helper(c.builder, viewtype)
    proxyout.pos = c.pyapi.number_as_ssize_t(pos_obj)
    proxyout.start = c.pyapi.number_as_ssize_t(start_obj)
    proxyout.stop = c.pyapi.number_as_ssize_t(stop_obj)
    proxyout.arrayptrs = c.context.make_helper(
        c.builder, LookupType.arraytype, lookup_proxy.arrayptrs
    ).data
    proxyout.sharedptrs = c.context.make_helper(
        c.builder, LookupType.arraytype, lookup_proxy.sharedptrs
    ).data
    proxyout.pylookup = lookup_obj

    c.pyapi.decref(lookup_obj)
    c.pyapi.decref(pos_obj)
    c.pyapi.decref(start_obj)
    c.pyapi.decref(stop_obj)

    if c.context.enable_nrt:
        c.context.nrt.decref(c.builder, LookupType(), lookup_val)

    is_error = numba.core.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(proxyout._getvalue(), is_error)


@numba.extending.box(ArrayViewType)
def box_Array(viewtype, viewval, c):
    arrayview_obj = box_ArrayView(viewtype, viewval, c)
    out = c.pyapi.call_method(arrayview_obj, "toarray", ())
    c.pyapi.decref(arrayview_obj)
    return out


def dict2serializable(obj):
    if obj is None:
        return None
    else:
        return tuple(obj.items())


def serializable2dict(obj):
    if obj is None:
        return None
    else:
        return dict(obj)


def box_ArrayView(viewtype, viewval, c):
    serializable2dict_obj = c.pyapi.unserialize(
        c.pyapi.serialize_object(serializable2dict)
    )
    behavior2_obj = c.pyapi.unserialize(
        c.pyapi.serialize_object(dict2serializable(viewtype.behavior))
    )
    behavior_obj = c.pyapi.call_function_objargs(
        serializable2dict_obj, (behavior2_obj,)
    )
    ArrayView_obj = c.pyapi.unserialize(c.pyapi.serialize_object(ArrayView))
    type_obj = c.pyapi.unserialize(c.pyapi.serialize_object(viewtype.type))
    fields_obj = c.pyapi.unserialize(c.pyapi.serialize_object(viewtype.fields))

    proxyin = c.context.make_helper(c.builder, viewtype, viewval)
    pos_obj = c.pyapi.long_from_ssize_t(proxyin.pos)
    start_obj = c.pyapi.long_from_ssize_t(proxyin.start)
    stop_obj = c.pyapi.long_from_ssize_t(proxyin.stop)
    lookup_obj = proxyin.pylookup

    out = c.pyapi.call_function_objargs(
        ArrayView_obj,
        (type_obj, behavior_obj, lookup_obj, pos_obj, start_obj, stop_obj, fields_obj),
    )

    c.pyapi.decref(serializable2dict_obj)
    c.pyapi.decref(behavior2_obj)
    c.pyapi.decref(behavior_obj)
    c.pyapi.decref(ArrayView_obj)
    c.pyapi.decref(type_obj)
    c.pyapi.decref(fields_obj)
    c.pyapi.decref(pos_obj)
    c.pyapi.decref(start_obj)
    c.pyapi.decref(stop_obj)

    return out


@numba.core.typing.templates.infer_global(len)
class type_len(numba.core.typing.templates.AbstractTemplate):
    def generic(self, args, kwargs):
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], ArrayViewType):
            return numba.intp(args[0])


@numba.extending.lower_builtin(len, ArrayViewType)
def lower_len(context, builder, sig, args):
    proxyin = context.make_helper(builder, sig.args[0], args[0])
    return builder.sub(proxyin.stop, proxyin.start)


@numba.core.typing.templates.infer_global(operator.getitem)
class type_getitem(numba.core.typing.templates.AbstractTemplate):
    def generic(self, args, kwargs):
        if len(args) == 2 and len(kwargs) == 0 and isinstance(args[0], ArrayViewType):
            viewtype, wheretype = args
            if isinstance(wheretype, numba.types.Integer):
                return viewtype.type.getitem_at_check(viewtype)(viewtype, wheretype)
            elif (
                isinstance(wheretype, numba.types.SliceType) and not wheretype.has_step
            ):
                return viewtype.type.getitem_range(viewtype)(viewtype, wheretype)
            elif isinstance(wheretype, numba.types.StringLiteral):
                return viewtype.type.getitem_field(viewtype, wheretype.literal_value)(
                    viewtype, wheretype
                )
            else:
                raise TypeError(
                    "only an integer, start:stop range, or a *constant* "
                    "field name string may be used as ak.Array "
                    "slices in compiled code" + ak._util.exception_suffix(__file__)
                )


@numba.extending.lower_builtin(operator.getitem, ArrayViewType, numba.types.Integer)
def lower_getitem_at(context, builder, sig, args):
    rettype, (viewtype, wheretype) = sig.return_type, sig.args
    viewval, whereval = args
    viewproxy = context.make_helper(builder, viewtype, viewval)
    return viewtype.type.lower_getitem_at_check(
        context,
        builder,
        rettype,
        viewtype,
        viewval,
        viewproxy,
        wheretype,
        whereval,
        True,
        True,
    )


@numba.extending.lower_builtin(operator.getitem, ArrayViewType, numba.types.slice2_type)
def lower_getitem_range(context, builder, sig, args):
    rettype, (viewtype, wheretype) = sig.return_type, sig.args
    viewval, whereval = args
    viewproxy = context.make_helper(builder, viewtype, viewval)
    whereproxy = context.make_helper(builder, wheretype, whereval)
    return viewtype.type.lower_getitem_range(
        context,
        builder,
        rettype,
        viewtype,
        viewval,
        viewproxy,
        whereproxy.start,
        whereproxy.stop,
        True,
    )


@numba.extending.lower_builtin(
    operator.getitem, ArrayViewType, numba.types.StringLiteral
)
def lower_getitem_field(context, builder, sig, args):
    _, (viewtype, wheretype) = sig.return_type, sig.args
    viewval, whereval = args
    return viewtype.type.lower_getitem_field(
        context, builder, viewtype, viewval, wheretype.literal_value
    )


@numba.core.typing.templates.infer_getattr
class type_getattr(numba.core.typing.templates.AttributeTemplate):
    key = ArrayViewType

    def generic_resolve(self, viewtype, attr):
        if attr == "ndim":
            return numba.intp
        else:
            return viewtype.type.getitem_field(viewtype, attr)


@numba.extending.lower_getattr_generic(ArrayViewType)
def lower_getattr_generic(context, builder, viewtype, viewval, attr):
    if attr == "ndim":
        return context.get_constant(numba.intp, viewtype.type.ndim)
    else:
        return viewtype.type.lower_getitem_field(
            context, builder, viewtype, viewval, attr
        )


class IteratorType(numba.types.common.SimpleIteratorType):
    def __init__(self, viewtype):
        super(IteratorType, self).__init__(
            "ak.Iterator({0})".format(viewtype.name),
            viewtype.type.getitem_at_check(viewtype),
        )
        self.viewtype = viewtype


@numba.core.typing.templates.infer
class type_getiter(numba.core.typing.templates.AbstractTemplate):
    key = "getiter"

    def generic(self, args, kwargs):
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], ArrayViewType):
            return IteratorType(args[0])(args[0])


@numba.core.datamodel.registry.register_default(IteratorType)
class IteratorModel(numba.core.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("view", fe_type.viewtype),
            ("length", numba.intp),
            ("at", numba.types.EphemeralPointer(numba.intp)),
        ]
        super(IteratorModel, self).__init__(dmm, fe_type, members)


@numba.extending.lower_builtin("getiter", ArrayViewType)
def lower_getiter(context, builder, sig, args):
    rettype, (viewtype,) = sig.return_type, sig.args
    (viewval,) = args
    viewproxy = context.make_helper(builder, viewtype, viewval)
    proxyout = context.make_helper(builder, rettype)
    proxyout.view = viewval
    proxyout.length = builder.sub(viewproxy.stop, viewproxy.start)
    proxyout.at = numba.core.cgutils.alloca_once_value(
        builder, context.get_constant(numba.intp, 0)
    )
    if context.enable_nrt:
        context.nrt.incref(builder, viewtype, viewval)
    return numba.core.imputils.impl_ret_new_ref(
        context, builder, rettype, proxyout._getvalue()
    )


@numba.extending.lower_builtin("iternext", IteratorType)
@numba.core.imputils.iternext_impl(numba.core.imputils.RefType.BORROWED)
def lower_iternext(context, builder, sig, args, result):
    (itertype,) = sig.args
    (iterval,) = args
    proxyin = context.make_helper(builder, itertype, iterval)
    at = builder.load(proxyin.at)

    is_valid = builder.icmp_signed("<", at, proxyin.length)
    result.set_valid(is_valid)

    with builder.if_then(is_valid, likely=True):
        result.yield_(
            lower_getitem_at(
                context,
                builder,
                itertype.yield_type(itertype.viewtype, numba.intp),
                (proxyin.view, at),
            )
        )
        nextat = numba.core.cgutils.increment_index(builder, at)
        builder.store(nextat, proxyin.at)


########## RecordView


class RecordView(object):
    @classmethod
    def fromrecord(cls, record):
        behavior = ak._util.behaviorof(record)
        layout = ak.operations.convert.to_layout(
            record,
            allow_record=True,
            allow_other=False,
            numpytype=(np.number, bool, np.bool_),
        )
        assert isinstance(layout, ak.layout.Record)
        arraylayout = layout.array
        return RecordView(
            ArrayView(
                ak._connect._numba.layout.typeof(arraylayout),
                behavior,
                Lookup(arraylayout),
                0,
                0,
                len(arraylayout),
                (),
            ),
            layout.at,
        )

    def __init__(self, arrayview, at):
        self.arrayview = arrayview
        self.at = at

    def torecord(self):
        arraylayout = self.arrayview.toarray().layout
        return ak._util.wrap(
            ak.layout.Record(arraylayout, self.at), self.arrayview.behavior
        )


@numba.extending.typeof_impl.register(RecordView)
def typeof_RecordView(obj, c):
    return RecordViewType(numba.typeof(obj.arrayview))


class RecordViewType(numba.types.Type):
    def __init__(self, arrayviewtype):
        super(RecordViewType, self).__init__(
            name="ak.RecordViewType({0})".format(arrayviewtype.name)
        )
        self.arrayviewtype = arrayviewtype

    @property
    def behavior(self):
        return self.arrayviewtype.behavior

    @property
    def fields(self):
        return self.arrayviewtype.fields

    def typer_field(self, key):
        return self.arrayviewtype.type.getitem_field_record(self, key)

    def lower_field(self, context, builder, val, key):
        return self.arrayviewtype.type.lower_getitem_field_record(
            context, builder, self, val, key
        )


@numba.extending.register_model(RecordViewType)
class RecordViewModel(numba.core.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("arrayview", fe_type.arrayviewtype), ("at", numba.intp)]
        super(RecordViewModel, self).__init__(dmm, fe_type, members)


@numba.core.imputils.lower_constant(RecordViewType)
def lower_const_Record(context, builder, recordviewtype, record):
    arrayview_val = lower_const_view(
        context, builder, recordviewtype.arrayviewtype, record._numbaview.arrayview
    )
    proxyout = context.make_helper(builder, recordviewtype)
    proxyout.arrayview = arrayview_val
    proxyout.at = context.get_constant(numba.intp, record._layout.at)
    return proxyout._getvalue()


@numba.extending.unbox(RecordViewType)
def unbox_RecordView(recordviewtype, recordobj, c):
    recordview_obj = c.pyapi.object_getattr_string(recordobj, "_numbaview")
    arrayview_obj = c.pyapi.object_getattr_string(recordview_obj, "arrayview")
    at_obj = c.pyapi.object_getattr_string(recordview_obj, "at")

    arrayview_val = unbox_ArrayView(
        recordviewtype.arrayviewtype, arrayview_obj, c
    ).value

    proxyout = c.context.make_helper(c.builder, recordviewtype)
    proxyout.arrayview = arrayview_val
    proxyout.at = c.pyapi.number_as_ssize_t(at_obj)

    c.pyapi.decref(recordview_obj)
    c.pyapi.decref(at_obj)

    if c.context.enable_nrt:
        c.context.nrt.decref(c.builder, recordviewtype.arrayviewtype, arrayview_val)

    is_error = numba.core.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(proxyout._getvalue(), is_error)


@numba.extending.box(RecordViewType)
def box_RecordView(recordviewtype, viewval, c):
    RecordView_obj = c.pyapi.unserialize(c.pyapi.serialize_object(RecordView))

    proxyin = c.context.make_helper(c.builder, recordviewtype, viewval)
    arrayview_obj = box_ArrayView(recordviewtype.arrayviewtype, proxyin.arrayview, c)
    at_obj = c.pyapi.long_from_ssize_t(proxyin.at)

    recordview_obj = c.pyapi.call_function_objargs(
        RecordView_obj, (arrayview_obj, at_obj)
    )

    out = c.pyapi.call_method(recordview_obj, "torecord", ())

    c.pyapi.decref(RecordView_obj)
    c.pyapi.decref(arrayview_obj)
    c.pyapi.decref(at_obj)
    c.pyapi.decref(recordview_obj)

    return out


@numba.core.typing.templates.infer_global(operator.getitem)
class type_getitem_record(numba.core.typing.templates.AbstractTemplate):
    def generic(self, args, kwargs):
        if len(args) == 2 and len(kwargs) == 0 and isinstance(args[0], RecordViewType):
            recordviewtype, wheretype = args
            if isinstance(wheretype, numba.types.StringLiteral):
                return recordviewtype.arrayviewtype.type.getitem_field_record(
                    recordviewtype, wheretype.literal_value
                )(recordviewtype, wheretype)
            else:
                raise TypeError(
                    "only a *constant* field name string may be used as "
                    "ak.Record slices in compiled code"
                    + ak._util.exception_suffix(__file__)
                )


@numba.extending.lower_builtin(
    operator.getitem, RecordViewType, numba.types.StringLiteral
)
def lower_getitem_field_record(context, builder, sig, args):
    _, (recordviewtype, wheretype) = sig.return_type, sig.args
    recordviewval, whereval = args
    return recordviewtype.arrayviewtype.type.lower_getitem_field_record(
        context, builder, recordviewtype, recordviewval, wheretype.literal_value
    )


@numba.core.typing.templates.infer_getattr
class type_getattr_record(numba.core.typing.templates.AttributeTemplate):
    key = RecordViewType

    def generic_resolve(self, recordviewtype, attr):
        for methodname, typer, lower in ak._util.numba_methods(
            recordviewtype.arrayviewtype.type, recordviewtype.arrayviewtype.behavior
        ):
            if attr == methodname:

                class type_method(numba.core.typing.templates.AbstractTemplate):
                    key = methodname

                    def generic(self, args, kwargs):
                        if len(kwargs) == 0:
                            sig = typer(recordviewtype, args)
                            sig = numba.core.typing.templates.Signature(
                                sig.return_type, sig.args, recordviewtype
                            )
                            numba.extending.lower_builtin(
                                methodname,
                                recordviewtype,
                                *[
                                    x.literal_type
                                    if isinstance(x, numba.types.Literal)
                                    else x
                                    for x in args
                                ]
                            )(lower)
                            return sig

                return numba.types.BoundFunction(type_method, recordviewtype)

        for attrname, typer, _ in ak._util.numba_attrs(
            recordviewtype.arrayviewtype.type, recordviewtype.arrayviewtype.behavior
        ):
            if attr == attrname:
                return typer(recordviewtype)

        else:
            return recordviewtype.typer_field(attr)


@numba.extending.lower_getattr_generic(RecordViewType)
def lower_getattr_generic_record(context, builder, recordviewtype, recordviewval, attr):
    for attrname, typer, lower in ak._util.numba_attrs(
        recordviewtype.arrayviewtype.type, recordviewtype.arrayviewtype.behavior
    ):
        if attr == attrname:
            return lower(
                context,
                builder,
                typer(recordviewtype)(recordviewtype),
                (recordviewval,),
            )
    else:
        return recordviewtype.lower_field(context, builder, recordviewval, attr)


def register_unary_operator(unaryop):
    @numba.core.typing.templates.infer_global(unaryop)
    class type_binary_operator(numba.core.typing.templates.AbstractTemplate):
        def generic(self, args, kwargs):
            if len(args) == 1 and len(kwargs) == 0:
                behavior = None

                if isinstance(args[0], RecordViewType):
                    left = args[0].arrayviewtype.type
                    behavior = args[0].arrayviewtype.behavior

                    for typer, lower in ak._util.numba_unaryops(
                        unaryop, left, behavior
                    ):
                        numba.extending.lower_builtin(unaryop, *args)(lower)
                        return typer(unaryop, args[0])


for unaryop in (
    abs,
    operator.inv,
    operator.invert,
    operator.neg,
    operator.not_,
    operator.pos,
    operator.truth,
):
    register_unary_operator(unaryop)


def register_binary_operator(binop):
    @numba.core.typing.templates.infer_global(binop)
    class type_binary_operator(numba.core.typing.templates.AbstractTemplate):
        def generic(self, args, kwargs):
            if len(args) == 2 and len(kwargs) == 0:
                left, right, behavior = None, None, None

                if isinstance(args[0], RecordViewType):
                    left = args[0].arrayviewtype.type
                    behavior = args[0].arrayviewtype.behavior

                if isinstance(args[1], RecordViewType):
                    right = args[1].arrayviewtype.type
                    if behavior is None:
                        behavior = args[1].arrayviewtype.behavior

                if left is not None or right is not None:
                    for typer, lower in ak._util.numba_binops(
                        binop, left, right, behavior
                    ):
                        numba.extending.lower_builtin(binop, *args)(lower)
                        return typer(binop, args[0], args[1])


for binop in (
    operator.add,
    operator.and_,
    operator.contains,
    operator.eq,
    operator.floordiv,
    operator.ge,
    operator.gt,
    operator.le,
    operator.lshift,
    operator.lt,
    operator.mod,
    operator.mul,
    operator.ne,
    operator.or_,
    operator.pow,
    operator.rshift,
    operator.sub,
    operator.truediv,
    operator.xor,
) + (() if not hasattr(operator, "matmul") else (operator.matmul,)):
    register_binary_operator(binop)


########## __contains__


@numba.extending.overload(operator.contains)
def overload_contains(obj, element):
    if isinstance(obj, (ArrayViewType, RecordViewType)) and (
        (element == numba.types.none)
        or (isinstance(element, (numba.types.Number, numba.types.Boolean)))
        or (
            isinstance(element, numba.types.Optional)
            and isinstance(element.type, (numba.types.Number, numba.types.Boolean))
        )
    ):
        statements = []

        def add_statement(indent, name, arraytype, is_array):
            if is_array:
                statements.append("for x in " + name + ":")
                name = "x"
                indent = indent + "    "

            if isinstance(arraytype, ak._connect._numba.layout.RecordArrayType):
                if arraytype.is_tuple:
                    for fi, ft in enumerate(arraytype.contenttypes):
                        add_statement(indent, name + "[" + repr(fi) + "]", ft, False)
                else:
                    for fn, ft in zip(arraytype.recordlookup, arraytype.contenttypes):
                        add_statement(indent, name + "[" + repr(fn) + "]", ft, False)

            elif arraytype.ndim == 1 and not arraytype.is_recordtype:
                if arraytype.is_optiontype:
                    statements.append(
                        indent + "if (element is None and {0} is None) or "
                        "({0} is not None and element == {0}): return True".format(name)
                    )
                else:
                    statements.append(
                        indent + "if element == {0}: return True".format(name)
                    )

            else:
                if arraytype.is_optiontype:
                    statements.append(
                        indent + "if (element is None and {0} is None) or "
                        "({0} is not None and element in {0}): return True".format(name)
                    )
                else:
                    statements.append(
                        indent + "if element in {0}: return True".format(name)
                    )

        if isinstance(obj, ArrayViewType):
            add_statement("", "obj", obj.type, True)
        else:
            add_statement("", "obj", obj.arrayviewtype.type, False)

        return code_to_function(
            """
def contains_impl(obj, element):
    {0}
    return False""".format(
                "\n    ".join(statements)
            ),
            "contains_impl",
        )


########## np.array and np.asarray


array_supported = (
    numba.types.boolean,
    numba.types.int8,
    numba.types.int16,
    numba.types.int32,
    numba.types.int64,
    numba.types.uint8,
    numba.types.uint16,
    numba.types.uint32,
    numba.types.uint64,
    numba.types.float32,
    numba.types.float64,
)


@numba.extending.overload(ak.nplike.numpy.array)
def overload_np_array(array, dtype=None):
    if isinstance(array, ArrayViewType):
        ndim = array.type.ndim
        inner_dtype = array.type.inner_dtype
        if ndim is not None and inner_dtype in array_supported:
            declare_shape = []
            compute_shape = []
            specify_shape = ["len(array)"]
            ensure_shape = []
            array_name = "array"
            for i in range(ndim - 1):
                declare_shape.append("shape{0} = -1".format(i))
                compute_shape.append(
                    "{0}for x{1} in {2}:".format("    " * i, i, array_name)
                )
                compute_shape.append("{0}    if shape{1} == -1:".format("    " * i, i))
                compute_shape.append(
                    "{0}        shape{1} = len(x{1})".format("    " * i, i)
                )
                compute_shape.append(
                    "{0}    elif shape{1} != len(x{1}):".format("    " * i, i)
                )
                compute_shape.append(
                    "{0}        raise ValueError('cannot convert to NumPy because "
                    "subarray lengths are not regular')".format("    " * i)
                )
                specify_shape.append("shape{0}".format(i))
                ensure_shape.append("if shape{0} == -1: shape{0} = 0".format(i))
                array_name = "x{0}".format(i)

            fill_array = []
            index = []
            array_name = "array"
            for i in range(ndim):
                fill_array.append(
                    "{0}for i{1}, x{1} in enumerate({2}):".format(
                        "    " * i, i, array_name
                    )
                )
                index.append("i{0}".format(i))
                array_name = "x{0}".format(i)

            fill_array.append(
                "{0}out[{1}] = x{2}".format("    " * ndim, "][".join(index), ndim - 1)
            )

            return code_to_function(
                """
def array_impl(array, dtype=None):
    {0}
    {1}
    {2}
    out = numpy.zeros(({3}), {4})
    {5}
    return out
""".format(
                    "\n    ".join(declare_shape),
                    "\n    ".join(compute_shape),
                    "\n    ".join(ensure_shape),
                    ", ".join(specify_shape),
                    "numpy.{0}".format(inner_dtype) if dtype is None else "dtype",
                    "\n    ".join(fill_array),
                ),
                "array_impl",
                {"numpy": ak.nplike.numpy},
            )


@numba.extending.type_callable(ak.nplike.numpy.asarray)
def type_asarray(context):
    def typer(arrayview):
        if (
            isinstance(arrayview, ArrayViewType)
            and isinstance(arrayview.type, ak._connect._numba.layout.NumpyArrayType)
            and arrayview.type.ndim == 1
            and arrayview.type.inner_dtype in array_supported
        ):
            return numba.types.Array(arrayview.type.inner_dtype, 1, "C")

    return typer


@numba.extending.lower_builtin(ak.nplike.numpy.asarray, ArrayViewType)
def lower_asarray(context, builder, sig, args):
    rettype, (viewtype,) = sig.return_type, sig.args
    (viewval,) = args
    viewproxy = context.make_helper(builder, viewtype, viewval)
    assert isinstance(viewtype.type, ak._connect._numba.layout.NumpyArrayType)

    whichpos = ak._connect._numba.layout.posat(
        context, builder, viewproxy.pos, viewtype.type.ARRAY
    )
    arrayptr = ak._connect._numba.layout.getat(
        context, builder, viewproxy.arrayptrs, whichpos
    )

    bitwidth = ak._connect._numba.layout.type_bitwidth(rettype.dtype)
    itemsize = context.get_constant(numba.intp, bitwidth // 8)

    data = numba.core.cgutils.pointer_add(
        builder,
        arrayptr,
        builder.mul(viewproxy.start, itemsize),
        context.get_value_type(numba.types.CPointer(rettype.dtype)),
    )

    shape = context.make_tuple(
        builder,
        numba.types.UniTuple(numba.types.intp, 1),
        (builder.sub(viewproxy.stop, viewproxy.start),),
    )
    strides = context.make_tuple(
        builder,
        numba.types.UniTuple(numba.types.intp, 1),
        (itemsize,),
    )

    out = numba.np.arrayobj.make_array(rettype)(context, builder)
    numba.np.arrayobj.populate_array(
        out,
        data=data,
        shape=shape,
        strides=strides,
        itemsize=itemsize,
        meminfo=None,
        parent=None,
    )
    return out._getvalue()


########## PartitionedView


class PartitionedView(object):
    def __init__(self, type, behavior, lookups, stops, start, stop, fields):
        self.type = type
        self.behavior = behavior
        self.lookups = lookups
        self.stops = stops
        self.start = start
        self.stop = stop
        self.fields = fields

    def toarray(self):
        output = []
        partition_start = 0
        for partitionid, lookup in enumerate(self.lookups):
            partition_stop = self.stops[partitionid]

            if partition_start <= self.start and self.stop <= partition_stop:
                layout = self.type.tolayout(lookup, 0, self.fields)
                output.append(
                    layout[self.start - partition_start : self.stop - partition_start]
                )
                break

            elif partition_start <= self.start < partition_stop:
                layout = self.type.tolayout(lookup, 0, self.fields)
                output.append(
                    layout[
                        self.start - partition_start : partition_stop - partition_start
                    ]
                )

            elif partition_start < self.stop <= partition_stop:
                layout = self.type.tolayout(lookup, 0, self.fields)
                output.append(layout[0 : self.stop - partition_start])
                break

            elif self.start < partition_start and partition_stop < self.stop:
                layout = self.type.tolayout(lookup, 0, self.fields)
                output.append(layout[0 : partition_stop - partition_start])

            partition_start = partition_stop

        return ak._util.wrap(
            ak.partition.IrregularlyPartitionedArray(output), self.behavior
        )


@numba.extending.typeof_impl.register(PartitionedView)
def typeof_PartitionedView(obj, c):
    return PartitionedViewType(obj.type, obj.behavior, obj.fields)


class PartitionedViewType(numba.types.IterableType, numba.types.Sized):
    stopstype = numba.types.Array(numba.intp, 1, "C")

    def __init__(self, type, behavior, fields):
        super(PartitionedViewType, self).__init__(
            name="ak.PartitionedView({0}, {1}, {2})".format(
                type.name,
                ak._connect._numba.repr_behavior(behavior),
                repr(fields),
            )
        )
        self.type = type
        self.behavior = behavior
        self.fields = fields

    @property
    def iterator_type(self):
        return PartitionedIteratorType(self)

    def toArrayViewType(self):
        return ArrayViewType(self.type, self.behavior, self.fields)

    def getitem_field(self, key):
        return PartitionedViewType(self.type, self.behavior, self.fields + (key,))

    def lower_get_localstart(self, context, builder, stops, partitionid):
        out = numba.core.cgutils.alloca_once_value(
            builder, context.get_constant(numba.intp, 0)
        )

        with builder.if_then(
            builder.icmp_signed("!=", partitionid, context.get_constant(numba.intp, 0))
        ):
            stopsproxy = context.make_helper(builder, self.stopstype, stops)
            newval = numba.np.arrayobj._getitem_array_single_int(
                context,
                builder,
                numba.intp,
                self.stopstype,
                stopsproxy,
                builder.sub(partitionid, context.get_constant(numba.intp, 1)),
            )
            builder.store(newval, out)

        return builder.load(out)

    def lower_get_localstop(self, context, builder, stops, partitionid):
        stopsproxy = context.make_helper(builder, self.stopstype, stops)
        return numba.np.arrayobj._getitem_array_single_int(
            context, builder, numba.intp, self.stopstype, stopsproxy, partitionid
        )

    def lower_get_partitionid(
        self, context, builder, pyapi, pylookups, partitionid, viewlength
    ):
        lookup_obj = pyapi.list_getitem(pylookups, partitionid)  # borrowed
        lookup = pyapi.to_native_value(LookupType(), lookup_obj).value
        lookupproxy = context.make_helper(builder, LookupType(), value=lookup)

        viewproxy = context.make_helper(builder, self.toArrayViewType())
        viewproxy.pos = context.get_constant(numba.intp, 0)
        viewproxy.start = context.get_constant(numba.intp, 0)
        viewproxy.stop = viewlength
        viewproxy.arrayptrs = context.make_helper(
            builder, LookupType.arraytype, lookupproxy.arrayptrs
        ).data
        viewproxy.sharedptrs = context.make_helper(
            builder, LookupType.arraytype, lookupproxy.sharedptrs
        ).data
        viewproxy.pylookup = lookup_obj
        return viewproxy._getvalue()


@numba.extending.register_model(PartitionedViewType)
class PartitionedViewModel(numba.core.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("pylookups", numba.types.pyobject),
            ("partitionid", numba.types.CPointer(numba.intp)),
            ("stops", fe_type.stopstype),
            ("view", numba.types.CPointer(fe_type.toArrayViewType())),
            ("start", numba.intp),
            ("stop", numba.intp),
        ]
        super(PartitionedViewModel, self).__init__(dmm, fe_type, members)


@numba.extending.unbox(PartitionedViewType)
def unbox_PartitionedArray(partviewtype, arrayobj, c):
    partview_obj = c.pyapi.object_getattr_string(arrayobj, "_numbaview")
    out = unbox_PartitionedView(partviewtype, partview_obj, c)
    c.pyapi.decref(partview_obj)
    return out


def unbox_PartitionedView(partviewtype, partview_obj, c):
    lookups_obj = c.pyapi.object_getattr_string(partview_obj, "lookups")
    stops_obj = c.pyapi.object_getattr_string(partview_obj, "stops")
    start_obj = c.pyapi.object_getattr_string(partview_obj, "start")
    stop_obj = c.pyapi.object_getattr_string(partview_obj, "stop")

    proxyout = c.context.make_helper(c.builder, partviewtype)
    proxyout.pylookups = lookups_obj
    partitionid = c.context.get_constant(numba.intp, 0)
    proxyout.partitionid = numba.core.cgutils.alloca_once_value(c.builder, partitionid)
    proxyout.stops = c.pyapi.to_native_value(partviewtype.stopstype, stops_obj).value
    viewlength = partviewtype.lower_get_localstop(
        c.context, c.builder, proxyout.stops, partitionid
    )
    proxyout.view = numba.core.cgutils.alloca_once_value(
        c.builder,
        partviewtype.lower_get_partitionid(
            c.context, c.builder, c.pyapi, proxyout.pylookups, partitionid, viewlength
        ),
    )
    proxyout.start = c.pyapi.number_as_ssize_t(start_obj)
    proxyout.stop = c.pyapi.number_as_ssize_t(stop_obj)

    c.pyapi.decref(lookups_obj)
    c.pyapi.decref(stops_obj)
    c.pyapi.decref(start_obj)
    c.pyapi.decref(stop_obj)

    is_error = numba.core.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(proxyout._getvalue(), is_error)


@numba.extending.box(PartitionedViewType)
def box_PartitionedArray(partviewtype, partviewval, c):
    arrayview_obj = box_PartitionedView(partviewtype, partviewval, c)
    out = c.pyapi.call_method(arrayview_obj, "toarray", ())
    c.pyapi.decref(arrayview_obj)
    return out


def box_PartitionedView(partviewtype, partviewval, c):
    serializable2dict_obj = c.pyapi.unserialize(
        c.pyapi.serialize_object(serializable2dict)
    )
    behavior2_obj = c.pyapi.unserialize(
        c.pyapi.serialize_object(dict2serializable(partviewtype.behavior))
    )
    behavior_obj = c.pyapi.call_function_objargs(
        serializable2dict_obj, (behavior2_obj,)
    )
    PartitionedView_obj = c.pyapi.unserialize(c.pyapi.serialize_object(PartitionedView))
    type_obj = c.pyapi.unserialize(c.pyapi.serialize_object(partviewtype.type))
    fields_obj = c.pyapi.unserialize(c.pyapi.serialize_object(partviewtype.fields))

    proxyin = c.context.make_helper(c.builder, partviewtype, partviewval)
    lookups_obj = proxyin.pylookups
    stops_obj = c.pyapi.from_native_value(
        partviewtype.stopstype, proxyin.stops, c.env_manager
    )
    start_obj = c.pyapi.long_from_ssize_t(proxyin.start)
    stop_obj = c.pyapi.long_from_ssize_t(proxyin.stop)

    out = c.pyapi.call_function_objargs(
        PartitionedView_obj,
        (
            type_obj,
            behavior_obj,
            lookups_obj,
            stops_obj,
            start_obj,
            stop_obj,
            fields_obj,
        ),
    )

    c.pyapi.decref(serializable2dict_obj)
    c.pyapi.decref(behavior2_obj)
    c.pyapi.decref(behavior_obj)
    c.pyapi.decref(PartitionedView_obj)
    c.pyapi.decref(type_obj)
    c.pyapi.decref(fields_obj)
    c.pyapi.decref(stops_obj)
    c.pyapi.decref(start_obj)
    c.pyapi.decref(stop_obj)

    return out


@numba.core.typing.templates.infer_global(operator.getitem)
class type_getitem_partitioned(numba.core.typing.templates.AbstractTemplate):
    def generic(self, args, kwargs):
        if (
            len(args) == 2
            and len(kwargs) == 0
            and isinstance(args[0], PartitionedViewType)
        ):
            partviewtype, wheretype = args

            if isinstance(wheretype, numba.types.Integer):
                arrayviewtype = partviewtype.toArrayViewType()
                rettype = partviewtype.type.getitem_at_check(arrayviewtype)
                return rettype(partviewtype, wheretype)

            elif (
                isinstance(wheretype, numba.types.SliceType) and not wheretype.has_step
            ):
                return partviewtype(partviewtype, wheretype)

            elif isinstance(wheretype, numba.types.StringLiteral):
                rettype = partviewtype.getitem_field(wheretype.literal_value)
                return rettype(partviewtype, wheretype)

            else:
                raise TypeError(
                    "only an integer, start:stop range, or a *constant* "
                    "field name string may be used as ak.Array "
                    "slices in compiled code" + ak._util.exception_suffix(__file__)
                )


@numba.extending.lower_builtin(
    operator.getitem, PartitionedViewType, numba.types.Integer
)
def lower_getitem_at_partitioned(context, builder, sig, args):
    rettype, (partviewtype, wheretype) = sig.return_type, sig.args
    partviewval, whereval = args
    partviewproxy = context.make_helper(builder, partviewtype, partviewval)

    length = builder.sub(partviewproxy.stop, partviewproxy.start)
    regular_atval = numba.core.cgutils.alloca_once_value(builder, whereval)

    with builder.if_then(
        builder.icmp_signed("<", whereval, context.get_constant(numba.intp, 0))
    ):
        builder.store(builder.add(whereval, length), regular_atval)
    atval = builder.load(regular_atval)

    with builder.if_then(
        builder.or_(
            builder.icmp_signed("<", atval, context.get_constant(numba.intp, 0)),
            builder.icmp_signed(">=", atval, length),
        )
    ):
        context.call_conv.return_user_exc(
            builder, ValueError, ("slice index out of bounds",)
        )

    localstart = partviewtype.lower_get_localstart(
        context, builder, partviewproxy.stops, builder.load(partviewproxy.partitionid)
    )
    localstop = partviewtype.lower_get_localstop(
        context, builder, partviewproxy.stops, builder.load(partviewproxy.partitionid)
    )

    with builder.if_then(
        builder.not_(
            builder.and_(
                builder.icmp_signed("<=", localstart, atval),
                builder.icmp_signed(">", atval, localstop),
            )
        ),
        likely=False,
    ):

        searchsorted_sig = numba.intp(partviewtype.stopstype, wheretype)
        searchsorted_args = (partviewproxy.stops, atval)

        def searchsorted_impl(stops, where):
            return ak.nplike.numpy.searchsorted(stops, where, side="right")

        partitionid_val = context.compile_internal(
            builder, searchsorted_impl, searchsorted_sig, searchsorted_args
        )
        builder.store(partitionid_val, partviewproxy.partitionid)

        pyapi = context.get_python_api(builder)
        gil = pyapi.gil_ensure()
        builder.store(
            partviewtype.lower_get_partitionid(
                context,
                builder,
                pyapi,
                partviewproxy.pylookups,
                builder.load(partviewproxy.partitionid),
                builder.sub(localstop, localstart),
            ),
            partviewproxy.view,
        )
        pyapi.gil_release(gil)

    viewtype = partviewtype.toArrayViewType()
    viewval = builder.load(partviewproxy.view)
    viewproxy = context.make_helper(builder, viewtype, value=viewval)

    reallocalstart = partviewtype.lower_get_localstart(
        context, builder, partviewproxy.stops, builder.load(partviewproxy.partitionid)
    )
    subatval = builder.sub(atval, reallocalstart)

    return viewtype.type.lower_getitem_at_check(
        context,
        builder,
        rettype,
        viewtype,
        viewval,
        viewproxy,
        numba.intp,
        subatval,
        False,
        False,
    )


@numba.extending.lower_builtin(
    operator.getitem, PartitionedViewType, numba.types.slice2_type
)
def lower_getitem_range_partitioned(context, builder, sig, args):
    _, (partviewtype, wheretype) = sig.return_type, sig.args
    partviewval, whereval = args

    whereproxy = context.make_helper(builder, wheretype, whereval)
    start = whereproxy.start
    stop = whereproxy.stop

    partviewproxy = context.make_helper(builder, partviewtype, partviewval)
    length = builder.sub(partviewproxy.stop, partviewproxy.start)

    regular_start = numba.core.cgutils.alloca_once_value(builder, start)
    regular_stop = numba.core.cgutils.alloca_once_value(builder, stop)

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
    with builder.if_then(builder.icmp_signed(">", builder.load(regular_start), length)):
        builder.store(length, regular_start)

    with builder.if_then(
        builder.icmp_signed(
            "<", builder.load(regular_stop), builder.load(regular_start)
        )
    ):
        builder.store(builder.load(regular_start), regular_stop)
    with builder.if_then(builder.icmp_signed(">", builder.load(regular_stop), length)):
        builder.store(length, regular_stop)

    proxyout = context.make_helper(builder, partviewtype)
    proxyout.pylookups = partviewproxy.pylookups
    proxyout.partitionid = numba.core.cgutils.alloca_once_value(
        builder, builder.load(partviewproxy.partitionid)
    )
    proxyout.stops = partviewproxy.stops
    proxyout.view = numba.core.cgutils.alloca_once_value(
        builder, builder.load(partviewproxy.view)
    )
    proxyout.start = builder.load(regular_start)
    proxyout.stop = builder.load(regular_stop)

    if context.enable_nrt:
        context.nrt.incref(builder, partviewtype.stopstype, proxyout.stops)

    return proxyout._getvalue()


@numba.extending.lower_builtin(
    operator.getitem, PartitionedViewType, numba.types.StringLiteral
)
def lower_getitem_field_partitioned(context, builder, sig, args):
    _, (partviewtype, _) = sig.return_type, sig.args
    partviewval, whereval = args

    if context.enable_nrt:
        partviewproxy = context.make_helper(builder, partviewtype, partviewval)
        context.nrt.incref(builder, partviewtype.stopstype, partviewproxy.stops)

    return partviewval


@numba.core.typing.templates.infer_global(len)
class type_len_partitioned(numba.core.typing.templates.AbstractTemplate):
    def generic(self, args, kwargs):
        if (
            len(args) == 1
            and len(kwargs) == 0
            and isinstance(args[0], PartitionedViewType)
        ):
            return numba.intp(args[0])


@numba.extending.lower_builtin(len, PartitionedViewType)
def lower_len_partitioned(context, builder, sig, args):
    proxyin = context.make_helper(builder, sig.args[0], args[0])
    return builder.sub(proxyin.stop, proxyin.start)


@numba.core.typing.templates.infer_getattr
class type_getattr_partitioned(numba.core.typing.templates.AttributeTemplate):
    key = PartitionedViewType

    def generic_resolve(self, partviewtype, attr):
        if attr == "ndim":
            return numba.intp
        else:
            return partviewtype.getitem_field(attr)


@numba.extending.lower_getattr_generic(PartitionedViewType)
def lower_getattr_generic_partitioned(
    context, builder, partviewtype, partviewval, attr
):
    if attr == "ndim":
        return context.get_constant(numba.intp, partviewtype.type.ndim)
    elif context.enable_nrt:
        partviewproxy = context.make_helper(builder, partviewtype, partviewval)
        context.nrt.incref(builder, partviewtype.stopstype, partviewproxy.stops)

    return partviewval


class PartitionedIteratorType(numba.types.common.SimpleIteratorType):
    def __init__(self, partviewtype):
        super(PartitionedIteratorType, self).__init__(
            "ak.PartitionedIterator({0})".format(partviewtype.name),
            partviewtype.type.getitem_at_check(partviewtype.toArrayViewType()),
        )
        self.partviewtype = partviewtype


@numba.core.typing.templates.infer
class type_getiter_partitioned(numba.core.typing.templates.AbstractTemplate):
    key = "getiter"

    def generic(self, args, kwargs):
        if (
            len(args) == 1
            and len(kwargs) == 0
            and isinstance(args[0], PartitionedViewType)
        ):
            return PartitionedIteratorType(args[0])(args[0])


@numba.core.datamodel.registry.register_default(PartitionedIteratorType)
class PartitionedIteratorModel(numba.core.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("partview", fe_type.partviewtype),
            ("length", numba.intp),
            ("at", numba.types.EphemeralPointer(numba.intp)),
        ]
        super(PartitionedIteratorModel, self).__init__(dmm, fe_type, members)


@numba.extending.lower_builtin("getiter", PartitionedViewType)
def lower_getiter_partitioned(context, builder, sig, args):
    rettype, (partviewtype,) = sig.return_type, sig.args
    (partviewval,) = args
    partviewproxy = context.make_helper(builder, partviewtype, partviewval)

    partitionid = context.get_constant(numba.intp, 0)
    viewlength = partviewtype.lower_get_localstop(
        context, builder, partviewproxy.stops, partitionid
    )

    partoutproxy = context.make_helper(builder, partviewtype)
    partoutproxy.pylookups = partviewproxy.pylookups
    partoutproxy.partitionid = numba.core.cgutils.alloca_once_value(
        builder, partitionid
    )
    partoutproxy.stops = partviewproxy.stops

    pyapi = context.get_python_api(builder)
    gil = pyapi.gil_ensure()
    partoutproxy.view = numba.core.cgutils.alloca_once_value(
        builder,
        partviewtype.lower_get_partitionid(
            context, builder, pyapi, partviewproxy.pylookups, partitionid, viewlength
        ),
    )
    pyapi.gil_release(gil)

    partoutproxy.start = partviewproxy.start
    partoutproxy.stop = partviewproxy.stop

    proxyout = context.make_helper(builder, rettype)
    proxyout.partview = partoutproxy._getvalue()
    proxyout.length = builder.sub(partviewproxy.stop, partviewproxy.start)
    proxyout.at = numba.core.cgutils.alloca_once_value(
        builder, context.get_constant(numba.intp, 0)
    )

    if context.enable_nrt:
        context.nrt.incref(builder, partviewtype.stopstype, partoutproxy.stops)

    return numba.core.imputils.impl_ret_new_ref(
        context, builder, rettype, proxyout._getvalue()
    )


@numba.extending.lower_builtin("iternext", PartitionedIteratorType)
@numba.core.imputils.iternext_impl(numba.core.imputils.RefType.BORROWED)
def lower_iternext_partitioned(context, builder, sig, args, result):
    (itertype,) = sig.args
    (iterval,) = args

    proxyin = context.make_helper(builder, itertype, iterval)
    partviewproxy = context.make_helper(
        builder, itertype.partviewtype, proxyin.partview
    )
    at = builder.load(proxyin.at)

    is_valid = builder.icmp_signed("<", at, proxyin.length)
    result.set_valid(is_valid)

    with builder.if_then(is_valid, likely=True):
        maybestop = itertype.partviewtype.lower_get_localstop(
            context,
            builder,
            partviewproxy.stops,
            builder.load(partviewproxy.partitionid),
        )

        with builder.if_then(builder.icmp_signed("==", at, maybestop)):
            builder.store(
                builder.add(
                    builder.load(partviewproxy.partitionid),
                    context.get_constant(numba.intp, 1),
                ),
                partviewproxy.partitionid,
            )

            localstart = itertype.partviewtype.lower_get_localstart(
                context,
                builder,
                partviewproxy.stops,
                builder.load(partviewproxy.partitionid),
            )
            localstop = itertype.partviewtype.lower_get_localstop(
                context,
                builder,
                partviewproxy.stops,
                builder.load(partviewproxy.partitionid),
            )

            pyapi = context.get_python_api(builder)
            gil = pyapi.gil_ensure()
            builder.store(
                itertype.partviewtype.lower_get_partitionid(
                    context,
                    builder,
                    pyapi,
                    partviewproxy.pylookups,
                    builder.load(partviewproxy.partitionid),
                    builder.sub(localstop, localstart),
                ),
                partviewproxy.view,
            )
            pyapi.gil_release(gil)

        realstart = itertype.partviewtype.lower_get_localstart(
            context,
            builder,
            partviewproxy.stops,
            builder.load(partviewproxy.partitionid),
        )

        outview = builder.load(partviewproxy.view)
        outviewtype = itertype.partviewtype.toArrayViewType()
        outviewproxy = context.make_helper(builder, outviewtype, outview)

        result.yield_(
            itertype.partviewtype.type.lower_getitem_at_check(
                context,
                builder,
                itertype.partviewtype.type.getitem_at_check(outviewtype),
                outviewtype,
                outview,
                outviewproxy,
                numba.intp,
                builder.sub(at, realstart),
                False,
                False,
            )
        )

        nextat = numba.core.cgutils.increment_index(builder, at)
        builder.store(nextat, proxyin.at)
