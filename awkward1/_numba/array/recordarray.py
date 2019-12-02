# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import operator

import numpy
import numba
import numba.typing.arraydecl

import awkward1.layout
from ..._numba import cpu, util, content

@numba.extending.typeof_impl.register(awkward1.layout.RecordArray)
def typeof(val, c):
    return RecordArrayType([numba.typeof(x) for x in val.values()], val.lookup, val.reverselookup, numba.typeof(val.id))

@numba.extending.typeof_impl.register(awkward1.layout.Record)
def typeof(val, c):
    return RecordType(numba.typeof(val.array))

class RecordArrayType(content.ContentType):
    def __init__(self, contenttpes, lookup, reverselookup, idtpe):
        super(RecordArrayType, self).__init__(name="RecordArrayType([{}], {}, {}, id={})".format(", ".join(x.name for x in contenttpes), lookup, reverselookup, idtpe.name))
        self.contenttpes = contenttpes
        self.lookup = lookup
        self.reverselookup = reverselookup
        self.idtpe = idtpe

    @property
    def istuple(self):
        return self.lookup is None

    @property
    def numfields(self):
        return len(self.contenttpes)

    @property
    def ndim(self):
        return 1

    def getitem_int(self):
        return RecordType(self)

    def getitem_range(self):
        return self

    def getitem_str(self, key):
        return self.contenttpes[awkward1.util.field2index(self.lookup, self.numfields, key)]

    def getitem_tuple(self, wheretpe):
        nexttpe = RegularArrayType(self, numba.none)
        out = nexttpe.getitem_next(wheretpe, False)
        return out.getitem_int()

    def getitem_next(self, wheretpe, isadvanced):
        if len(wheretpe.types) == 0:
            return self
        headtpe = wheretpe.types[0]
        tailtpe = numba.types.Tuple(wheretpe.types[1:])

        raise NotImplementedError

    def carry(self):
        return RecordArrayType([x.carry() for x in self.contenttpes], self.lookup, self.idtpe)

    @property
    def lower_len(self):
        return lower_len

    @property
    def lower_getitem_nothing(self):
        return content.lower_getitem_nothing

    @property
    def lower_getitem_int(self):
        return lower_getitem_int

    @property
    def lower_getitem_range(self):
        return lower_getitem_range

    @property
    def lower_getitem_str(self):
        return lower_getitem_str

    @property
    def lower_getitem_next(self):
        return lower_getitem_next

    @property
    def lower_carry(self):
        return lower_carry

class RecordType(numba.types.Type):
    def __init__(self, arraytpe):
        self.arraytpe = arraytpe
        super(RecordType, self).__init__("Record({})".format(self.arraytpe.name))
        assert isinstance(arraytpe, RecordArrayType)

    @property
    def istuple(self):
        return self.arraytpe.istuple

    def getitem_str(self, key):
        outtpe = self.arraytpe.getitem_str(key)
        return outtpe.getitem_int()

    def getitem_tuple(self, wheretpe):
        nextwheretpe = numba.types.Tuple((numba.int64,) + wheretpe.types)
        return self.arraytpe.getitem_tuple(nextwheretpe)

@numba.typing.templates.infer_global(operator.getitem)
class type_getitem_record(numba.typing.templates.AbstractTemplate):
    def generic(self, args, kwargs):
        if len(args) == 2 and len(kwargs) == 0:
            tpe, wheretpe = args

            if isinstance(tpe, RecordType):
                original_wheretpe = wheretpe
                if isinstance(wheretpe, numba.types.StringLiteral):
                    return numba.typing.templates.signature(tpe.getitem_str(wheretpe.literal_value), tpe, original_wheretpe)

                if not isinstance(wheretpe, numba.types.BaseTuple):
                    wheretpe = numba.types.Tuple((wheretpe,))

                wheretpe = util.typing_regularize_slice(wheretpe)
                content.type_getitem.check_slice_types(wheretpe)

                return numba.typing.templates.signature(tpe.getitem_tuple(wheretpe), tpe, original_wheretpe)

def field(i):
    return "f" + str(i)

@numba.extending.register_model(RecordArrayType)
class RecordArrayModel(numba.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("length", numba.int64)]
        for i, x in enumerate(fe_type.contenttpes):
            members.append((field(i), x))
        if fe_type.idtpe != numba.none:
            members.append(("id", fe_type.idtpe))
        super(RecordArrayModel, self).__init__(dmm, fe_type, members)

@numba.datamodel.registry.register_default(RecordType)
class RecordModel(numba.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("array", fe_type.arraytpe),
                   ("at", numba.int64)]
        super(RecordModel, self).__init__(dmm, fe_type, members)

@numba.extending.unbox(RecordArrayType)
def unbox(tpe, obj, c):
    len_obj = c.pyapi.unserialize(c.pyapi.serialize_object(len))
    length_obj = c.pyapi.call_function_objargs(len_obj, (obj,))
    proxyout = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder)
    proxyout.length = c.pyapi.to_native_value(numba.int64, length_obj).value
    c.pyapi.decref(len_obj)
    c.pyapi.decref(length_obj)
    field_obj = c.pyapi.object_getattr_string(obj, "field")
    for i, t in enumerate(tpe.contenttpes):
        i_obj = c.pyapi.long_from_longlong(c.context.get_constant(numba.int64, i))
        x_obj = c.pyapi.call_function_objargs(field_obj, (i_obj,))
        setattr(proxyout, field(i), c.pyapi.to_native_value(t, x_obj).value)
        c.pyapi.decref(i_obj)
        c.pyapi.decref(x_obj)
    c.pyapi.decref(field_obj)
    if tpe.idtpe != numba.none:
        id_obj = c.pyapi.object_getattr_string(obj, "id")
        proxyout.id = c.pyapi.to_native_value(tpe.idtpe, id_obj).value
        c.pyapi.decref(id_obj)
    is_error = numba.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(proxyout._getvalue(), is_error)

@numba.extending.unbox(RecordType)
def unbox_record(tpe, obj, c):
    array_obj = c.pyapi.object_getattr_string(obj, "array")
    at_obj = c.pyapi.object_getattr_string(obj, "at")
    proxyout = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder)
    proxyout.array = c.pyapi.to_native_value(tpe.arraytpe, array_obj).value
    proxyout.at = c.pyapi.to_native_value(numba.int64, at_obj).value
    if c.context.enable_nrt:
        c.context.nrt.incref(c.builder, tpe.arraytpe, proxyout.array)
    c.pyapi.decref(array_obj)
    c.pyapi.decref(at_obj)
    is_error = numba.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(proxyout._getvalue(), is_error)

@numba.extending.box(RecordArrayType)
def box(tpe, val, c):
    RecordArray_obj = c.pyapi.unserialize(c.pyapi.serialize_object(awkward1.layout.RecordArray))
    istuple_obj = c.pyapi.unserialize(c.pyapi.serialize_object(tpe.istuple))
    proxyin = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder, value=val)
    length_obj = c.pyapi.long_from_longlong(proxyin.length)
    if tpe.idtpe != numba.none:
        id_obj = c.pyapi.from_native_value(tpe.idtpe, proxyin.id, c.env_manager)
        out = c.pyapi.call_function_objargs(RecordArray_obj, (length_obj, istuple_obj, id_obj))
        c.pyapi.decref(id_obj)
    else:
        out = c.pyapi.call_function_objargs(RecordArray_obj, (length_obj, istuple_obj))
    append_obj = c.pyapi.object_getattr_string(out, "append")
    for i, t in enumerate(tpe.contenttpes):
        x_obj = c.pyapi.from_native_value(t, getattr(proxyin, field(i)), c.env_manager)
        if tpe.reverselookup is None or len(tpe.reverselookup) <= i:
            c.pyapi.call_function_objargs(append_obj, (x_obj,))
        else:
            key_obj = c.pyapi.unserialize(c.pyapi.serialize_object(tpe.reverselookup[i]))
            c.pyapi.call_function_objargs(append_obj, (x_obj, key_obj))
            c.pyapi.decref(key_obj)
        c.pyapi.decref(x_obj)
    c.pyapi.decref(RecordArray_obj)
    c.pyapi.decref(istuple_obj)
    c.pyapi.decref(length_obj)
    c.pyapi.decref(append_obj)
    return out

@numba.extending.box(RecordType)
def box(tpe, val, c):
    Record_obj = c.pyapi.unserialize(c.pyapi.serialize_object(awkward1.layout.Record))
    proxyin = numba.cgutils.create_struct_proxy(tpe)(c.context, c.builder, value=val)
    array_obj = c.pyapi.from_native_value(tpe.arraytpe, proxyin.array, c.env_manager)
    at_obj = c.pyapi.from_native_value(numba.int64, proxyin.at, c.env_manager)
    out = c.pyapi.call_function_objargs(Record_obj, (array_obj, at_obj))
    c.pyapi.decref(Record_obj)
    c.pyapi.decref(array_obj)
    c.pyapi.decref(at_obj)
    return out

@numba.extending.lower_builtin(len, RecordArrayType)
def lower_len(context, builder, sig, args):
    rettpe, (tpe,) = sig.return_type, sig.args
    val, = args
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
    return util.cast(context, builder, numba.int64, numba.intp, proxyin.length)

@numba.extending.lower_builtin(operator.getitem, RecordArrayType, numba.types.Integer)
def lower_getitem_int(context, builder, sig, args):
    rettpe, (tpe, wheretpe) = sig.return_type, sig.args
    val, whereval = args
    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)
    proxyout = numba.cgutils.create_struct_proxy(rettpe)(context, builder)
    proxyout.array = val
    proxyout.at = util.cast(context, builder, wheretpe, numba.int64, whereval)
    if context.enable_nrt:
        context.nrt.incref(builder, tpe, val)
    return numba.targets.imputils.impl_ret_new_ref(context, builder, rettpe, proxyout._getvalue())

@numba.extending.lower_builtin(operator.getitem, RecordArrayType, numba.types.slice2_type)
def lower_getitem_range(context, builder, sig, args):
    import awkward1._numba.identity

    rettpe, (tpe, wheretpe) = sig.return_type, sig.args
    val, whereval = args

    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)

    proxyslicein = context.make_helper(builder, wheretpe, value=whereval)
    numba.targets.slicing.guard_invalid_slice(context, builder, wheretpe, proxyslicein)
    numba.targets.slicing.fix_slice(builder, proxyslicein, util.cast(context, builder, numba.int64, numba.intp, proxyin.length))
    proxysliceout = numba.cgutils.create_struct_proxy(numba.types.slice2_type)(context, builder)
    proxysliceout.start = proxyslicein.start
    proxysliceout.stop = proxyslicein.stop
    proxysliceout.step = proxyslicein.step
    sliceout = proxysliceout._getvalue()

    proxyout = numba.cgutils.create_struct_proxy(tpe)(context, builder)
    proxyout.length = util.cast(context, builder, numba.intp, numba.int64, builder.sub(proxyslicein.stop, proxyslicein.start))
    for i, t in enumerate(tpe.contenttpes):
        setattr(proxyout, field(i), t.lower_getitem_range(context, builder, t.getitem_range()(t, numba.types.slice2_type), (getattr(proxyin, field(i)), sliceout)))
    if tpe.idtpe != numba.none:
        proxyout.id = awkward1._numba.identity.lower_getitem_any(context, builder, tpe.idtpe, wheretpe, proxyin.id, whereval)

    out = proxyout._getvalue()
    if context.enable_nrt:
        context.nrt.incref(builder, rettpe, out)
    return out

@numba.extending.lower_builtin(operator.getitem, RecordArrayType, numba.types.StringLiteral)
def lower_getitem_str(context, builder, sig, args):
    rettpe, (tpe, wheretpe) = sig.return_type, sig.args
    val, whereval = args
    index = awkward1.util.field2index(tpe.lookup, tpe.numfields, wheretpe.literal_value)

    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)

    out = getattr(proxyin, field(index))
    if context.enable_nrt:
        context.nrt.incref(builder, rettpe, out)
    return out

@numba.extending.lower_builtin(operator.getitem, RecordType, numba.types.StringLiteral)
def lower_getitem_str_record(context, builder, sig, args):
    rettpe, (tpe, wheretpe) = sig.return_type, sig.args
    val, whereval = args

    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)

    outtpe = tpe.arraytpe.getitem_str(wheretpe.literal_value)
    outval = lower_getitem_str(context, builder, outtpe(tpe.arraytpe, wheretpe), (proxyin.array, whereval))
    return outtpe.lower_getitem_int(context, builder, rettpe(outtpe, numba.int64), (outval, proxyin.at))

@numba.extending.lower_builtin(operator.getitem, RecordArrayType, numba.types.BaseTuple)
def lower_getitem_tuple(context, builder, sig, args):
    return content.lower_getitem_tuple(context, builder, sig, args)

@numba.extending.lower_builtin(operator.getitem, RecordType, numba.types.BaseTuple)
def lower_getitem_tuple_record(context, builder, sig, args):
    rettpe, (tpe, wheretpe) = sig.return_type, sig.args
    val, whereval = args

    proxyin = numba.cgutils.create_struct_proxy(tpe)(context, builder, value=val)

    nextwheretpe = numba.types.Tuple((numba.int64,) + wheretpe.types)
    nextwhereval = context.make_tuple(builder, nextwheretpe, (proxyin.at,) + numba.cgutils.unpack_tuple(builder, whereval))

    return lower_getitem_tuple(context, builder, rettpe(tpe.arraytpe, nextwheretpe), (proxyin.array, nextwhereval))

@numba.extending.lower_builtin(operator.getitem, RecordArrayType, numba.types.Array)
@numba.extending.lower_builtin(operator.getitem, RecordArrayType, numba.types.List)
@numba.extending.lower_builtin(operator.getitem, RecordArrayType, numba.types.ArrayCompatible)
@numba.extending.lower_builtin(operator.getitem, RecordArrayType, numba.types.EllipsisType)
@numba.extending.lower_builtin(operator.getitem, RecordArrayType, type(numba.typeof(numpy.newaxis)))
def lower_getitem_other(context, builder, sig, args):
    return content.lower_getitem_other(context, builder, sig, args)
