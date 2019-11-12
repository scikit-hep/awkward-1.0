# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import numbers

import numpy

import awkward1.layout

def typeof(array):
    if array is None:
        return awkward1.layout.UnknownType()

    elif isinstance(array, (bool, numpy.bool, numpy.bool_)):
        return awkward1.layout.PrimitiveType("bool")

    elif isinstance(array, numbers.Integral):
        return awkward1.layout.PrimitiveType("int64")

    elif isinstance(array, numbers.Real):
        return awkward1.layout.PrimitiveType("float64")

    elif isinstance(array, (numpy.int8, numpy.int16, numpy.int32, numpy.int64, numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64, numpy.float32, numpy.float64)):
        return awkward1.layout.PrimitiveType(typeof.dtype2primitive[array.dtype.type])

    elif isinstance(array, numpy.generic):
        raise ValueError("cannot describe {0} as a PrimitiveType".format(type(array)))

    elif isinstance(array, numpy.ndarray):
        if len(array.shape) == 0:
            return typeof(array.reshape((1,))[0])
        elif len(array.shape) == 1:
            return awkward1.layout.ArrayType(array.shape[0], awkward1.layout.PrimitiveType(typeof.dtype2primitive[array.dtype.type]))
        else:
            return awkward1.layout.ArrayType(array.shape[0], awkward1.layout.RegularType(array.shape[1:], awkward1.layout.PrimitiveType(typeof.dtype2primitive[array.dtype.type])))

    elif isinstance(array, awkward1.layout.FillableArray):
        return array.type

    elif isinstance(array, awkward1.layout.Content):
        return array.type

    else:
        raise TypeError("unrecognized array type: {0}".format(repr(array)))

typeof.dtype2primitive = {
    numpy.int8:    "int8",
    numpy.int16:   "int16",
    numpy.int32:   "int32",
    numpy.int64:   "int64",
    numpy.uint8:   "uint8",
    numpy.uint16:  "uint16",
    numpy.uint32:  "uint32",
    numpy.uint64:  "uint64",
    numpy.float32: "float32",
    numpy.float64: "float64",
}
