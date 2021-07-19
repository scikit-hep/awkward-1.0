# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import ctypes

import numpy

import awkward as ak


def of(*arrays):
    libs = set()
    for array in arrays:
        if isinstance(array, numpy.ndarray):
            ptr_lib = "cpu"
        elif (
            type(array).__module__.startswith("cupy.")
            and type(array).__name__ == "ndarray"
        ):
            ptr_lib = "cuda"
        else:
            ptr_lib = ak.operations.convert.kernels(array)
            if ptr_lib is None:
                pass
            elif ptr_lib == "cpu":
                libs.add("cpu")
            elif ptr_lib == "cuda":
                libs.add("cuda")
            else:
                raise ValueError(
                    """structure mixes 'cpu' and 'cuda' buffers; use one of

    ak.to_kernels(array, 'cpu')
    ak.to_kernels(array, 'cuda')

to obtain an unmixed array in main memory or the GPU(s)."""
                    + ak._util.exception_suffix(__file__)
                )

    if libs == set() or libs == set(["cpu"]):
        return NumPy.instance()
    elif libs == set(["cuda"]):
        return CuPy.instance()
    else:
        raise ValueError(
            """attempting to use both a 'cpu' array and a 'cuda' array in the """
            """same operation; use one of

    ak.to_kernels(array, 'cpu')
    ak.to_kernels(array, 'cuda')

to move one or the other to main memory or the GPU(s)."""
            + ak._util.exception_suffix(__file__)
        )


class Singleton(object):
    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


class NumPyMetadata(Singleton):
    bool_ = numpy.bool_
    int8 = numpy.int8
    int16 = numpy.int16
    int32 = numpy.int32
    int64 = numpy.int64
    uint8 = numpy.uint8
    uint16 = numpy.uint16
    uint32 = numpy.uint32
    uint64 = numpy.uint64
    float32 = numpy.float32
    float64 = numpy.float64
    complex64 = numpy.complex64
    complex128 = numpy.complex128
    str_ = numpy.str_
    bytes_ = numpy.bytes_

    intp = numpy.intp
    integer = numpy.integer
    floating = numpy.floating
    number = numpy.number
    object_ = numpy.object_
    generic = numpy.generic

    dtype = numpy.dtype
    ufunc = numpy.ufunc
    iinfo = numpy.iinfo
    errstate = numpy.errstate
    newaxis = numpy.newaxis

    ndarray = numpy.ndarray

    nan = numpy.nan
    inf = numpy.inf

    nat = numpy.datetime64("NaT")
    datetime_data = numpy.datetime_data
    issubdtype = numpy.issubdtype


if hasattr(numpy, "float16"):
    NumPyMetadata.float16 = numpy.float16

if hasattr(numpy, "float128"):
    NumPyMetadata.float128 = numpy.float128

if hasattr(numpy, "complex256"):
    NumPyMetadata.complex256 = numpy.complex256

if hasattr(numpy, "datetime64"):
    NumPyMetadata.datetime64 = numpy.datetime64

if hasattr(numpy, "timedelta64"):
    NumPyMetadata.timedelta64 = numpy.timedelta64


class NumPyLike(Singleton):
    ############################ array creation

    def array(self, *args, **kwargs):
        # data[, dtype=[, copy=]]
        return self._module.array(*args, **kwargs)

    def asarray(self, *args, **kwargs):
        # array[, dtype=]
        return self._module.asarray(*args, **kwargs)

    def ascontiguousarray(self, *args, **kwargs):
        # array[, dtype=]
        return self._module.ascontiguousarray(*args, **kwargs)

    def isscalar(self, *args, **kwargs):
        return self._module.isscalar(*args, **kwargs)

    def frombuffer(self, *args, **kwargs):
        # array[, dtype=]
        return self._module.frombuffer(*args, **kwargs)

    def zeros(self, *args, **kwargs):
        # shape/len[, dtype=]
        return self._module.zeros(*args, **kwargs)

    def ones(self, *args, **kwargs):
        # shape/len[, dtype=]
        return self._module.ones(*args, **kwargs)

    def empty(self, *args, **kwargs):
        # shape/len[, dtype=]
        return self._module.empty(*args, **kwargs)

    def full(self, *args, **kwargs):
        # shape/len, value[, dtype=]
        return self._module.full(*args, **kwargs)

    def full_like(self, *args, **kwargs):
        # array, fill_value
        return self._module.full_like(*args, **kwargs)

    def zeros_like(self, *args, **kwargs):
        # array
        return self._module.zeros_like(*args, **kwargs)

    def ones_like(self, *args, **kwargs):
        # array
        return self._module.ones_like(*args, **kwargs)

    def arange(self, *args, **kwargs):
        # stop[, dtype=]
        # start, stop[, dtype=]
        # start, stop, step[, dtype=]
        return self._module.arange(*args, **kwargs)

    def meshgrid(self, *args, **kwargs):
        # *arrays, indexing="ij"
        return self._module.meshgrid(*args, **kwargs)

    ############################ testing

    def array_equal(self, *args, **kwargs):
        # array1, array2
        return self._module.array_equal(*args, **kwargs)

    def size(self, *args, **kwargs):
        # array
        return self._module.size(*args, **kwargs)

    def searchsorted(self, *args, **kwargs):
        # haystack, needle, side="right"
        return self._module.searchsorted(*args, **kwargs)

    def argsort(self, *args, **kwargs):
        return self._module.argsort(*args, **kwargs)

    ############################ manipulation

    def add(self, *args, **kwargs):
        # array1, array2[, out=]
        return self._module.add(*args, **kwargs)

    def cumsum(self, *args, **kwargs):
        # arrays[, out=]
        return self._module.cumsum(*args, **kwargs)

    def cumprod(self, *args, **kwargs):
        # arrays[, out=]
        return self._module.cumprod(*args, **kwargs)

    def nonzero(self, *args, **kwargs):
        # array
        return self._module.nonzero(*args, **kwargs)

    def unique(self, *args, **kwargs):
        # array
        return self._module.unique(*args, **kwargs)

    def concatenate(self, *args, **kwargs):
        # arrays
        return self._module.concatenate(*args, **kwargs)

    def repeat(self, *args, **kwargs):
        # array, int
        # array1, array2
        return self._module.repeat(*args, **kwargs)

    def stack(self, *args, **kwargs):
        # arrays
        return self._module.stack(*args, **kwargs)

    def vstack(self, *args, **kwargs):
        # arrays
        return self._module.vstack(*args, **kwargs)

    def packbits(self, *args, **kwargs):
        # array
        return self._module.packbits(*args, **kwargs)

    def unpackbits(self, *args, **kwargs):
        # array
        return self._module.unpackbits(*args, **kwargs)

    def atleast_1d(self, *args, **kwargs):
        # *arrays
        return self._module.atleast_1d(*args, **kwargs)

    def broadcast_to(self, *args, **kwargs):
        # array, shape
        return self._module.broadcast_to(*args, **kwargs)

    def append(self, *args, **kwargs):
        # array, element
        return self._module.append(*args, **kwargs)

    ############################ ufuncs

    def sqrt(self, *args, **kwargs):
        # array
        return self._module.sqrt(*args, **kwargs)

    def exp(self, *args, **kwargs):
        # array
        return self._module.exp(*args, **kwargs)

    def true_divide(self, *args, **kwargs):
        # array1, array2
        return self._module.true_divide(*args, **kwargs)

    def bitwise_or(self, *args, **kwargs):
        # array1, array2[, out=output]
        return self._module.bitwise_or(*args, **kwargs)

    def logical_and(self, *args, **kwargs):
        # array1, array2
        return self._module.logical_and(*args, **kwargs)

    def equal(self, *args, **kwargs):
        # array1, array2
        return self._module.equal(*args, **kwargs)

    def ceil(self, *args, **kwargs):
        # array
        return self._module.ceil(*args, **kwargs)

    ############################ almost-ufuncs

    def nan_to_num(self, *args, **kwargs):
        # array, copy=True, nan=0.0, posinf=None, neginf=None
        return self._module.nan_to_num(*args, **kwargs)

    def isclose(self, *args, **kwargs):
        # a, b, rtol=1e-05, atol=1e-08, equal_nan=False
        return self._module.isclose(*args, **kwargs)

    ############################ reducers

    def all(self, *args, **kwargs):
        # array
        return self._module.all(*args, **kwargs)

    def any(self, *args, **kwargs):
        # array
        return self._module.any(*args, **kwargs)

    def count_nonzero(self, *args, **kwargs):
        # array
        return self._module.count_nonzero(*args, **kwargs)

    def sum(self, *args, **kwargs):
        # array
        return self._module.sum(*args, **kwargs)

    def prod(self, *args, **kwargs):
        # array
        return self._module.prod(*args, **kwargs)

    def min(self, *args, **kwargs):
        # array
        return self._module.min(*args, **kwargs)

    def max(self, *args, **kwargs):
        # array
        return self._module.max(*args, **kwargs)

    def argmin(self, *args, **kwargs):
        # array[, axis=]
        return self._module.argmin(*args, **kwargs)

    def argmax(self, *args, **kwargs):
        # array[, axis=]
        return self._module.argmax(*args, **kwargs)

    def array_str(self, *args, **kwargs):
        # array, max_line_width, precision=None, suppress_small=None
        return self._module.array_str(*args, **kwargs)

    def datetime_as_string(self, *args, **kwargs):
        return self._module.datetime_as_string(*args, **kwargs)


class NumPyKernel(object):
    def __init__(self, kernel):
        self._kernel = kernel

    def __call__(self, *args):
        assert len(args) == len(self._kernel.argtypes)
        return self._kernel(
            *(
                ctypes.cast(x, t) if issubclass(t, ctypes._Pointer) else x
                for x, t in zip(args, self._kernel.argtypes)
            )
        )


class NumPy(NumPyLike):
    def to_rectilinear(self, array, *args, **kwargs):
        return ak.operations.convert.to_numpy(array, *args, **kwargs)

    def __getitem__(self, args):
        return NumPyKernel(ak._cpu_kernels.kernel[args])

    def __init__(self):
        self._module = numpy

    @property
    def ma(self):
        return self._module.ma

    @property
    def char(self):
        return self._module.char

    @property
    def ndarray(self):
        return self._module.ndarray


class CuPy(NumPyLike):
    def to_rectilinear(self, array, *args, **kwargs):
        return ak.operations.convert.to_cupy(array, *args, **kwargs)

    def __getitem__(self, args):
        raise NotImplementedError("no CUDA in v2 yet")

    def __init__(self):
        try:
            import cupy
        except ImportError:
            raise ImportError(
                """to use CUDA arrays in Python, install the 'cupy' package with:

    pip install cupy --upgrade

or

    conda install cupy"""
            )
        self._module = cupy

    @property
    def ma(self):
        raise ValueError(
            "CUDA arrays cannot have missing values until CuPy implements "
            "numpy.ma.MaskedArray" + ak._util.exception_suffix(__file__)
        )

    @property
    def char(self):
        raise ValueError(
            "CUDA arrays cannot do string manipulations until CuPy implements "
            "numpy.char" + ak._util.exception_suffix(__file__)
        )

    @property
    def ndarray(self):
        return self._module.ndarray

    def asarray(self, array, dtype=None):
        if isinstance(
            array,
            (
                ak.highlevel.Array,
                ak.highlevel.Record,
                ak.layout.Content,
                ak.layout.Record,
            ),
        ):
            out = ak.operations.convert.to_cupy(array)
            if dtype is not None and out.dtype != dtype:
                return self._module.asarray(out, dtype=dtype)
            else:
                return out
        else:
            return self._module.asarray(array, dtype=dtype)

    def ascontiguousarray(self, array, dtype=None):
        if isinstance(
            array,
            (
                ak.highlevel.Array,
                ak.highlevel.Record,
                ak.layout.Content,
                ak.layout.Record,
            ),
        ):
            out = ak.operations.convert.to_cupy(array)
            if dtype is not None and out.dtype != dtype:
                return self._module.ascontiguousarray(out, dtype=dtype)
            else:
                return out
        else:
            return self._module.ascontiguousarray(array, dtype=dtype)

    def frombuffer(self, *args, **kwargs):
        np_array = numpy.frombuffer(*args, **kwargs)
        return self._module.array(np_array)

    def array_equal(self, array1, array2):
        # CuPy issue?
        if array1.shape != array2.shape:
            return False
        else:
            return self._module.all(array1 - array2 == 0)

    def repeat(self, array, repeats):
        # https://github.com/cupy/cupy/issues/3849
        if isinstance(repeats, self._module.ndarray):
            all_stops = self._module.cumsum(repeats)
            parents = self._module.zeros(all_stops[-1].item(), dtype=int)
            stops, stop_counts = self._module.unique(all_stops[:-1], return_counts=True)
            parents[stops] = stop_counts
            self._module.cumsum(parents, out=parents)
            return array[parents]
        else:
            return self._module.repeat(array, repeats)

    def nan_to_num(self, array, copy=True, nan=0.0, posinf=None, neginf=None):
        # https://github.com/cupy/cupy/issues/4867
        if copy:
            array = self._module.copy(array)
        if posinf is None:
            if array.dtype.kind == "f":
                posinf = numpy.finfo(array.dtype.type).max
            else:
                posinf = numpy.iinfo(array.dtype.type).max
        if neginf is None:
            if array.dtype.kind == "f":
                neginf = numpy.finfo(array.dtype.type).min
            else:
                neginf = numpy.iinfo(array.dtype.type).min

        array[self._module.isnan(array)] = nan
        array[self._module.isinf(array) & (array > 0)] = posinf
        array[self._module.isinf(array) & (array < 0)] = neginf
        return array

    # For all reducers: https://github.com/cupy/cupy/issues/3819

    def all(self, array, axis=None):
        out = self._module.all(array, axis=axis)
        if axis is None and isinstance(out, self._module.ndarray):
            return out.item()
        else:
            return out

    def any(self, array, axis=None):
        out = self._module.any(array, axis=axis)
        if axis is None and isinstance(out, self._module.ndarray):
            return out.item()
        else:
            return out

    def count_nonzero(self, array, axis=None):
        out = self._module.count_nonzero(array, axis=axis)
        if axis is None and isinstance(out, self._module.ndarray):
            return out.item()
        else:
            return out

    def sum(self, array, axis=None):
        out = self._module.sum(array, axis=axis)
        if axis is None and isinstance(out, self._module.ndarray):
            return out.item()
        else:
            return out

    def prod(self, array, axis=None):
        out = self._module.prod(array, axis=axis)
        if axis is None and isinstance(out, self._module.ndarray):
            return out.item()
        else:
            return out

    def min(self, array, axis=None):
        out = self._module.min(array, axis=axis)
        if axis is None and isinstance(out, self._module.ndarray):
            return out.item()
        else:
            return out

    def max(self, array, axis=None):
        out = self._module.max(array, axis=axis)
        if axis is None and isinstance(out, self._module.ndarray):
            return out.item()
        else:
            return out

    def argmin(self, array, axis=None):
        out = self._module.argmin(array, axis=axis)
        if axis is None and isinstance(out, self._module.ndarray):
            return out.item()
        else:
            return out

    def argmax(self, array, axis=None):
        out = self._module.argmax(array, axis=axis)
        if axis is None and isinstance(out, self._module.ndarray):
            return out.item()
        else:
            return out

    def array_str(
        self, array, max_line_width=None, precision=None, suppress_small=None
    ):
        # array, max_line_width, precision=None, suppress_small=None
        return self._module.array_str(array, max_line_width, precision, suppress_small)
