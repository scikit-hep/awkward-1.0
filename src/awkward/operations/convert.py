# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import numbers
import json
import collections
import math
import os
import threading
import distutils.version
import glob
import re

try:
    from collections.abc import Iterable, Sized
    from collections.abc import MutableMapping
except ImportError:
    from collections import Iterable, Sized
    from collections import MutableMapping

import awkward as ak

np = ak.nplike.NumPyMetadata.instance()
numpy = ak.nplike.NumPy.instance()


def _regularize_path(path):
    if isinstance(path, getattr(os, "PathLike", ())):
        path = os.fspath(path)

    elif hasattr(path, "__fspath__"):
        path = path.__fspath__()

    elif path.__class__.__module__ == "pathlib":
        import pathlib

        if isinstance(path, pathlib.Path):
            path = str(path)

    if isinstance(path, str):
        path = os.path.expanduser(path)

    return path


def from_numpy(
    array, regulararray=False, recordarray=True, highlevel=True, behavior=None
):
    """
    Args:
        array (np.ndarray): The NumPy array to convert into an Awkward Array.
            This array can be a np.ma.MaskedArray.
        regulararray (bool): If True and the array is multidimensional,
            the dimensions are represented by nested #ak.layout.RegularArray
            nodes; if False and the array is multidimensional, the dimensions
            are represented by a multivalued #ak.layout.NumpyArray.shape.
            If the array is one-dimensional, this has no effect.
        recordarray (bool): If True and the array is a NumPy structured array
            (dtype.names is not None), the fields are represented by an
            #ak.layout.RecordArray; if False and the array is a structured
            array, the structure is left in the #ak.layout.NumpyArray `format`,
            which some functions do not recognize.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Converts a NumPy array into an Awkward Array.

    The resulting layout may involve the following #ak.layout.Content types
    (only):

       * #ak.layout.NumpyArray
       * #ak.layout.ByteMaskedArray or #ak.layout.UnmaskedArray if the
         `array` is an np.ma.MaskedArray.
       * #ak.layout.RegularArray if `regulararray=True`.
       * #ak.layout.RecordArray if `recordarray=True`.

    See also #ak.to_numpy and #ak.from_cupy.
    """

    def recurse(array, mask):
        if regulararray and len(array.shape) > 1:
            return ak.layout.RegularArray(
                recurse(array.reshape((-1,) + array.shape[2:]), mask),
                array.shape[1],
                array.shape[0],
            )

        if len(array.shape) == 0:
            array = array.reshape(1)

        if array.dtype.kind == "S":
            asbytes = array.reshape(-1)
            itemsize = asbytes.dtype.itemsize
            starts = numpy.arange(0, len(asbytes) * itemsize, itemsize, dtype=np.int64)
            stops = starts + numpy.char.str_len(asbytes)
            data = ak.layout.ListArray64(
                ak.layout.Index64(starts),
                ak.layout.Index64(stops),
                ak.layout.NumpyArray(
                    asbytes.view("u1"), parameters={"__array__": "byte"}
                ),
                parameters={"__array__": "bytestring"},
            )
            for i in range(len(array.shape) - 1, 0, -1):
                data = ak.layout.RegularArray(data, array.shape[i], array.shape[i - 1])
        elif array.dtype.kind == "U":
            asbytes = numpy.char.encode(array.reshape(-1), "utf-8", "surrogateescape")
            itemsize = asbytes.dtype.itemsize
            starts = numpy.arange(0, len(asbytes) * itemsize, itemsize, dtype=np.int64)
            stops = starts + numpy.char.str_len(asbytes)
            data = ak.layout.ListArray64(
                ak.layout.Index64(starts),
                ak.layout.Index64(stops),
                ak.layout.NumpyArray(
                    asbytes.view("u1"), parameters={"__array__": "char"}
                ),
                parameters={"__array__": "string"},
            )
            for i in range(len(array.shape) - 1, 0, -1):
                data = ak.layout.RegularArray(data, array.shape[i], array.shape[i - 1])
        else:
            data = ak.layout.NumpyArray(array)

        if mask is None:
            return data
        elif mask is False or (isinstance(mask, np.bool_) and not mask):
            # NumPy's MaskedArray with mask == False is an UnmaskedArray
            if len(array.shape) == 1:
                return ak.layout.UnmaskedArray(data)
            else:

                def attach(x):
                    if isinstance(x, ak.layout.NumpyArray):
                        return ak.layout.UnmaskedArray(x)
                    else:
                        return ak.layout.RegularArray(attach(x.content), x.size, len(x))

                return attach(data.toRegularArray())
        else:
            # NumPy's MaskedArray is a ByteMaskedArray with valid_when=False
            return ak.layout.ByteMaskedArray(
                ak.layout.Index8(mask), data, valid_when=False
            )

    if isinstance(array, numpy.ma.MaskedArray):
        mask = numpy.ma.getmask(array)
        array = numpy.ma.getdata(array)
        if isinstance(mask, np.ndarray) and len(mask.shape) > 1:
            regulararray = True
            mask = mask.reshape(-1)
    else:
        mask = None

    if not recordarray or array.dtype.names is None:
        layout = recurse(array, mask)
    else:
        contents = []
        for name in array.dtype.names:
            contents.append(recurse(array[name], mask))
        layout = ak.layout.RecordArray(contents, array.dtype.names)

    return ak._util.maybe_wrap(layout, behavior, highlevel)


def to_numpy(array, allow_missing=True):
    """
    Converts `array` (many types supported, including all Awkward Arrays and
    Records) into a NumPy array, if possible.

    If the data are numerical and regular (nested lists have equal lengths
    in each dimension, as described by the #type), they can be losslessly
    converted to a NumPy array and this function returns without an error.

    Otherwise, the function raises an error. It does not create a NumPy
    array with dtype `"O"` for `np.object_` (see the
    [note on object_ type](https://docs.scipy.org/doc/numpy/reference/arrays.scalars.html#arrays-scalars-built-in))
    since silent conversions to dtype `"O"` arrays would not only be a
    significant performance hit, but would also break functionality, since
    nested lists in a NumPy `"O"` array are severed from the array and
    cannot be sliced as dimensions.

    If `array` is a scalar, it is converted into a NumPy scalar.

    If `allow_missing` is True; NumPy
    [masked arrays](https://docs.scipy.org/doc/numpy/reference/maskedarray.html)
    are a possible result; otherwise, missing values (None) cause this
    function to raise an error.

    See also #ak.from_numpy and #ak.to_cupy.
    """
    if isinstance(array, (bool, str, bytes, numbers.Number)):
        return numpy.array([array])[0]

    elif ak._util.py27 and isinstance(array, ak._util.unicode):
        return numpy.array([array])[0]

    elif isinstance(array, np.ndarray):
        return array

    elif isinstance(array, ak.highlevel.Array):
        return to_numpy(array.layout, allow_missing=allow_missing)

    elif isinstance(array, ak.highlevel.Record):
        out = array.layout
        return to_numpy(out.array[out.at : out.at + 1], allow_missing=allow_missing)[0]

    elif isinstance(array, ak.highlevel.ArrayBuilder):
        return to_numpy(array.snapshot().layout, allow_missing=allow_missing)

    elif isinstance(array, ak.layout.ArrayBuilder):
        return to_numpy(array.snapshot(), allow_missing=allow_missing)

    elif ak.operations.describe.parameters(array).get("__array__") == "bytestring":
        return numpy.array(
            [
                ak.behaviors.string.ByteBehavior(array[i]).__bytes__()
                for i in range(len(array))
            ]
        )

    elif ak.operations.describe.parameters(array).get("__array__") == "string":
        return numpy.array(
            [
                ak.behaviors.string.CharBehavior(array[i]).__str__()
                for i in range(len(array))
            ]
        )

    elif (
        str(ak.operations.describe.type(array)) == "datetime64"
        or str(ak.operations.describe.type(array)) == "timedelta64"
    ):
        return array

    elif isinstance(array, ak.partition.PartitionedArray):
        tocat = [to_numpy(x, allow_missing=allow_missing) for x in array.partitions]
        if any(isinstance(x, numpy.ma.MaskedArray) for x in tocat):
            return numpy.ma.concatenate(tocat)
        else:
            return numpy.concatenate(tocat)

    elif isinstance(array, ak._util.virtualtypes):
        return to_numpy(array.array, allow_missing=True)

    elif isinstance(array, ak._util.unknowntypes):
        return numpy.array([])

    elif isinstance(array, ak._util.indexedtypes):
        return to_numpy(array.project(), allow_missing=allow_missing)

    elif isinstance(array, ak._util.uniontypes):
        contents = [
            to_numpy(array.project(i), allow_missing=allow_missing)
            for i in range(array.numcontents)
        ]

        if any(isinstance(x, numpy.ma.MaskedArray) for x in contents):
            try:
                out = numpy.ma.concatenate(contents)
            except Exception:
                raise ValueError(
                    "cannot convert {0} into numpy.ma.MaskedArray".format(array)
                    + ak._util.exception_suffix(__file__)
                )
        else:
            try:
                out = numpy.concatenate(contents)
            except Exception:
                raise ValueError(
                    "cannot convert {0} into np.ndarray".format(array)
                    + ak._util.exception_suffix(__file__)
                )

        tags = numpy.asarray(array.tags)
        for tag, content in enumerate(contents):
            mask = tags == tag
            out[mask] = content
        return out

    elif isinstance(array, ak.layout.UnmaskedArray):
        content = to_numpy(array.content, allow_missing=allow_missing)
        if allow_missing:
            return numpy.ma.MaskedArray(content)
        else:
            return content

    elif isinstance(array, ak._util.optiontypes):
        content = to_numpy(array.project(), allow_missing=allow_missing)

        shape = list(content.shape)
        shape[0] = len(array)
        data = numpy.empty(shape, dtype=content.dtype)
        mask0 = numpy.asarray(array.bytemask()).view(np.bool_)
        if mask0.any():
            if allow_missing:
                mask = numpy.broadcast_to(
                    mask0.reshape((shape[0],) + (1,) * (len(shape) - 1)), shape
                )
                if isinstance(content, numpy.ma.MaskedArray):
                    mask1 = numpy.ma.getmaskarray(content)
                    mask = mask.copy()
                    mask[~mask0] |= mask1

                data[~mask0] = content
                return numpy.ma.MaskedArray(data, mask)
            else:
                raise ValueError(
                    "ak.to_numpy cannot convert 'None' values to "
                    "np.ma.MaskedArray unless the "
                    "'allow_missing' parameter is set to True"
                    + ak._util.exception_suffix(__file__)
                )
        else:
            if allow_missing:
                return numpy.ma.MaskedArray(content)
            else:
                return content

    elif isinstance(array, ak.layout.RegularArray):
        out = to_numpy(array.content, allow_missing=allow_missing)
        head, tail = out.shape[0], out.shape[1:]
        if array.size == 0:
            shape = (0, 0) + tail
        else:
            shape = (head // array.size, array.size) + tail
        return out[: shape[0] * array.size].reshape(shape)

    elif isinstance(array, ak._util.listtypes):
        return to_numpy(array.toRegularArray(), allow_missing=allow_missing)

    elif isinstance(array, ak._util.recordtypes):
        if array.numfields == 0:
            return numpy.empty(len(array), dtype=[])
        contents = [
            to_numpy(array.field(i), allow_missing=allow_missing)
            for i in range(array.numfields)
        ]
        if any(len(x.shape) != 1 for x in contents):
            raise ValueError(
                "cannot convert {0} into np.ndarray".format(array)
                + ak._util.exception_suffix(__file__)
            )
        out = numpy.empty(
            len(contents[0]),
            dtype=[(str(n), x.dtype) for n, x in zip(array.keys(), contents)],
        )
        for n, x in zip(array.keys(), contents):
            out[n] = x
        return out

    elif isinstance(array, ak.layout.NumpyArray):
        out = ak.nplike.of(array).asarray(array)
        if type(out).__module__.startswith("cupy."):
            return out.get()
        else:
            return out

    elif isinstance(array, ak.layout.Content):
        raise AssertionError(
            "unrecognized Content type: {0}".format(type(array))
            + ak._util.exception_suffix(__file__)
        )

    elif isinstance(array, Iterable):
        return numpy.asarray(array)

    else:
        raise ValueError(
            "cannot convert {0} into np.ndarray".format(array)
            + ak._util.exception_suffix(__file__)
        )


def from_cupy(array, regulararray=False, highlevel=True, behavior=None):
    """
    Args:
        array (cp.ndarray): The CuPy array to convert into an Awkward Array.
        regulararray (bool): If True and the array is multidimensional,
            the dimensions are represented by nested #ak.layout.RegularArray
            nodes; if False and the array is multidimensional, the dimensions
            are represented by a multivalued #ak.layout.NumpyArray.shape.
            If the array is one-dimensional, this has no effect.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Converts a CuPy array into an Awkward Array.

    The resulting layout may involve the following #ak.layout.Content types
    (only):

       * #ak.layout.NumpyArray
       * #ak.layout.RegularArray if `regulararray=True`.

    See also #ak.to_cupy, #ak.from_numpy and #ak.from_jax.
    """

    def recurse(array):
        if regulararray and len(array.shape) > 1:
            return ak.layout.RegularArray(
                recurse(array.reshape((-1,) + array.shape[2:])),
                array.shape[1],
                array.shape[0],
            )

        if len(array.shape) == 0:
            data = ak.layout.NumpyArray.from_cupy(array.reshape(1))
        else:
            data = ak.layout.NumpyArray.from_cupy(array)

        return data

    layout = recurse(array)

    return ak._util.maybe_wrap(layout, behavior, highlevel)


def to_cupy(array):
    """
    Converts `array` (many types supported) into a CuPy array, if possible.

    If the data are numerical and regular (nested lists have equal lengths
    in each dimension, as described by the #type), they can be losslessly
    converted to a CuPy array and this function returns without an error.

    Otherwise, the function raises an error.

    If `array` is a scalar, it is converted into a CuPy scalar.

    See also #ak.from_cupy and #ak.to_numpy.
    """
    cupy = ak.nplike.CuPy.instance()
    np = ak.nplike.NumPyMetadata.instance()

    if isinstance(array, (bool, numbers.Number)):
        return cupy.array([array])[0]

    elif isinstance(array, cupy.ndarray):
        return array

    elif isinstance(array, np.ndarray):
        return cupy.asarray(array)

    elif isinstance(array, ak.highlevel.Array):
        return to_cupy(array.layout)

    elif isinstance(array, ak.highlevel.Record):
        raise ValueError(
            "CuPy does not support record structures"
            + ak._util.exception_suffix(__file__)
        )

    elif isinstance(array, ak.highlevel.ArrayBuilder):
        return to_cupy(array.snapshot().layout)

    elif isinstance(array, ak.layout.ArrayBuilder):
        return to_cupy(array.snapshot())

    elif (
        ak.operations.describe.parameters(array).get("__array__") == "bytestring"
        or ak.operations.describe.parameters(array).get("__array__") == "string"
    ):
        raise ValueError(
            "CuPy does not support arrays of strings"
            + ak._util.exception_suffix(__file__)
        )

    elif isinstance(array, ak.partition.PartitionedArray):
        return cupy.concatenate([to_cupy(x) for x in array.partitions])

    elif isinstance(array, ak._util.virtualtypes):
        return to_cupy(array.array)

    elif isinstance(array, ak._util.unknowntypes):
        return cupy.array([])

    elif isinstance(array, ak._util.indexedtypes):
        return to_cupy(array.project())

    elif isinstance(array, ak._util.uniontypes):
        contents = [to_cupy(array.project(i)) for i in range(array.numcontents)]
        out = cupy.concatenate(contents)

        tags = cupy.asarray(array.tags)
        for tag, content in enumerate(contents):
            mask = tags == tag
            out[mask] = content
        return out

    elif isinstance(array, ak.layout.UnmaskedArray):
        return to_cupy(array.content)

    elif isinstance(array, ak._util.optiontypes):
        content = to_cupy(array.project())

        shape = list(content.shape)
        shape[0] = len(array)
        mask0 = cupy.asarray(array.bytemask()).view(np.bool_)
        if mask0.any():
            raise ValueError(
                "CuPy does not support masked arrays"
                + ak._util.exception_suffix(__file__)
            )
        else:
            return content

    elif isinstance(array, ak.layout.RegularArray):
        out = to_cupy(array.content)
        head, tail = out.shape[0], out.shape[1:]
        shape = (head // array.size, array.size) + tail
        return out[: shape[0] * array.size].reshape(shape)

    elif isinstance(array, ak._util.listtypes):
        return to_cupy(array.toRegularArray())

    elif isinstance(array, ak._util.recordtypes):
        raise ValueError(
            "CuPy does not support record structures"
            + ak._util.exception_suffix(__file__)
        )

    elif isinstance(array, ak.layout.NumpyArray):
        return array.to_cupy()

    elif isinstance(array, ak.layout.Content):
        raise AssertionError(
            "unrecognized Content type: {0}".format(type(array))
            + ak._util.exception_suffix(__file__)
        )

    elif isinstance(array, Iterable):
        return cupy.asarray(array)

    else:
        raise ValueError(
            "cannot convert {0} into cp.ndarray".format(array)
            + ak._util.exception_suffix(__file__)
        )


def from_jax(array, regulararray=False, highlevel=True, behavior=None):
    """
    Args:
        array (jax.numpy.array): The `jax.numpy.array` to convert into an Awkward Array.
        regulararray (bool): If True and the array is multidimensional,
            the dimensions are represented by nested #ak.layout.RegularArray
            nodes; if False and the array is multidimensional, the dimensions
            are represented by a multivalued #ak.layout.NumpyArray.shape.
            If the array is one-dimensional, this has no effect.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Converts a JAX array into an Awkward Array.

    The resulting layout may involve the following #ak.layout.Content types
    (only):

       * #ak.layout.NumpyArray
       * #ak.layout.RegularArray if `regulararray=True`.

    See also #ak.from_cupy and #ak.from_numpy.
    """

    def recurse(array):
        if regulararray and len(array.shape) > 1:
            return ak.layout.RegularArray(
                recurse(array.reshape((-1,) + array.shape[2:])),
                array.shape[1],
                array.shape[0],
            )

        if len(array.shape) == 0:
            data = ak.layout.NumpyArray.from_jax(array.reshape(1))
        else:
            data = ak.layout.NumpyArray.from_jax(array)

        return data

    layout = recurse(array)

    return ak._util.maybe_wrap(layout, behavior, highlevel)


def to_jax(array):
    """
    Converts `array` (many types supported) into a JAX array, if possible.

    If the data are numerical and regular (nested lists have equal lengths
    in each dimension, as described by the #type), they can be losslessly
    converted to a CuPy array and this function returns without an error.

    Otherwise, the function raises an error.

    If `array` is a scalar, it is converted into a JAX scalar.

    See also #ak.to_cupy, #ak.from_jax and #ak.to_numpy.
    """
    try:
        import jax
    except ImportError:
        raise ImportError(
            """to use {0}, you must install jax:

                pip install jax jaxlib
            """
        )

    if isinstance(array, (bool, numbers.Number)):
        return jax.numpy.array([array])[0]

    elif isinstance(array, jax.numpy.ndarray):
        return array

    elif isinstance(array, np.ndarray):
        return jax.numpy.asarray(array)

    elif isinstance(array, ak.highlevel.Array):
        return to_jax(array.layout)

    elif isinstance(array, ak.highlevel.Record):
        raise ValueError(
            "JAX does not support record structures"
            + ak._util.exception_suffix(__file__)
        )

    elif isinstance(array, ak.highlevel.ArrayBuilder):
        return to_jax(array.snapshot().layout)

    elif isinstance(array, ak.layout.ArrayBuilder):
        return to_jax(array.snapshot())

    elif (
        ak.operations.describe.parameters(array).get("__array__") == "bytestring"
        or ak.operations.describe.parameters(array).get("__array__") == "string"
    ):
        raise ValueError(
            "JAX does not support arrays of strings"
            + ak._util.exception_suffix(__file__)
        )

    elif isinstance(array, ak.partition.PartitionedArray):
        return jax.numpy.concatenate([to_jax(x) for x in array.partitions])

    elif isinstance(array, ak._util.virtualtypes):
        return to_jax(array.array)

    elif isinstance(array, ak._util.unknowntypes):
        return jax.numpy.array([])

    elif isinstance(array, ak._util.indexedtypes):
        return to_jax(array.project())

    elif isinstance(array, ak._util.uniontypes):
        array = array.simplify()
        if isinstance(array, ak._util.uniontypes):
            raise ValueError(
                "cannot convert {0} into jax.numpy.array".format(array)
                + ak._util.exception_suffix(__file__)
            )
        return to_jax(array)

    elif isinstance(array, ak.layout.UnmaskedArray):
        return to_jax(array.content)

    elif isinstance(array, ak._util.optiontypes):
        content = to_jax(array.project())

        shape = list(content.shape)
        shape[0] = len(array)
        mask0 = jax.numpy.asarray(array.bytemask()).view(np.bool_)
        if mask0.any():
            raise ValueError(
                "JAX does not support masked arrays"
                + ak._util.exception_suffix(__file__)
            )
        else:
            return content

    elif isinstance(array, ak.layout.RegularArray):
        out = to_jax(array.content)
        head, tail = out.shape[0], out.shape[1:]
        shape = (head // array.size, array.size) + tail
        return out[: shape[0] * array.size].reshape(shape)

    elif isinstance(array, ak._util.listtypes):
        return to_jax(array.toRegularArray())

    elif isinstance(array, ak._util.recordtypes):
        raise ValueError(
            "JAX does not support record structures"
            + ak._util.exception_suffix(__file__)
        )

    elif isinstance(array, ak.layout.NumpyArray):
        return array.to_jax()

    elif isinstance(array, ak.layout.Content):
        raise AssertionError(
            "unrecognized Content type: {0}".format(type(array))
            + ak._util.exception_suffix(__file__)
        )

    elif isinstance(array, Iterable):
        return jax.numpy.asarray(array)

    else:
        raise ValueError(
            "cannot convert {0} into jax.numpy.array".format(array)
            + ak._util.exception_suffix(__file__)
        )


def kernels(*arrays):
    """
    Returns the names of the kernels library used by `arrays`. May be

       * `"cpu"` for `libawkward-cpu-kernels.so`;
       * `"cuda"` for `libawkward-cuda-kernels.so`;
       * `"mixed"` if any of the arrays have different labels within their
         structure or any arrays have different labels from each other;
       * None if the objects are not Awkward, NumPy, or CuPy arrays (e.g.
         Python numbers, booleans, strings).

    Mixed arrays can't be used in any operations, and two arrays on different
    devices can't be used in the same operation.

    To use `"cuda"`, the package
    [awkward-cuda-kernels](https://pypi.org/project/awkward-cuda-kernels)
    be installed, either by

        pip install awkward-cuda-kernels

    or as an optional dependency with

        pip install awkward[cuda] --upgrade

    It is only available for Linux as a binary wheel, and only supports Nvidia
    GPUs (it is written in CUDA).

    See #ak.to_kernels.
    """
    libs = set()
    for array in arrays:
        layout = ak.operations.convert.to_layout(
            array,
            allow_record=True,
            allow_other=True,
        )

        if isinstance(
            layout, (ak.layout.Content, ak.layout.Record, ak.partition.PartitionedArray)
        ):
            libs.add(layout.kernels)

        elif isinstance(layout, ak.nplike.numpy.ndarray):
            libs.add("cpu")

        elif type(layout).__module__.startswith("cupy."):
            libs.add("cuda")

    if libs == set():
        return None
    elif libs == set(["cpu"]):
        return "cpu"
    elif libs == set(["cuda"]):
        return "cuda"
    else:
        return "mixed"


def to_kernels(array, kernels, highlevel=True, behavior=None):
    """
    Args:
        array: Data to convert to a specified `kernels` set.
        kernels (`"cpu"` or `"cuda"`): If `"cpu"`, the array structure is
            recursively copied (if need be) to main memory for use with
            the default `libawkward-cpu-kernels.so`; if `"cuda"`, the
            structure is copied to the GPU(s) for use with
            `libawkward-cuda-kernels.so`.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Converts an array from `"cpu"`, `"cuda"`, or `"mixed"` kernels to `"cpu"`
    or `"cuda"`.

    An array is `"mixed"` if some components are set to use `"cpu"` kernels and
    others are set to use `"cuda"` kernels. Mixed arrays can't be used in any
    operations, and two arrays set to different kernels can't be used in the
    same operation.

    Any components that are already in the desired kernels library are viewed,
    rather than copied, so this operation can be an inexpensive way to ensure
    that an array is ready for a particular library.

    To use `"cuda"`, the package
    [awkward-cuda-kernels](https://pypi.org/project/awkward-cuda-kernels)
    be installed, either by

        pip install awkward-cuda-kernels

    or as an optional dependency with

        pip install awkward[cuda] --upgrade

    It is only available for Linux as a binary wheel, and only supports Nvidia
    GPUs (it is written in CUDA).

    See #ak.kernels.
    """
    arr = ak.to_layout(array)
    out = arr.copy_to(kernels)

    return ak._util.maybe_wrap_like(out, array, behavior, highlevel)


def from_iter(
    iterable, highlevel=True, behavior=None, allow_record=True, initial=1024, resize=1.5
):
    """
    Args:
        iterable (Python iterable): Data to convert into an Awkward Array.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        allow_record (bool): If True, the outermost element may be a record
            (returning #ak.Record or #ak.layout.Record type, depending on
            `highlevel`); if False, the outermost element must be an array.
        initial (int): Initial size (in bytes) of buffers used by
            #ak.layout.ArrayBuilder (see #ak.layout.ArrayBuilderOptions).
        resize (float): Resize multiplier for buffers used by
            #ak.layout.ArrayBuilder (see #ak.layout.ArrayBuilderOptions);
            should be strictly greater than 1.

    Converts Python data into an Awkward Array.

    Internally, this function uses #ak.layout.ArrayBuilder (see the high-level
    #ak.ArrayBuilder documentation for a more complete description), so it
    has the same flexibility and the same constraints. Any heterogeneous
    and deeply nested Python data can be converted, but the output will never
    have regular-typed array lengths.

    The following Python types are supported.

       * bool, including `np.bool_`: converted into #ak.layout.NumpyArray.
       * int, including `np.integer`: converted into #ak.layout.NumpyArray.
       * float, including `np.floating`: converted into #ak.layout.NumpyArray.
       * bytes: converted into #ak.layout.ListOffsetArray with parameter
         `"__array__"` equal to `"bytestring"` (unencoded bytes).
       * str: converted into #ak.layout.ListOffsetArray with parameter
         `"__array__"` equal to `"string"` (UTF-8 encoded string).
       * tuple: converted into #ak.layout.RecordArray without field names
         (i.e. homogeneously typed, uniform sized tuples).
       * dict: converted into #ak.layout.RecordArray with field names
         (i.e. homogeneously typed records with the same sets of fields).
       * iterable, including np.ndarray: converted into
         #ak.layout.ListOffsetArray.

    See also #ak.to_list.
    """
    if isinstance(iterable, dict):
        if allow_record:
            return from_iter(
                [iterable],
                highlevel=highlevel,
                behavior=behavior,
                initial=initial,
                resize=resize,
            )[0]
        else:
            raise ValueError(
                "cannot produce an array from a dict"
                + ak._util.exception_suffix(__file__)
            )

    out = ak.layout.ArrayBuilder(initial=initial, resize=resize)
    for x in iterable:
        out.fromiter(x)
    layout = out.snapshot()
    return ak._util.maybe_wrap(layout, behavior, highlevel)


def to_list(array):
    """
    Converts `array` (many types supported, including all Awkward Arrays and
    Records) into Python objects.

    Awkward Array types have the following Pythonic translations.

       * #ak.types.PrimitiveType: converted into bool, int, float.
       * #ak.types.OptionType: missing values are converted into None.
       * #ak.types.ListType: converted into list.
       * #ak.types.RegularType: also converted into list. Python (and JSON)
         forms lose information about the regularity of list lengths.
       * #ak.types.ListType with parameter `"__array__"` equal to
         `"__bytestring__"`: converted into bytes.
       * #ak.types.ListType with parameter `"__array__"` equal to
         `"__string__"`: converted into str.
       * #ak.types.RecordArray without field names: converted into tuple.
       * #ak.types.RecordArray with field names: converted into dict.
       * #ak.types.UnionArray: Python data are naturally heterogeneous.

    See also #ak.from_iter and #ak.Array.tolist.
    """
    if array is None or isinstance(array, (bool, str, bytes, numbers.Number)):
        return array

    elif ak._util.py27 and isinstance(array, ak._util.unicode):
        return array

    elif isinstance(array, np.ndarray):
        return array.tolist()

    elif isinstance(array, ak.behaviors.string.ByteBehavior):
        return array.__bytes__()

    elif isinstance(array, ak.behaviors.string.CharBehavior):
        return array.__str__()

    elif ak.operations.describe.parameters(array).get("__array__") == "byte":
        return ak.behaviors.string.CharBehavior(array).__bytes__()

    elif ak.operations.describe.parameters(array).get("__array__") == "char":
        return ak.behaviors.string.CharBehavior(array).__str__()

    elif isinstance(array, np.datetime64) or isinstance(array, np.timedelta64):
        return array

    elif isinstance(array, ak.highlevel.Array):
        return [to_list(x) for x in array]

    elif isinstance(array, ak.highlevel.Record):
        return to_list(array.layout)

    elif isinstance(array, ak.highlevel.ArrayBuilder):
        return to_list(array.snapshot())

    elif isinstance(array, ak.layout.Record) and array.istuple:
        return tuple(to_list(x) for x in array.fields())

    elif isinstance(array, ak.layout.Record):
        return {n: to_list(x) for n, x in array.fielditems()}

    elif isinstance(array, ak.layout.ArrayBuilder):
        return [to_list(x) for x in array.snapshot()]

    elif isinstance(array, ak.layout.NumpyArray):
        if array.format.upper().startswith("M"):
            return (
                [
                    x
                    for x in ak.nplike.of(array)
                    .asarray(array.view_int64)
                    .view(array.format)
                ]
                # FIXME: .tolist() returns
                # [[1567416600000000000], [1568367000000000000], [1569096000000000000]]
                # instead of [numpy.datetime64('2019-09-02T09:30:00'), numpy.datetime64('2019-09-13T09:30:00'), numpy.datetime64('2019-09-21T20:00:00')]
                # see test_from_pandas() test
            )
        else:
            return ak.nplike.of(array).asarray(array).tolist()

    elif isinstance(array, (ak.layout.Content, ak.partition.PartitionedArray)):
        return [to_list(x) for x in array]

    elif isinstance(array, ak._v2.contents.Content):
        import awkward._v2.tmp_for_testing

        return to_list(awkward._v2.tmp_for_testing.v2_to_v1(array))

    elif isinstance(array, dict):
        return dict((n, to_list(x)) for n, x in array.items())

    elif isinstance(array, Iterable):
        return [to_list(x) for x in array]

    else:
        raise TypeError(
            "unrecognized array type: {0}".format(type(array))
            + ak._util.exception_suffix(__file__)
        )


_maybe_json_str = re.compile(r"^\s*(\[|\{|\"|[0-9]|true|false|null)")
_maybe_json_bytes = re.compile(br"^\s*(\[|\{|\"|[0-9]|true|false|null)")


def from_json(
    source,
    nan_string=None,
    infinity_string=None,
    minus_infinity_string=None,
    complex_record_fields=None,
    highlevel=True,
    behavior=None,
    initial=1024,
    resize=1.5,
    buffersize=65536,
):
    """
    Args:
        source (str): JSON-formatted string or filename to convert into an array.
        nan_string (None or str): If not None, strings with this value will be
            interpreted as floating-point NaN values.
        infinity_string (None or str): If not None, strings with this value will
            be interpreted as floating-point positive infinity values.
        minus_infinity_string (None or str): If not None, strings with this value
            will be interpreted as floating-point negative infinity values.
        complex_record_fields (None or (str, str)): If not None, defines a pair of
            field names to interpret records as complex numbers.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        initial (int): Initial size (in bytes) of buffers used by
            #ak.layout.ArrayBuilder (see #ak.layout.ArrayBuilderOptions).
        resize (float): Resize multiplier for buffers used by
            #ak.layout.ArrayBuilder (see #ak.layout.ArrayBuilderOptions);
            should be strictly greater than 1.
        buffersize (int): Size (in bytes) of the buffer used by the JSON
            parser.

    Converts a JSON string into an Awkward Array.

    Internally, this function uses #ak.layout.ArrayBuilder (see the high-level
    #ak.ArrayBuilder documentation for a more complete description), so it
    has the same flexibility and the same constraints. Any heterogeneous
    and deeply nested JSON can be converted, but the output will never have
    regular-typed array lengths.

    See also #ak.to_json.
    """

    if complex_record_fields is None:
        complex_real_string = None
        complex_imag_string = None
    elif (
        isinstance(complex_record_fields, tuple)
        and len(complex_record_fields) == 2
        and isinstance(complex_record_fields[0], str)
        and isinstance(complex_record_fields[1], str)
    ):
        complex_real_string, complex_imag_string = complex_record_fields

    is_path, source = ak._util.regularize_path(source)

    if os.path.isfile(source):
        layout = ak._ext.fromjsonfile(
            source,
            nan_string=nan_string,
            infinity_string=infinity_string,
            minus_infinity_string=minus_infinity_string,
            initial=initial,
            resize=resize,
            buffersize=buffersize,
        )
    elif not is_path and (
        (isinstance(source, bytes) and _maybe_json_bytes.match(source))
        or _maybe_json_str.match(source)
    ):
        layout = ak._ext.fromjson(
            source,
            nan_string=nan_string,
            infinity_string=infinity_string,
            minus_infinity_string=minus_infinity_string,
            initial=initial,
            resize=resize,
            buffersize=buffersize,
        )
    else:
        if ak._util.py27:
            exc = IOError
        else:
            exc = FileNotFoundError
        raise exc("file not found or not a regular file: {0}".format(source))

    def getfunction(recordnode):
        if isinstance(recordnode, ak.layout.RecordArray):
            keys = recordnode.keys()
            if complex_record_fields[0] in keys and complex_record_fields[1] in keys:
                nplike = ak.nplike.of(recordnode)
                real = recordnode[complex_record_fields[0]]
                imag = recordnode[complex_record_fields[1]]
                if (
                    isinstance(real, ak.layout.NumpyArray)
                    and len(real.shape) == 1
                    and isinstance(imag, ak.layout.NumpyArray)
                    and len(imag.shape) == 1
                ):
                    return lambda: nplike.asarray(real) + nplike.asarray(imag) * 1j
                else:
                    raise ValueError(
                        "Complex number fields must be numbers"
                        + ak._util.exception_suffix(__file__)
                    )
                return lambda: ak.layout.NumpyArray(real + imag * 1j)
            else:
                return None
        else:
            return None

    if complex_imag_string is not None:
        layout = ak._util.recursively_apply(layout, getfunction, pass_depth=False)

    return ak._util.maybe_wrap(layout, behavior, highlevel)


def to_json(
    array,
    destination=None,
    pretty=False,
    maxdecimals=None,
    nan_string=None,
    infinity_string=None,
    minus_infinity_string=None,
    complex_record_fields=None,
    buffersize=65536,
):
    """
    Args:
        array: Data to convert to JSON.
        destination (None or str): If None, this function returns a JSON str;
            if a str, it uses that as a file name and writes (overwrites) that
            file (returning None).
        pretty (bool): If True, indent the output for human readability; if
            False, output compact JSON without spaces.
        maxdecimals (None or int): If an int, limit the number of
            floating-point decimals to this number; if None, write all digits.
        nan_string (None or str): If not None, floating-point NaN values will be
            replaced with this string instead of a JSON number.
        infinity_string (None or str): If not None, floating-point positive infinity
            values will be replaced with this string instead of a JSON number.
        minus_infinity_string (None or str): If not None, floating-point negative
            infinity values will be replaced with this string instead of a JSON
            number.
        complex_record_fields (None or (str, str)): If not None, defines a pair of
            field names to interpret records as complex numbers.
        buffersize (int): Size (in bytes) of the buffer used by the JSON
            parser.

    Converts `array` (many types supported, including all Awkward Arrays and
    Records) into a JSON string or file.

    Awkward Array types have the following JSON translations.

       * #ak.types.PrimitiveType: converted into JSON booleans and numbers.
       * #ak.types.OptionType: missing values are converted into None.
       * #ak.types.ListType: converted into JSON lists.
       * #ak.types.RegularType: also converted into JSON lists. JSON (and
         Python) forms lose information about the regularity of list lengths.
       * #ak.types.ListType with parameter `"__array__"` equal to
         `"__bytestring__"` or `"__string__"`: converted into JSON strings.
       * #ak.types.RecordArray without field names: converted into JSON
         objects with numbers as strings for keys.
       * #ak.types.RecordArray with field names: converted into JSON objects.
       * #ak.types.UnionArray: JSON data are naturally heterogeneous.

    See also #ak.from_json and #ak.Array.tojson.
    """
    if array is None or isinstance(array, (bool, str, bytes, numbers.Number)):
        return json.dumps(array)

    elif isinstance(array, bytes):
        return json.dumps(array.decode("utf-8", "surrogateescape"))

    elif ak._util.py27 and isinstance(array, ak._util.unicode):
        return json.dumps(array)

    elif isinstance(array, np.ndarray):
        out = ak.layout.NumpyArray(array)

    elif isinstance(array, ak.highlevel.Array):
        out = array.layout

    elif isinstance(array, ak.highlevel.Record):
        out = array.layout

    elif isinstance(array, ak.highlevel.ArrayBuilder):
        out = array.snapshot().layout

    elif isinstance(array, ak.layout.Record):
        out = array

    elif isinstance(array, ak.layout.ArrayBuilder):
        out = array.snapshot()

    elif isinstance(array, (ak.layout.Content, ak.partition.PartitionedArray)):
        out = array

    else:
        raise TypeError(
            "unrecognized array type: {0}".format(repr(array))
            + ak._util.exception_suffix(__file__)
        )

    if complex_record_fields is None:
        complex_real_string = None
        complex_imag_string = None
    elif (
        isinstance(complex_record_fields, tuple)
        and len(complex_record_fields) == 2
        and isinstance(complex_record_fields[0], str)
        and isinstance(complex_record_fields[1], str)
    ):
        complex_real_string, complex_imag_string = complex_record_fields

    if destination is None:
        return out.tojson(
            pretty=pretty,
            maxdecimals=maxdecimals,
            nan_string=nan_string,
            infinity_string=infinity_string,
            minus_infinity_string=minus_infinity_string,
            complex_real_string=complex_real_string,
            complex_imag_string=complex_imag_string,
        )
    else:
        return out.tojson(
            destination,
            pretty=pretty,
            maxdecimals=maxdecimals,
            buffersize=buffersize,
            nan_string=nan_string,
            infinity_string=infinity_string,
            minus_infinity_string=minus_infinity_string,
            complex_real_string=complex_real_string,
            complex_imag_string=complex_imag_string,
        )


def from_awkward0(
    array,
    keep_layout=False,
    regulararray=False,
    recordarray=True,
    highlevel=True,
    behavior=None,
):
    """
    Args:
        array (Awkward 0.x or Awkward 1.x array): Data to convert to Awkward
            1.x.
        keep_layout (bool): If True, stay true to the Awkward 0.x layout,
            ensuring zero-copy; otherwise, allow transformations that copy
            data for more flexibility.
        regulararray (bool): If True and the array is multidimensional,
            the dimensions are represented by nested #ak.layout.RegularArray
            nodes; if False and the array is multidimensional, the dimensions
            are represented by a multivalued #ak.layout.NumpyArray.shape.
            If the array is one-dimensional, this has no effect.
        recordarray (bool): If True and the array is a NumPy structured array
            (dtype.names is not None), the fields are represented by an
            #ak.layout.RecordArray; if False and the array is a structured
            array, the structure is left in the #ak.layout.NumpyArray `format`,
            which some functions do not recognize.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Converts an array from Awkward 0.x to Awkward 1.x.

    This is only needed during the transition from the old library to the
    new library.

    If `array` is already an Awkward 1.x Array, it is simply passed through
    this function (so that interfaces that scripts don't need to remove this
    function when their 0.x sources are replaced by 1.x).
    """
    # See https://github.com/scikit-hep/awkward-0.x/blob/405b7eaeea51b60947a79c782b1abf0d72f6729b/specification.adoc
    import awkward0

    # If a source of Awkward 0.x arrays ever starts emitting Awkward 1.x arrays
    # (e.g. Uproot), this function turns into a pass-through.
    if isinstance(array, (ak.highlevel.Array, ak.highlevel.Record)):
        if highlevel:
            return array
        else:
            return array.layout
    elif isinstance(array, ak.highlevel.ArrayBuilder):
        if highlevel:
            return array.snapshot()
        else:
            return array._layout.snapshot()
    elif isinstance(array, (ak.layout.Content, ak.layout.Record)):
        return ak._util.maybe_wrap(array, behavior, highlevel)
    elif isinstance(array, ak.layout.ArrayBuilder):
        return ak._util.maybe_wrap(array.snapshot(), behavior, highlevel)

    def recurse(array, level):
        if isinstance(array, dict):
            keys = []
            values = []
            for n, x in array.items():
                keys.append(n)
                if isinstance(
                    x,
                    (
                        dict,
                        tuple,
                        numpy.ma.MaskedArray,
                        np.ndarray,
                        awkward0.array.base.AwkwardArray,
                    ),
                ):
                    values.append(recurse(x, level + 1)[np.newaxis])
                else:
                    values.append(ak.layout.NumpyArray(numpy.array([x])))
            return ak.layout.RecordArray(values, keys)[0]

        elif isinstance(array, tuple):
            values = []
            for x in array:
                if isinstance(
                    x,
                    (
                        dict,
                        tuple,
                        numpy.ma.MaskedArray,
                        np.ndarray,
                        awkward0.array.base.AwkwardArray,
                    ),
                ):
                    values.append(recurse(x, level + 1)[np.newaxis])
                else:
                    values.append(ak.layout.NumpyArray(numpy.array([x])))
            return ak.layout.RecordArray(values)[0]

        elif isinstance(array, numpy.ma.MaskedArray):
            return from_numpy(
                array,
                regulararray=regulararray,
                recordarray=recordarray,
                highlevel=False,
            )

        elif isinstance(array, np.ndarray):
            return from_numpy(
                array,
                regulararray=regulararray,
                recordarray=recordarray,
                highlevel=False,
            )

        elif isinstance(array, awkward0.JaggedArray):
            # starts, stops, content
            # offsetsaliased(starts, stops)
            startsmax = np.iinfo(array.starts.dtype.type).max
            stopsmax = np.iinfo(array.stops.dtype.type).max
            if (
                len(array.starts.shape) == 1
                and len(array.stops.shape) == 1
                and awkward0.JaggedArray.offsetsaliased(array.starts, array.stops)
            ):
                if startsmax >= from_awkward0.int64max:
                    offsets = ak.layout.Index64(array.offsets)
                    return ak.layout.ListOffsetArray64(
                        offsets, recurse(array.content, level + 1)
                    )
                elif startsmax >= from_awkward0.uint32max:
                    offsets = ak.layout.IndexU32(array.offsets)
                    return ak.layout.ListOffsetArrayU32(
                        offsets, recurse(array.content, level + 1)
                    )
                else:
                    offsets = ak.layout.Index32(array.offsets)
                    return ak.layout.ListOffsetArray32(
                        offsets, recurse(array.content, level + 1)
                    )

            else:
                if (
                    startsmax >= from_awkward0.int64max
                    or stopsmax >= from_awkward0.int64max
                ):
                    starts = ak.layout.Index64(array.starts.reshape(-1))
                    stops = ak.layout.Index64(array.stops.reshape(-1))
                    out = ak.layout.ListArray64(
                        starts, stops, recurse(array.content, level + 1)
                    )
                elif (
                    startsmax >= from_awkward0.uint32max
                    or stopsmax >= from_awkward0.uint32max
                ):
                    starts = ak.layout.IndexU32(array.starts.reshape(-1))
                    stops = ak.layout.IndexU32(array.stops.reshape(-1))
                    out = ak.layout.ListArrayU32(
                        starts, stops, recurse(array.content, level + 1)
                    )
                else:
                    starts = ak.layout.Index32(array.starts.reshape(-1))
                    stops = ak.layout.Index32(array.stops.reshape(-1))
                    out = ak.layout.ListArray32(
                        starts, stops, recurse(array.content, level + 1)
                    )
                for i in range(len(array.starts.shape) - 1, 0, -1):
                    out = ak.layout.RegularArray(
                        out, array.starts.shape[i], array.starts.shape[i - 1]
                    )
                return out

        elif isinstance(array, awkward0.Table):
            # contents
            if array.istuple:
                out = ak.layout.RecordArray(
                    [recurse(x, level + 1) for x in array.contents.values()]
                )
            else:
                keys = []
                values = []
                for n, x in array.contents.items():
                    keys.append(n)
                    values.append(recurse(x, level + 1))
                out = ak.layout.RecordArray(values, keys)

            if array._view is None:
                return out
            elif isinstance(array._view, tuple):
                start, step, length = array._view
                stop = start + step * length
                if stop < 0:
                    stop = None
                if step == 1 or step is None:
                    return out[start:stop]
                else:
                    return out[start:stop:step]
            else:
                return out[array._view]

        elif isinstance(array, awkward0.UnionArray):
            # tags, index, contents
            indexmax = np.iinfo(array.index.dtype.type).max
            if indexmax >= from_awkward0.int64max:
                tags = ak.layout.Index8(array.tags.reshape(-1))
                index = ak.layout.Index64(array.index.reshape(-1))
                out = ak.layout.UnionArray8_64(
                    tags, index, [recurse(x, level + 1) for x in array.contents]
                )
            elif indexmax >= from_awkward0.uint32max:
                tags = ak.layout.Index8(array.tags.reshape(-1))
                index = ak.layout.IndexU32(array.index.reshape(-1))
                out = ak.layout.UnionArray8_U32(
                    tags, index, [recurse(x, level + 1) for x in array.contents]
                )
            else:
                tags = ak.layout.Index8(array.tags.reshape(-1))
                index = ak.layout.Index32(array.index.reshape(-1))
                out = ak.layout.UnionArray8_32(
                    tags, index, [recurse(x, level + 1) for x in array.contents]
                )

            for i in range(len(array.tags.shape) - 1, 0, -1):
                out = ak.layout.RegularArray(
                    out, array.tags.shape[i], array.tags.shape[i - 1]
                )
            return out

        elif isinstance(array, awkward0.MaskedArray):
            # mask, content, maskedwhen
            mask = ak.layout.Index8(array.mask.view(np.int8).reshape(-1))
            out = ak.layout.ByteMaskedArray(
                mask,
                recurse(array.content, level + 1),
                valid_when=(not array.maskedwhen),
            )
            for i in range(len(array.mask.shape) - 1, 0, -1):
                out = ak.layout.RegularArray(
                    out, array.mask.shape[i], array.mask.shape[i - 1]
                )
            return out

        elif isinstance(array, awkward0.BitMaskedArray):
            # mask, content, maskedwhen, lsborder
            mask = ak.layout.IndexU8(array.mask.view(np.uint8))
            return ak.layout.BitMaskedArray(
                mask,
                recurse(array.content, level + 1),
                valid_when=(not array.maskedwhen),
                length=len(array.content),
                lsb_order=array.lsborder,
            )

        elif isinstance(array, awkward0.IndexedMaskedArray):
            # mask, content, maskedwhen
            indexmax = np.iinfo(array.index.dtype.type).max
            if indexmax >= from_awkward0.int64max:
                index = ak.layout.Index64(array.index.reshape(-1))
                out = ak.layout.IndexedOptionArray64(
                    index, recurse(array.content, level + 1)
                )
            elif indexmax >= from_awkward0.uint32max:
                index = ak.layout.IndexU32(array.index.reshape(-1))
                out = ak.layout.IndexedOptionArrayU32(
                    index, recurse(array.content, level + 1)
                )
            else:
                index = ak.layout.Index32(array.index.reshape(-1))
                out = ak.layout.IndexedOptionArray32(
                    index, recurse(array.content, level + 1)
                )
            for i in range(len(array.index.shape) - 1, 0, -1):
                out = ak.layout.RegularArray(
                    out, array.index.shape[i], array.index.shape[i - 1]
                )
            return out

        elif isinstance(array, awkward0.IndexedArray):
            # index, content
            indexmax = np.iinfo(array.index.dtype.type).max
            if indexmax >= from_awkward0.int64max:
                index = ak.layout.Index64(array.index.reshape(-1))
                out = ak.layout.IndexedArray64(index, recurse(array.content, level + 1))
            elif indexmax >= from_awkward0.uint32max:
                index = ak.layout.IndexU32(array.index.reshape(-1))
                out = ak.layout.IndexedArrayU32(
                    index, recurse(array.content, level + 1)
                )
            else:
                index = ak.layout.Index32(array.index.reshape(-1))
                out = ak.layout.IndexedArray32(index, recurse(array.content, level + 1))
            for i in range(len(array.index.shape) - 1, 0, -1):
                out = ak.layout.RegularArray(
                    out, array.index.shape[i], array.index.shape[i - 1]
                )
            return out

        elif isinstance(array, awkward0.SparseArray):
            # length, index, content, default
            if keep_layout:
                raise ValueError(
                    "ak.SparseArray hasn't been written (if at all); "
                    "try keep_layout=False" + ak._util.exception_suffix(__file__)
                )
            return recurse(array.dense, level + 1)

        elif isinstance(array, awkward0.StringArray):
            # starts, stops, content, encoding
            out = recurse(array._content, level + 1)
            if array.encoding is None:
                out.content.setparameter("__array__", "byte")
                out.setparameter("__array__", "bytestring")
            elif array.encoding == "utf-8":
                out.content.setparameter("__array__", "char")
                out.setparameter("__array__", "string")
            else:
                raise ValueError(
                    "unsupported encoding: {0}".format(repr(array.encoding))
                    + ak._util.exception_suffix(__file__)
                )
            return out

        elif isinstance(array, awkward0.ObjectArray):
            # content, generator, args, kwargs
            if keep_layout:
                raise ValueError(
                    "there isn't (and won't ever be) an Awkward 1.x equivalent "
                    "of awkward0.ObjectArray; try keep_layout=False"
                    + ak._util.exception_suffix(__file__)
                )
            out = recurse(array.content, level + 1)
            out.setparameter(
                "__record__",
                getattr(
                    array.generator,
                    "__qualname__",
                    getattr(array.generator, "__name__", repr(array.generator)),
                ),
            )
            return out

        if isinstance(array, awkward0.ChunkedArray):
            # chunks, chunksizes
            if keep_layout and level != 0:
                raise ValueError(
                    "Awkward 1.x PartitionedArrays are only allowed "
                    "at the root of a data structure, unlike "
                    "awkward0.ChunkedArray; try keep_layout=False"
                    + ak._util.exception_suffix(__file__)
                )
            elif level == 0:
                return ak.partition.IrregularlyPartitionedArray(
                    [recurse(x, level + 1) for x in array.chunks]
                )
            else:
                return ak.operations.structure.concatenate(
                    [recurse(x, level + 1) for x in array.chunks], highlevel=False
                )

        elif isinstance(array, awkward0.AppendableArray):
            # chunkshape, dtype, chunks
            raise ValueError(
                "the Awkward 1.x equivalent of awkward0.AppendableArray is "
                "ak.ArrayBuilder, but it is not a Content type, not "
                "mixable with immutable array elements"
                + ak._util.exception_suffix(__file__)
            )

        elif isinstance(array, awkward0.VirtualArray):
            # generator, args, kwargs, cache, persistentkey, type, nbytes, persistvirtual
            if keep_layout:
                raise NotImplementedError("FIXME" + ak._util.exception_suffix(__file__))
            else:
                return recurse(array.array, level + 1)

        else:
            raise TypeError(
                "not an awkward0 array: {0}".format(repr(array))
                + ak._util.exception_suffix(__file__)
            )

    out = recurse(array, 0)
    return ak._util.maybe_wrap(out, behavior, highlevel)


from_awkward0.int8max = np.iinfo(np.int8).max
from_awkward0.int32max = np.iinfo(np.int32).max
from_awkward0.uint32max = np.iinfo(np.uint32).max
from_awkward0.int64max = np.iinfo(np.int64).max


def to_awkward0(array, keep_layout=False):
    """
    Args:
        array: Data to convert into an Awkward 0.x array.
        keep_layout (bool): If True, stay true to the Awkward 1.x layout,
            ensuring zero-copy; otherwise, allow transformations that copy
            data for more flexibility.

    Converts `array` (many types supported, including all Awkward Arrays and
    Records) into an Awkward 0.x array.

    This is only needed during the transition from the old library to the new
    library.
    """
    # See https://github.com/scikit-hep/awkward-0.x/blob/405b7eaeea51b60947a79c782b1abf0d72f6729b/specification.adoc
    import awkward0

    def recurse(layout):
        if isinstance(layout, ak.partition.PartitionedArray):
            return awkward0.ChunkedArray([recurse(x) for x in layout.partitions])

        elif isinstance(layout, ak.layout.NumpyArray):
            return numpy.asarray(layout)

        elif isinstance(layout, ak.layout.EmptyArray):
            return numpy.array([])

        elif isinstance(layout, ak.layout.RegularArray):
            # content, size
            if keep_layout:
                raise ValueError(
                    "awkward0 has no equivalent of RegularArray; "
                    "try keep_layout=False" + ak._util.exception_suffix(__file__)
                )
            offsets = numpy.arange(0, (len(layout) + 1) * layout.size, layout.size)
            return awkward0.JaggedArray.fromoffsets(offsets, recurse(layout.content))

        elif isinstance(layout, ak.layout.ListArray32):
            # starts, stops, content
            return awkward0.JaggedArray(
                numpy.asarray(layout.starts),
                numpy.asarray(layout.stops),
                recurse(layout.content),
            )

        elif isinstance(layout, ak.layout.ListArrayU32):
            # starts, stops, content
            return awkward0.JaggedArray(
                numpy.asarray(layout.starts),
                numpy.asarray(layout.stops),
                recurse(layout.content),
            )

        elif isinstance(layout, ak.layout.ListArray64):
            # starts, stops, content
            return awkward0.JaggedArray(
                numpy.asarray(layout.starts),
                numpy.asarray(layout.stops),
                recurse(layout.content),
            )

        elif isinstance(layout, ak.layout.ListOffsetArray32):
            # offsets, content
            return awkward0.JaggedArray.fromoffsets(
                numpy.asarray(layout.offsets), recurse(layout.content)
            )

        elif isinstance(layout, ak.layout.ListOffsetArrayU32):
            # offsets, content
            return awkward0.JaggedArray.fromoffsets(
                numpy.asarray(layout.offsets), recurse(layout.content)
            )

        elif isinstance(layout, ak.layout.ListOffsetArray64):
            # offsets, content
            return awkward0.JaggedArray.fromoffsets(
                numpy.asarray(layout.offsets), recurse(layout.content)
            )

        elif isinstance(layout, ak.layout.Record):
            # istuple, numfields, field(i)
            out = []
            for i in range(layout.numfields):
                content = layout.field(i)
                if isinstance(content, (ak.layout.Content, ak.layout.Record)):
                    out.append(recurse(content))
                else:
                    out.append(content)
            if layout.istuple:
                return tuple(out)
            else:
                return dict(zip(layout.keys(), out))

        elif isinstance(layout, ak.layout.RecordArray):
            # istuple, numfields, field(i)
            if layout.numfields == 0 and len(layout) != 0:
                raise ValueError(
                    "cannot convert zero-field, nonzero-length RecordArray "
                    "to awkward0.Table (limitation in awkward0)"
                    + ak._util.exception_suffix(__file__)
                )
            keys = layout.keys()
            values = [recurse(x) for x in layout.contents]
            pairs = collections.OrderedDict(zip(keys, values))
            out = awkward0.Table(pairs)
            if layout.istuple:
                out._rowname = "tuple"
            return out

        elif isinstance(layout, ak.layout.UnionArray8_32):
            # tags, index, numcontents, content(i)
            return awkward0.UnionArray(
                numpy.asarray(layout.tags),
                numpy.asarray(layout.index),
                [recurse(x) for x in layout.contents],
            )

        elif isinstance(layout, ak.layout.UnionArray8_U32):
            # tags, index, numcontents, content(i)
            return awkward0.UnionArray(
                numpy.asarray(layout.tags),
                numpy.asarray(layout.index),
                [recurse(x) for x in layout.contents],
            )

        elif isinstance(layout, ak.layout.UnionArray8_64):
            # tags, index, numcontents, content(i)
            return awkward0.UnionArray(
                numpy.asarray(layout.tags),
                numpy.asarray(layout.index),
                [recurse(x) for x in layout.contents],
            )

        elif isinstance(layout, ak.layout.IndexedOptionArray32):
            # index, content
            index = numpy.asarray(layout.index)
            toosmall = index < -1
            if toosmall.any():
                index = index.copy()
                index[toosmall] = -1
            return awkward0.IndexedMaskedArray(index, recurse(layout.content))

        elif isinstance(layout, ak.layout.IndexedOptionArray64):
            # index, content
            index = numpy.asarray(layout.index)
            toosmall = index < -1
            if toosmall.any():
                index = index.copy()
                index[toosmall] = -1
            return awkward0.IndexedMaskedArray(index, recurse(layout.content))

        elif isinstance(layout, ak.layout.IndexedArray32):
            # index, content
            return awkward0.IndexedArray(
                numpy.asarray(layout.index), recurse(layout.content)
            )

        elif isinstance(layout, ak.layout.IndexedArrayU32):
            # index, content
            return awkward0.IndexedArray(
                numpy.asarray(layout.index), recurse(layout.content)
            )

        elif isinstance(layout, ak.layout.IndexedArray64):
            # index, content
            return awkward0.IndexedArray(
                numpy.asarray(layout.index), recurse(layout.content)
            )

        elif isinstance(layout, ak.layout.ByteMaskedArray):
            # mask, content, valid_when
            return awkward0.MaskedArray(
                numpy.asarray(layout.mask),
                recurse(layout.content),
                maskedwhen=(not layout.valid_when),
            )

        elif isinstance(layout, ak.layout.BitMaskedArray):
            # mask, content, valid_when, length, lsb_order
            return awkward0.BitMaskedArray(
                numpy.asarray(layout.mask),
                recurse(layout.content),
                maskedwhen=(not layout.valid_when),
                lsborder=layout.lsb_order,
            )

        elif isinstance(layout, ak.layout.UnmaskedArray):
            # content
            return recurse(layout.content)  # no equivalent in awkward0

        elif isinstance(layout, ak.layout.VirtualArray):
            raise NotImplementedError("FIXME" + ak._util.exception_suffix(__file__))

        else:
            raise AssertionError(
                "missing converter for {0}".format(type(layout).__name__)
                + ak._util.exception_suffix(__file__)
            )

    layout = to_layout(
        array, allow_record=True, allow_other=False, numpytype=(np.generic,)
    )
    return recurse(layout)


def to_layout(
    array,
    allow_record=True,
    allow_other=False,
    numpytype=(np.number, np.bool_, np.str_, np.bytes_, np.datetime64, np.timedelta64),
):
    """
    Args:
        array: Data to convert into an #ak.layout.Content and maybe
            #ak.layout.Record and other types.
        allow_record (bool): If True, allow #ak.layout.Record as an output;
            otherwise, if the output would be a scalar record, raise an error.
        allow_other (bool): If True, allow non-Awkward outputs; otherwise,
            if the output would be another type, raise an error.
        numpytype (tuple of NumPy types): Allowed NumPy types in
            #ak.layout.NumpyArray outputs.

    Converts `array` (many types supported, including all Awkward Arrays and
    Records) into a #ak.layout.Content and maybe #ak.layout.Record and other
    types.

    This function is usually used to sanitize inputs for other functions; it
    would rarely be used in a data analysis.
    """
    if isinstance(array, ak.highlevel.Array):
        return array.layout

    elif allow_record and isinstance(array, ak.highlevel.Record):
        return array.layout

    elif isinstance(array, ak.highlevel.ArrayBuilder):
        return array.snapshot().layout

    elif isinstance(array, ak.layout.ArrayBuilder):
        return array.snapshot()

    elif isinstance(array, (ak.layout.Content, ak.partition.PartitionedArray)):
        return array

    elif allow_record and isinstance(array, ak.layout.Record):
        return array

    elif isinstance(array, (np.ndarray, numpy.ma.MaskedArray)):
        if not issubclass(array.dtype.type, numpytype):
            raise ValueError(
                "NumPy {0} not allowed".format(repr(array.dtype))
                + ak._util.exception_suffix(__file__)
            )
        return from_numpy(array, regulararray=True, recordarray=True, highlevel=False)

    elif (
        type(array).__module__.startswith("cupy.") and type(array).__name__ == "ndarray"
    ):
        return from_cupy(array, regulararray=True, highlevel=False)

    elif isinstance(array, (str, bytes)) or (
        ak._util.py27 and isinstance(array, ak._util.unicode)
    ):
        return from_iter([array], highlevel=False)

    elif isinstance(array, Iterable):
        return from_iter(array, highlevel=False)

    elif not allow_other:
        raise TypeError(
            "{0} cannot be converted into an Awkward Array".format(array)
            + ak._util.exception_suffix(__file__)
        )

    else:
        return array


def regularize_numpyarray(array, allow_empty=True, highlevel=True, behavior=None):
    """
    Args:
        array: Data to convert into an Awkward Array.
        allow_empty (bool): If True, allow #ak.layout.EmptyArray in the output;
            otherwise, convert empty arrays into #ak.layout.NumpyArray.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Converts any multidimensional #ak.layout.NumpyArray.shape into nested
    #ak.layout.RegularArray nodes. The output may have any Awkward data type:
    this only changes the representation of #ak.layout.NumpyArray.

    This function is usually used to sanitize inputs for other functions; it
    would rarely be used in a data analysis.
    """

    def getfunction(layout):
        if isinstance(layout, ak.layout.NumpyArray) and layout.ndim != 1:
            return lambda: layout.toRegularArray()
        elif isinstance(layout, ak.layout.EmptyArray) and not allow_empty:
            return lambda: layout.toNumpyArray()
        elif isinstance(layout, ak.layout.VirtualArray):
            # FIXME: we must transform the Form (replacing inner_shape with
            # RegularForms) and wrap the ArrayGenerator with regularize_numpy
            return lambda: layout
        else:
            return None

    out = ak._util.recursively_apply(to_layout(array), getfunction, pass_depth=False)
    return ak._util.maybe_wrap_like(out, array, behavior, highlevel)


def _import_pyarrow(name):
    try:
        import pyarrow
    except ImportError:
        raise ImportError(
            """to use {0}, you must install pyarrow:

    pip install pyarrow

or

    conda install -c conda-forge pyarrow
""".format(
                name
            )
        )
    else:
        if distutils.version.LooseVersion(
            pyarrow.__version__
        ) < distutils.version.LooseVersion("2.0.0"):
            raise ImportError("pyarrow 2.0.0 or later required for {0}".format(name))
        return pyarrow


def to_arrow(
    array, list_to32=False, string_to32=True, bytestring_to32=True, allow_tensor=True
):
    """
    Args:
        array: Data to convert to an Apache Arrow array.
        list_to32 (bool): If True, convert Awkward lists into 32-bit Arrow lists
            if they're small enough, even if it means an extra conversion. Otherwise,
            signed 32-bit #ak.layout.ListOffsetArray maps to Arrow `ListType` and
            all others map to Arrow `LargeListType`.
        string_to32 (bool): Same as the above for Arrow `string` and `large_string`.
        bytestring_to32 (bool): Same as the above for Arrow `binary` and `large_binary`.
        allow_tensor (bool): If True, convert regular-length lists to `pyarrow.lib.Tensor`;
            otherwise, make `pyarrow.lib.ListArray` (generating offsets). This is used
            by #ak.to_parquet, since Parquet files can't contain regular-length tensors.

    Converts an Awkward Array into an Apache Arrow array.

    This produces arrays of type `pyarrow.Array`. You might need to further
    manipulations (using the pyarrow library) to build a `pyarrow.ChunkedArray`,
    a `pyarrow.RecordBatch`, or a `pyarrow.Table`.

    Arrow arrays can maintain the distinction between "option-type but no elements are
    missing" and "not option-type" at all levels except the top level. Also, there is
    no distinction between `?union[X, Y, Z]]` type and `union[?X, ?Y, ?Z]` type. Be
    aware of these type distinctions when passing data through Arrow or Parquet.

    See also #ak.from_arrow, #ak.to_arrow_table, #ak.to_parquet.
    """
    pyarrow = _import_pyarrow("ak.to_arrow")

    layout = to_layout(array, allow_record=False, allow_other=False)

    def recurse(layout, mask, is_option):
        if isinstance(layout, ak.layout.NumpyArray):
            numpy_arr = numpy.asarray(layout)
            length = len(numpy_arr)
            arrow_type = pyarrow.from_numpy_dtype(numpy_arr.dtype)

            if issubclass(numpy_arr.dtype.type, (bool, np.bool_)):
                if numpy_arr.ndim == 1:
                    if len(numpy_arr) % 8 == 0:
                        ready_to_pack = numpy_arr
                    else:
                        ready_to_pack = numpy.empty(
                            int(numpy.ceil(len(numpy_arr) / 8.0)) * 8,
                            dtype=numpy_arr.dtype,
                        )
                        ready_to_pack[: len(numpy_arr)] = numpy_arr
                        ready_to_pack[len(numpy_arr) :] = 0
                    numpy_arr = numpy.packbits(
                        ready_to_pack.reshape(-1, 8)[:, ::-1].reshape(-1)
                    )
                else:
                    return recurse(
                        from_numpy(numpy_arr, regulararray=True, highlevel=False),
                        mask,
                        is_option,
                    )

            if numpy_arr.ndim == 1:
                if mask is not None:
                    return pyarrow.Array.from_buffers(
                        arrow_type,
                        length,
                        [pyarrow.py_buffer(mask), pyarrow.py_buffer(numpy_arr)],
                    )
                else:
                    return pyarrow.Array.from_buffers(
                        arrow_type, length, [None, pyarrow.py_buffer(numpy_arr)]
                    )
            elif allow_tensor:
                return pyarrow.Tensor.from_numpy(numpy_arr)
            else:
                return recurse(
                    from_numpy(numpy_arr, regulararray=True, highlevel=False),
                    mask,
                    is_option,
                )

        elif isinstance(layout, ak.layout.EmptyArray):
            return pyarrow.Array.from_buffers(pyarrow.null(), 0, [None])

        elif isinstance(layout, ak.layout.ListOffsetArray32):
            offsets = numpy.asarray(layout.offsets, dtype=np.int32)

            if layout.parameter("__array__") == "bytestring":
                if mask is None:
                    arrow_arr = pyarrow.Array.from_buffers(
                        pyarrow.binary(),
                        len(offsets) - 1,
                        [
                            None,
                            pyarrow.py_buffer(offsets),
                            pyarrow.py_buffer(layout.content),
                        ],
                        children=[],
                    )
                else:
                    arrow_arr = pyarrow.Array.from_buffers(
                        pyarrow.binary(),
                        len(offsets) - 1,
                        [
                            pyarrow.py_buffer(mask),
                            pyarrow.py_buffer(offsets),
                            pyarrow.py_buffer(layout.content),
                        ],
                        children=[],
                    )
                return arrow_arr

            if layout.parameter("__array__") == "string":
                if mask is None:
                    arrow_arr = pyarrow.StringArray.from_buffers(
                        len(offsets) - 1,
                        pyarrow.py_buffer(offsets),
                        pyarrow.py_buffer(layout.content),
                    )
                else:
                    arrow_arr = pyarrow.StringArray.from_buffers(
                        len(offsets) - 1,
                        pyarrow.py_buffer(offsets),
                        pyarrow.py_buffer(layout.content),
                        pyarrow.py_buffer(mask),
                    )
                return arrow_arr

            content_buffer = recurse(layout.content[: offsets[-1]], None, False)
            content_type = pyarrow.list_(content_buffer.type).value_field.with_nullable(
                isinstance(
                    ak.operations.describe.type(layout.content), ak.types.OptionType
                )
            )
            if mask is None:
                arrow_arr = pyarrow.Array.from_buffers(
                    pyarrow.list_(content_type),
                    len(offsets) - 1,
                    [None, pyarrow.py_buffer(offsets)],
                    children=[content_buffer],
                )
            else:
                arrow_arr = pyarrow.Array.from_buffers(
                    pyarrow.list_(content_type),
                    len(offsets) - 1,
                    [pyarrow.py_buffer(mask), pyarrow.py_buffer(offsets)],
                    children=[content_buffer],
                )
            return arrow_arr

        elif isinstance(
            layout,
            (ak.layout.ListOffsetArray64, ak.layout.ListOffsetArrayU32),
        ):
            if layout.parameter("__array__") == "bytestring":
                downsize = bytestring_to32
            elif layout.parameter("__array__") == "string":
                downsize = string_to32
            else:
                downsize = list_to32

            offsets = numpy.asarray(layout.offsets)

            if downsize and offsets[-1] <= np.iinfo(np.int32).max:
                small_layout = ak.layout.ListOffsetArray32(
                    ak.layout.Index32(offsets.astype(np.int32)),
                    layout.content,
                    parameters=layout.parameters,
                )
                return recurse(small_layout, mask, is_option)

            offsets = numpy.asarray(layout.offsets, dtype=np.int64)

            if layout.parameter("__array__") == "bytestring":
                if mask is None:
                    arrow_arr = pyarrow.Array.from_buffers(
                        pyarrow.large_binary(),
                        len(offsets) - 1,
                        [
                            None,
                            pyarrow.py_buffer(offsets),
                            pyarrow.py_buffer(layout.content),
                        ],
                        children=[],
                    )
                else:
                    arrow_arr = pyarrow.Array.from_buffers(
                        pyarrow.large_binary(),
                        len(offsets) - 1,
                        [
                            pyarrow.py_buffer(mask),
                            pyarrow.py_buffer(offsets),
                            pyarrow.py_buffer(layout.content),
                        ],
                        children=[],
                    )
                return arrow_arr

            if layout.parameter("__array__") == "string":
                if mask is None:
                    arrow_arr = pyarrow.LargeStringArray.from_buffers(
                        len(offsets) - 1,
                        pyarrow.py_buffer(offsets),
                        pyarrow.py_buffer(layout.content),
                    )
                else:
                    arrow_arr = pyarrow.LargeStringArray.from_buffers(
                        len(offsets) - 1,
                        pyarrow.py_buffer(offsets),
                        pyarrow.py_buffer(layout.content),
                        pyarrow.py_buffer(mask),
                    )
                return arrow_arr

            content_buffer = recurse(layout.content[: offsets[-1]], None, False)
            content_type = pyarrow.list_(content_buffer.type).value_field.with_nullable(
                isinstance(
                    ak.operations.describe.type(layout.content), ak.types.OptionType
                )
            )
            if mask is None:
                arrow_arr = pyarrow.Array.from_buffers(
                    pyarrow.large_list(content_type),
                    len(offsets) - 1,
                    [None, pyarrow.py_buffer(offsets)],
                    children=[content_buffer],
                )
            else:
                arrow_arr = pyarrow.Array.from_buffers(
                    pyarrow.large_list(content_type),
                    len(offsets) - 1,
                    [pyarrow.py_buffer(mask), pyarrow.py_buffer(offsets)],
                    children=[content_buffer],
                )
            return arrow_arr

        elif isinstance(layout, ak.layout.RegularArray):
            return recurse(
                layout.broadcast_tooffsets64(layout.compact_offsets64()),
                mask,
                is_option,
            )

        elif isinstance(
            layout,
            (
                ak.layout.ListArray32,
                ak.layout.ListArrayU32,
                ak.layout.ListArray64,
            ),
        ):
            return recurse(
                layout.broadcast_tooffsets64(layout.compact_offsets64()),
                mask,
                is_option,
            )

        elif isinstance(layout, ak.layout.RecordArray):
            values = [
                recurse(x[: len(layout)], mask, is_option) for x in layout.contents
            ]

            min_list_len = min(map(len, values))

            types = pyarrow.struct(
                [
                    pyarrow.field(layout.key(i), values[i].type).with_nullable(
                        isinstance(ak.operations.describe.type(x), ak.types.OptionType)
                    )
                    for i, x in enumerate(layout.contents)
                ]
            )

            if mask is not None:
                return pyarrow.Array.from_buffers(
                    types, min_list_len, [pyarrow.py_buffer(mask)], children=values
                )
            else:
                return pyarrow.Array.from_buffers(
                    types, min_list_len, [None], children=values
                )

        elif isinstance(
            layout,
            (
                ak.layout.UnionArray8_32,
                ak.layout.UnionArray8_64,
                ak.layout.UnionArray8_U32,
            ),
        ):
            tags = numpy.asarray(layout.tags)
            index = numpy.asarray(layout.index)
            copied_index = False
            if mask is not None:
                bytemask = (
                    numpy.unpackbits(mask)
                    .reshape(-1, 8)[:, ::-1]
                    .reshape(-1)
                    .view(np.bool_)
                )[: len(tags)]

            values = []
            for tag, content in enumerate(layout.contents):
                selected_tags = tags == tag
                this_index = index[selected_tags]
                if mask is not None:
                    length = int(numpy.ceil(len(this_index) / 8.0)) * 8
                    if len(numpy.unique(this_index)) == len(this_index):
                        this_bytemask = numpy.zeros(length, dtype=np.uint8)
                        this_bytemask[this_index] = bytemask[selected_tags]
                    else:
                        this_bytemask = numpy.empty(length, dtype=np.uint8)
                        this_bytemask[: len(this_index)] = bytemask[selected_tags]
                        this_bytemask[len(this_index) :] = 0

                        content = content[this_index]
                        this_index = numpy.arange(len(this_index))
                        if not copied_index:
                            copied_index = True
                            index = numpy.array(index, copy=True)
                        index[selected_tags] = this_index

                    this_mask = numpy.packbits(
                        this_bytemask.reshape(-1, 8)[:, ::-1].reshape(-1)
                    )

                else:
                    this_mask = None

                values.append(recurse(content, this_mask, is_option))

            types = pyarrow.union(
                [
                    pyarrow.field(str(i), values[i].type).with_nullable(
                        is_option
                        or isinstance(layout.content(i).type, ak.types.OptionType)
                    )
                    for i in range(len(values))
                ],
                "dense",
                list(range(len(values))),
            )

            return pyarrow.Array.from_buffers(
                types,
                len(layout.tags),
                [
                    None,
                    pyarrow.py_buffer(tags),
                    pyarrow.py_buffer(index.astype(np.int32)),
                ],
                children=values,
            )

        elif isinstance(
            layout,
            (
                ak.layout.IndexedArray32,
                ak.layout.IndexedArrayU32,
                ak.layout.IndexedArray64,
            ),
        ):
            index = numpy.asarray(layout.index)

            if layout.parameter("__array__") == "categorical":
                dictionary = recurse(layout.content, None, False)
                if mask is None:
                    return pyarrow.DictionaryArray.from_arrays(index, dictionary)
                else:
                    bytemask = (
                        numpy.unpackbits(~mask)
                        .reshape(-1, 8)[:, ::-1]
                        .reshape(-1)
                        .view(np.bool_)
                    )[: len(index)]
                    return pyarrow.DictionaryArray.from_arrays(
                        index, dictionary, bytemask
                    )

            else:
                layout_content = layout.content

                if len(layout_content) == 0:
                    empty = recurse(layout_content, None, False)
                    if mask is None:
                        return empty
                    else:
                        return pyarrow.array([None] * len(index)).cast(empty.type)

                elif isinstance(layout_content, ak.layout.RecordArray):
                    values = [
                        recurse(x[: len(layout_content)][index], mask, is_option)
                        for x in layout_content.contents
                    ]

                    min_list_len = min(map(len, values))

                    types = pyarrow.struct(
                        [
                            pyarrow.field(
                                layout_content.key(i), values[i].type
                            ).with_nullable(
                                isinstance(
                                    ak.operations.describe.type(x), ak.types.OptionType
                                )
                            )
                            for i, x in enumerate(layout_content.contents)
                        ]
                    )

                    if mask is not None:
                        return pyarrow.Array.from_buffers(
                            types,
                            min_list_len,
                            [pyarrow.py_buffer(mask)],
                            children=values,
                        )
                    else:
                        return pyarrow.Array.from_buffers(
                            types, min_list_len, [None], children=values
                        )

                else:
                    return recurse(layout_content[index], mask, is_option)

        elif isinstance(
            layout,
            (ak.layout.IndexedOptionArray32, ak.layout.IndexedOptionArray64),
        ):
            index = numpy.array(layout.index, copy=True)
            nulls = index < 0
            index[nulls] = 0

            if layout.parameter("__array__") == "categorical":
                dictionary = recurse(layout.content, None, False)

                if mask is None:
                    bytemask = nulls
                else:
                    bytemask = (
                        numpy.unpackbits(~mask)
                        .reshape(-1, 8)[:, ::-1]
                        .reshape(-1)
                        .view(np.bool_)
                    )[: len(index)]
                    bytemask[nulls] = True

                return pyarrow.DictionaryArray.from_arrays(index, dictionary, bytemask)

            else:
                if len(nulls) % 8 == 0:
                    this_bytemask = (~nulls).view(np.uint8)
                else:
                    length = int(numpy.ceil(len(nulls) / 8.0)) * 8
                    this_bytemask = numpy.empty(length, dtype=np.uint8)
                    this_bytemask[: len(nulls)] = ~nulls
                    this_bytemask[len(nulls) :] = 0

                this_bitmask = numpy.packbits(
                    this_bytemask.reshape(-1, 8)[:, ::-1].reshape(-1)
                )

                if isinstance(layout, ak.layout.IndexedOptionArray32):
                    next = ak.layout.IndexedArray32(
                        ak.layout.Index32(index), layout.content
                    )
                else:
                    next = ak.layout.IndexedArray64(
                        ak.layout.Index64(index), layout.content
                    )

                if mask is None:
                    return recurse(next, this_bitmask, True)
                else:
                    return recurse(next, mask & this_bitmask, True)

        elif isinstance(layout, ak.layout.BitMaskedArray):
            bitmask = numpy.asarray(layout.mask, dtype=np.uint8)

            if layout.lsb_order is False:
                bitmask = numpy.packbits(
                    numpy.unpackbits(bitmask).reshape(-1, 8)[:, ::-1].reshape(-1)
                )

            if layout.valid_when is False:
                bitmask = ~bitmask

            return recurse(layout.content[: len(layout)], bitmask, True).slice(
                length=min(len(bitmask) * 8, len(layout.content))
            )

        elif isinstance(layout, ak.layout.ByteMaskedArray):
            mask = numpy.asarray(layout.mask, dtype=np.bool_) == layout.valid_when

            bytemask = numpy.zeros(
                8 * math.ceil(len(layout.content) / 8), dtype=np.bool_
            )
            bytemask[: len(mask)] = mask
            bytemask[len(mask) :] = 0
            bitmask = numpy.packbits(bytemask.reshape(-1, 8)[:, ::-1].reshape(-1))

            return recurse(layout.content[: len(layout)], bitmask, True).slice(
                length=len(mask)
            )

        elif isinstance(layout, (ak.layout.UnmaskedArray)):
            return recurse(layout.content, None, True)

        elif isinstance(layout, (ak.layout.VirtualArray)):
            return recurse(layout.array, None, False)

        elif isinstance(layout, (ak.partition.PartitionedArray)):
            return pyarrow.chunked_array(
                [recurse(x, None, False) for x in layout.partitions]
            )

        else:
            raise TypeError(
                "unrecognized array type: {0}".format(repr(layout))
                + ak._util.exception_suffix(__file__)
            )

    return recurse(layout, None, False)


def to_arrow_table(
    array,
    explode_records=False,
    list_to32=False,
    string_to32=True,
    bytestring_to32=True,
):
    """
    Args:
        array: Data to convert to an Apache Arrow table.
        explode_records (bool): If True, lists of records are written as
            records of lists, so that nested fields become top-level fields
            (which can be zipped when read back).
        list_to32 (bool): If True, convert Awkward lists into 32-bit Arrow lists
            if they're small enough, even if it means an extra conversion. Otherwise,
            signed 32-bit #ak.layout.ListOffsetArray maps to Arrow `ListType` and
            all others map to Arrow `LargeListType`.
        string_to32 (bool): Same as the above for Arrow `string` and `large_string`.
        bytestring_to32 (bool): Same as the above for Arrow `binary` and `large_binary`.

    Converts an Awkward Array into an Apache Arrow table (`pyarrow.Table`).

    If the `array` does not contain records at top-level, the Arrow table will consist
    of one field whose name is `""`.

    Arrow tables can maintain the distinction between "option-type but no elements are
    missing" and "not option-type" at all levels, including the top level. However,
    there is no distinction between `?union[X, Y, Z]]` type and `union[?X, ?Y, ?Z]` type.
    Be aware of these type distinctions when passing data through Arrow or Parquet.

    See also #ak.to_arrow, #ak.from_arrow, #ak.to_parquet.
    """
    pyarrow = _import_pyarrow("ak.to_arrow_table")

    layout = to_layout(array, allow_record=False, allow_other=False)

    if explode_records or isinstance(
        ak.operations.describe.type(layout), ak.types.RecordType
    ):
        names = layout.keys()
        contents = [layout[name] for name in names]
    else:
        names = [""]
        contents = [layout]

    pa_arrays = []
    pa_fields = []
    for name, content in zip(names, contents):
        pa_arrays.append(
            to_arrow(
                content,
                list_to32=list_to32,
                string_to32=string_to32,
                bytestring_to32=bytestring_to32,
            )
        )
        pa_fields.append(
            pyarrow.field(name, pa_arrays[-1].type).with_nullable(
                isinstance(ak.operations.describe.type(content), ak.types.OptionType)
            )
        )

    batch = pyarrow.RecordBatch.from_arrays(pa_arrays, schema=pyarrow.schema(pa_fields))
    return pyarrow.Table.from_batches([batch])


def from_arrow(array, highlevel=True, behavior=None):
    """
    Args:
        array (`pyarrow.Array`, `pyarrow.ChunkedArray`, `pyarrow.RecordBatch`,
            or `pyarrow.Table`): Apache Arrow array to convert into an
            Awkward Array.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Converts an Apache Arrow array into an Awkward Array.

    Arrow arrays can maintain the distinction between "option-type but no elements are
    missing" and "not option-type" at all levels except the top level. Arrow tables
    can maintain the distinction at all levels. However, note that there is no distinction
    between `?union[X, Y, Z]]` type and `union[?X, ?Y, ?Z]` type. Be aware of these
    type distinctions when passing data through Arrow or Parquet.

    See also #ak.to_arrow, #ak.to_arrow_table.
    """
    return _from_arrow(array, True, highlevel=highlevel, behavior=behavior)


_pyarrow_to_numpy_dtype = {
    "date32": (True, np.dtype("M8[D]")),
    "date64": (False, np.dtype("M8[ms]")),
    "date32[day]": (True, np.dtype("M8[D]")),
    "date64[ms]": (False, np.dtype("M8[ms]")),
    "time32[s]": (True, np.dtype("M8[s]")),
    "time32[ms]": (True, np.dtype("M8[ms]")),
    "time64[us]": (False, np.dtype("M8[us]")),
    "time64[ns]": (False, np.dtype("M8[ns]")),
    "timestamp[s]": (False, np.dtype("M8[s]")),
    "timestamp[ms]": (False, np.dtype("M8[ms]")),
    "timestamp[us]": (False, np.dtype("M8[us]")),
    "timestamp[ns]": (False, np.dtype("M8[ns]")),
    "duration[s]": (False, np.dtype("m8[s]")),
    "duration[ms]": (False, np.dtype("m8[ms]")),
    "duration[us]": (False, np.dtype("m8[us]")),
    "duration[ns]": (False, np.dtype("m8[ns]")),
}


def _from_arrow(
    array, pass_empty_field, struct_only=None, highlevel=True, behavior=None
):
    pyarrow = _import_pyarrow("ak.from_arrow")

    def popbuffers(array, tpe, buffers):
        if isinstance(tpe, pyarrow.lib.DictionaryType):
            index = popbuffers(array.indices, tpe.index_type, buffers)
            content = handle_arrow(array.dictionary)

            out = ak.layout.IndexedArray32(
                ak.layout.Index32(index.content),
                content,
                parameters={"__array__": "categorical"},
            ).simplify()

            if isinstance(index, ak.layout.BitMaskedArray):
                return ak.layout.BitMaskedArray(
                    index.mask,
                    out,
                    True,
                    len(index),
                    True,
                    parameters={"__array__": "categorical"},
                ).simplify()
            else:
                return out
            # RETURNED because 'index' has already been offset-corrected.

        elif isinstance(tpe, pyarrow.lib.StructType):
            assert tpe.num_buffers == 1
            mask = buffers.pop(0)
            child_arrays = []
            keys = []

            if struct_only is None:
                for i in range(tpe.num_fields):
                    content = popbuffers(array.field(tpe[i].name), tpe[i].type, buffers)
                    if not tpe[i].nullable:
                        content = content.content
                    child_arrays.append(content)
                    keys.append(tpe[i].name)
            else:
                target = struct_only.pop()
                found = False
                for i in range(tpe.num_fields):
                    if tpe[i].name == target:
                        found = True
                        content = popbuffers(
                            array.field(tpe[i].name), tpe[i].type, buffers
                        )
                        if not tpe[i].nullable:
                            content = content.content
                        child_arrays.append(content)
                        keys.append(tpe[i].name)
                assert found

            out = ak.layout.RecordArray(child_arrays, keys, length=len(array))
            if mask is not None:
                mask = ak.layout.IndexU8(numpy.frombuffer(mask, dtype=np.uint8))
                return ak.layout.BitMaskedArray(mask, out, True, len(out), True)
            else:
                return ak.layout.UnmaskedArray(out)
            # RETURNED because each field has already been offset-corrected.

        elif isinstance(tpe, pyarrow.lib.ListType):
            assert tpe.num_buffers == 2
            mask = buffers.pop(0)
            offsets = ak.layout.Index32(
                numpy.frombuffer(buffers.pop(0), dtype=np.int32)
            )
            content = popbuffers(array.values, tpe.value_type, buffers)
            if not tpe.value_field.nullable:
                content = content.content

            out = ak.layout.ListOffsetArray32(offsets, content)
            # No return yet!

        elif isinstance(tpe, pyarrow.lib.LargeListType):
            assert tpe.num_buffers == 2
            mask = buffers.pop(0)
            offsets = ak.layout.Index64(
                numpy.frombuffer(buffers.pop(0), dtype=np.int64)
            )
            content = popbuffers(array.values, tpe.value_type, buffers)
            if not tpe.value_field.nullable:
                content = content.content

            out = ak.layout.ListOffsetArray64(offsets, content)
            # No return yet!

        elif isinstance(tpe, pyarrow.lib.FixedSizeListType):
            assert tpe.num_buffers == 1
            mask = buffers.pop(0)
            content = popbuffers(array.values, tpe.value_type, buffers)
            if not tpe.value_field.nullable:
                content = content.content

            out = ak.layout.RegularArray(content, tpe.list_size)
            # No return yet!

        elif isinstance(tpe, pyarrow.lib.UnionType):
            if tpe.mode == "sparse":
                assert tpe.num_buffers == 2
            elif tpe.mode == "dense":
                assert tpe.num_buffers == 3
            else:
                raise TypeError(
                    "unrecognized Arrow union array mode: {0}".format(repr(tpe.mode))
                    + ak._util.exception_suffix(__file__)
                )

            mask = buffers.pop(0)
            tags = numpy.frombuffer(buffers.pop(0), dtype=np.int8)
            if tpe.mode == "sparse":
                index = numpy.arange(len(tags), dtype=np.int32)
            else:
                index = numpy.frombuffer(buffers.pop(0), dtype=np.int32)

            contents = []
            for i in range(tpe.num_fields):
                contents.append(popbuffers(array.field(i), tpe[i].type, buffers))
            for i in range(len(contents)):
                these = index[tags == i]
                if len(these) == 0:
                    contents[i] = contents[i][0:0]
                elif tpe.mode == "sparse":
                    contents[i] = contents[i][: these[-1] + 1]
                else:
                    contents[i] = contents[i][: these.max() + 1]
            for i in range(len(contents)):
                if not tpe[i].nullable:
                    contents[i] = contents[i].content

            tags = ak.layout.Index8(tags)
            index = ak.layout.Index32(index)
            out = ak.layout.UnionArray8_32(tags, index, contents)
            # No return yet!

        elif tpe == pyarrow.string():
            assert tpe.num_buffers == 3
            mask = buffers.pop(0)
            offsets = ak.layout.Index32(
                numpy.frombuffer(buffers.pop(0), dtype=np.int32)
            )
            contents = ak.layout.NumpyArray(
                numpy.frombuffer(buffers.pop(0), dtype=np.uint8)
            )
            contents.setparameter("__array__", "char")
            out = ak.layout.ListOffsetArray32(offsets, contents)
            out.setparameter("__array__", "string")
            # No return yet!

        elif tpe == pyarrow.large_string():
            assert tpe.num_buffers == 3
            mask = buffers.pop(0)
            offsets = ak.layout.Index64(
                numpy.frombuffer(buffers.pop(0), dtype=np.int64)
            )
            contents = ak.layout.NumpyArray(
                numpy.frombuffer(buffers.pop(0), dtype=np.uint8)
            )
            contents.setparameter("__array__", "char")
            out = ak.layout.ListOffsetArray64(offsets, contents)
            out.setparameter("__array__", "string")
            # No return yet!

        elif tpe == pyarrow.binary():
            assert tpe.num_buffers == 3
            mask = buffers.pop(0)
            offsets = ak.layout.Index32(
                numpy.frombuffer(buffers.pop(0), dtype=np.int32)
            )
            contents = ak.layout.NumpyArray(
                numpy.frombuffer(buffers.pop(0), dtype=np.uint8)
            )
            contents.setparameter("__array__", "byte")
            out = ak.layout.ListOffsetArray32(offsets, contents)
            out.setparameter("__array__", "bytestring")
            # No return yet!

        elif tpe == pyarrow.large_binary():
            assert tpe.num_buffers == 3
            mask = buffers.pop(0)
            offsets = ak.layout.Index64(
                numpy.frombuffer(buffers.pop(0), dtype=np.int64)
            )
            contents = ak.layout.NumpyArray(
                numpy.frombuffer(buffers.pop(0), dtype=np.uint8)
            )
            contents.setparameter("__array__", "byte")
            out = ak.layout.ListOffsetArray64(offsets, contents)
            out.setparameter("__array__", "bytestring")
            # No return yet!

        elif tpe == pyarrow.bool_():
            assert tpe.num_buffers == 2
            mask = buffers.pop(0)
            data = buffers.pop(0)
            as_bytes = (
                numpy.unpackbits(numpy.frombuffer(data, dtype=np.uint8))
                .reshape(-1, 8)[:, ::-1]
                .reshape(-1)
            )
            out = ak.layout.NumpyArray(as_bytes.view(np.bool_))
            # No return yet!

        elif isinstance(tpe, pyarrow.lib.DataType) and tpe.num_buffers == 1:
            # This is a DataType(null)
            mask = buffers.pop(0)
            assert tpe.num_fields == 0
            assert mask is None
            out = ak.layout.IndexedOptionArray64(
                ak.layout.Index64(numpy.full(len(array), -1, dtype=np.int64)),
                ak.layout.EmptyArray(),
            )
            # No return yet!

        elif isinstance(tpe, pyarrow.lib.DataType):
            assert tpe.num_buffers == 2
            mask = buffers.pop(0)
            data = buffers.pop(0)

            to64, dt = _pyarrow_to_numpy_dtype.get(str(tpe), (False, None))
            if to64:
                data = numpy.frombuffer(data, dtype=np.int32).astype(np.int64)
            if dt is None:
                dt = tpe.to_pandas_dtype()
            out = ak.layout.NumpyArray(numpy.frombuffer(data, dtype=dt))
            # No return yet!

        else:
            raise TypeError(
                "unrecognized Arrow array type: {0}".format(repr(tpe))
                + ak._util.exception_suffix(__file__)
            )

        # All 'no return yet' cases need to become option-type (even if the UnmaskedArray
        # is just going to get stripped off in the recursive step that calls this one).
        if mask is not None:
            mask = ak.layout.IndexU8(numpy.frombuffer(mask, dtype=np.uint8))
            out = ak.layout.BitMaskedArray(mask, out, True, len(out), True)
        else:
            out = ak.layout.UnmaskedArray(out)

        # All 'no return yet' cases need to be corrected for pyarrow's 'offset'.
        if array.offset == 0 and len(array) == len(out):
            return out
        else:
            return out[array.offset : array.offset + len(array)]

    def handle_arrow(obj):
        if isinstance(obj, pyarrow.lib.Array):
            buffers = obj.buffers()
            out = popbuffers(obj, obj.type, buffers)
            assert len(buffers) == 0
            if isinstance(out, ak.layout.UnmaskedArray):
                return out.content
            else:
                return out

        elif isinstance(obj, pyarrow.lib.ChunkedArray):
            layouts = [handle_arrow(x) for x in obj.chunks if len(x) > 0]
            if all(isinstance(x, ak.layout.UnmaskedArray) for x in layouts):
                layouts = [x.content for x in layouts]
            if len(layouts) == 1:
                return layouts[0]
            else:
                return ak.operations.structure.concatenate(layouts, highlevel=False)

        elif isinstance(obj, pyarrow.lib.RecordBatch):
            child_array = []
            for i in range(obj.num_columns):
                layout = handle_arrow(obj.column(i))
                if obj.schema.field(i).nullable and not isinstance(
                    layout, ak._util.optiontypes
                ):
                    layout = ak.layout.UnmaskedArray(layout)
                child_array.append(layout)
            if pass_empty_field and list(obj.schema.names) == [""]:
                return child_array[0]
            else:
                return ak.layout.RecordArray(child_array, obj.schema.names)

        elif isinstance(obj, pyarrow.lib.Table):
            batches = obj.combine_chunks().to_batches()
            if len(batches) == 0:
                # zero-length array with the right type
                return from_buffers(_parquet_schema_to_form(obj.schema), 0, {})
            elif len(batches) == 1:
                return handle_arrow(batches[0])
            else:
                arrays = [handle_arrow(batch) for batch in batches if len(batch) > 0]
                return ak.operations.structure.concatenate(arrays, highlevel=False)

        elif (
            isinstance(obj, Iterable)
            and isinstance(obj, Sized)
            and len(obj) > 0
            and all(isinstance(x, pyarrow.lib.RecordBatch) for x in obj)
            and any(len(x) > 0 for x in obj)
        ):
            chunks = []
            for batch in obj:
                chunk = handle_arrow(batch)
                if len(chunk) > 0:
                    chunks.append(chunk)
            if len(chunks) == 1:
                return chunks[0]
            else:
                return ak.operations.structure.concatenate(chunks, highlevel=False)

        else:
            raise TypeError(
                "unrecognized Arrow type: {0}".format(type(obj))
                + ak._util.exception_suffix(__file__)
            )

    return ak._util.maybe_wrap(handle_arrow(array), behavior, highlevel)


def to_parquet(
    array,
    where,
    explode_records=False,
    list_to32=False,
    string_to32=True,
    bytestring_to32=True,
    **options  # NOTE: a comma after **options breaks Python 2
):
    """
    Args:
        array: Data to write to a Parquet file.
        where (str, Path, file-like object): Where to write the Parquet file.
        explode_records (bool): If True, lists of records are written as
            records of lists, so that nested fields become top-level fields
            (which can be zipped when read back).
        list_to32 (bool): If True, convert Awkward lists into 32-bit Arrow lists
            if they're small enough, even if it means an extra conversion. Otherwise,
            signed 32-bit #ak.layout.ListOffsetArray maps to Arrow `ListType` and
            all others map to Arrow `LargeListType`.
        string_to32 (bool): Same as the above for Arrow `string` and `large_string`.
        bytestring_to32 (bool): Same as the above for Arrow `binary` and `large_binary`.
        options: All other options are passed to pyarrow.parquet.ParquetWriter.
            In particular, if no `schema` is given, a schema is derived from
            the array type.

    Writes an Awkward Array to a Parquet file (through pyarrow).

        >>> array1 = ak.Array([[1, 2, 3], [], [4, 5], [], [], [6, 7, 8, 9]])
        >>> ak.to_parquet(array1, "array1.parquet")

    If the `array` does not contain records at top-level, the Arrow table will consist
    of one field whose name is `""`.

    Parquet files can maintain the distinction between "option-type but no elements are
    missing" and "not option-type" at all levels, including the top level. However,
    there is no distinction between `?union[X, Y, Z]]` type and `union[?X, ?Y, ?Z]` type.
    Be aware of these type distinctions when passing data through Arrow or Parquet.

    To make a partitioned Parquet dataset, use this function to write each Parquet
    file to a directory (as separate invocations, probably in parallel with multiple
    processes), then give them common metadata by calling `ak.to_parquet.dataset`.

        >>> ak.to_parquet(array1, "directory-name/file1.parquet")
        >>> ak.to_parquet(array2, "directory-name/file2.parquet")
        >>> ak.to_parquet(array3, "directory-name/file3.parquet")
        >>> ak.to_parquet.dataset("directory-name")

    Then all of the flies in the collection can be addressed as one array. For example,

        >>> dataset = ak.from_parquet("directory_name", lazy=True)

    (If it is large, you will likely want to load it lazily.)

    See also #ak.to_arrow, which is used as an intermediate step.
    See also #ak.from_parquet.
    """
    pyarrow = _import_pyarrow("ak.to_parquet")
    import pyarrow.parquet

    options["where"] = where

    def batch_iterator(layout):
        if isinstance(layout, ak.partition.PartitionedArray):
            for partition in layout.partitions:
                for x in batch_iterator(partition):
                    yield x

        else:
            if explode_records or isinstance(
                ak.operations.describe.type(layout), ak.types.RecordType
            ):
                names = layout.keys()
                contents = [layout[name] for name in names]
            else:
                names = [""]
                contents = [layout]

            pa_arrays = []
            pa_fields = []
            for name, content in zip(names, contents):
                pa_arrays.append(
                    to_arrow(
                        content,
                        list_to32=list_to32,
                        string_to32=string_to32,
                        bytestring_to32=bytestring_to32,
                        allow_tensor=False,
                    )
                )
                pa_fields.append(
                    pyarrow.field(name, pa_arrays[-1].type).with_nullable(
                        isinstance(
                            ak.operations.describe.type(content), ak.types.OptionType
                        )
                    )
                )
            yield pyarrow.RecordBatch.from_arrays(
                pa_arrays, schema=pyarrow.schema(pa_fields)
            )

    layout = to_layout(array, allow_record=False, allow_other=False)
    iterator = batch_iterator(layout)
    first = next(iterator)

    if "schema" not in options:
        options["schema"] = first.schema

    writer = pyarrow.parquet.ParquetWriter(**options)
    writer.write_table(pyarrow.Table.from_batches([first]))

    try:
        while True:
            try:
                record_batch = next(iterator)
            except StopIteration:
                break
            else:
                writer.write_table(pyarrow.Table.from_batches([record_batch]))
    finally:
        writer.close()


def _common_parquet_schema(pq, filenames, relpaths):
    assert len(filenames) != 0

    schema = None
    metadata_collector = []
    for filename, relpath in zip(filenames, relpaths):
        if schema is None:
            schema = pq.ParquetFile(filename).schema_arrow
            first_filename = filename
        elif schema != pq.ParquetFile(filename).schema_arrow:
            raise ValueError(
                "schema in {0} differs from the first schema (in {1})".format(
                    repr(filename), repr(first_filename)
                )
                + ak._util.exception_suffix(__file__)
            )
        metadata_collector.append(pq.read_metadata(filename))
        metadata_collector[-1].set_file_path(relpath)
    return schema, metadata_collector


def _to_parquet_dataset(directory, filenames=None, filename_extension=".parquet"):
    """
    Args:
        directory (str or Path): A local directory in which to write `_common_metadata`
            and `_metadata`, making the directory of Parquet files into a dataset.
        filenames (None or list of str or Path): If None, the `directory` will be
            recursively searched for files ending in `filename_extension` and
            sorted lexicographically. Otherwise, this explicit list of files is
            taken and row-groups are concatenated in its given order. If any
            filenames are relative, they are interpreted relative to `directory`.
        filename_extension (str): Filename extension (including `.`) to use to
            search for files recursively. Ignored if `filenames` is None.

    Creates a `_common_metadata` and a `_metadata` in a directory of Parquet files.

    The `_common_metadata` contains the schema that all files share. (If the files
    have different schemas, this function raises an exception.)

    The `_metadata` contains row-group metadata used to seek to specific row-groups
    within the multi-file dataset.
    """
    pyarrow = _import_pyarrow("ak.to_parquet.dataset")
    import pyarrow.parquet

    directory = _regularize_path(directory)
    if not os.path.isdir(directory):
        raise ValueError(
            "{0} is not a local filesystem directory".format(repr(directory))
            + ak._util.exception_suffix(__file__)
        )

    if filenames is None:
        filenames = sorted(
            glob.glob(directory + "/**/*{0}".format(filename_extension), recursive=True)
        )
    else:
        filenames = [_regularize_path(x) for x in filenames]
        filenames = [
            x if os.path.isabs(x) else os.path.join(directory, x) for x in filenames
        ]
    relpaths = [os.path.relpath(x, directory) for x in filenames]

    schema, metadata_collector = _common_parquet_schema(
        pyarrow.parquet, filenames, relpaths
    )
    pyarrow.parquet.write_metadata(schema, os.path.join(directory, "_common_metadata"))
    pyarrow.parquet.write_metadata(
        schema,
        os.path.join(directory, "_metadata"),
        metadata_collector=metadata_collector,
    )


to_parquet.dataset = _to_parquet_dataset


_from_parquet_key_number = 0
_from_parquet_key_lock = threading.Lock()


def _from_parquet_key():
    global _from_parquet_key_number
    with _from_parquet_key_lock:
        out = _from_parquet_key_number
        _from_parquet_key_number += 1
    return out


def _parquet_schema_to_form(schema):
    pyarrow = _import_pyarrow("ak.from_parquet")

    def lst(path):
        return "lst:" + ".".join(path)

    def col(path):
        return "col:" + ".".join(path)

    def maybe_nullable(field, content):
        if field.nullable:
            if isinstance(content, ak.forms.EmptyForm):
                return ak.forms.IndexedOptionForm(
                    "i64",
                    content.with_form_key(None),
                    form_key=content.form_key,
                )
            else:
                return ak.forms.ByteMaskedForm(
                    "i8",
                    content.with_form_key(None),
                    valid_when=True,
                    form_key=content.form_key,
                )
        else:
            return content

    def contains_record(form):
        if isinstance(form, ak.forms.RecordForm):
            return True
        elif isinstance(form, ak.forms.ListOffsetForm):
            return contains_record(form.content)
        else:
            return False

    def recurse(arrow_type, path):
        if isinstance(arrow_type, pyarrow.StructType):
            names = []
            contents = []
            for index in range(arrow_type.num_fields):
                field = arrow_type[index]
                names.append(field.name)
                content = maybe_nullable(
                    field, recurse(field.type, path + (field.name,))
                )
                contents.append(ak.forms.VirtualForm(content, has_length=True))
            assert len(contents) != 0
            return ak.forms.RecordForm(contents, names)

        elif isinstance(arrow_type, pyarrow.ListType):
            field = arrow_type.value_field
            content = maybe_nullable(
                field, recurse(field.type, path + ("list", "item"))
            )
            form_key = None if contains_record(content) else lst(path)
            return ak.forms.ListOffsetForm("i32", content, form_key=form_key)

        elif isinstance(arrow_type, pyarrow.LargeListType):
            field = arrow_type.value_field
            content = maybe_nullable(
                field, recurse(field.type, path + ("list", "item"))
            )
            form_key = None if contains_record(content) else lst(path)
            return ak.forms.ListOffsetForm("i64", content, form_key=form_key)

        elif arrow_type == pyarrow.string():
            return ak.forms.ListOffsetForm(
                "i32",
                ak.forms.NumpyForm((), 1, "B", parameters={"__array__": "char"}),
                parameters={"__array__": "string"},
                form_key=col(path),
            )

        elif arrow_type == pyarrow.large_string():
            return ak.forms.ListOffsetForm(
                "i64",
                ak.forms.NumpyForm((), 1, "B", parameters={"__array__": "char"}),
                parameters={"__array__": "string"},
                form_key=col(path),
            )

        elif arrow_type == pyarrow.binary():
            return ak.forms.ListOffsetForm(
                "i32",
                ak.forms.NumpyForm((), 1, "B", parameters={"__array__": "byte"}),
                parameters={"__array__": "bytestring"},
                form_key=col(path),
            )

        elif arrow_type == pyarrow.large_binary():
            return ak.forms.ListOffsetForm(
                "i64",
                ak.forms.NumpyForm((), 1, "B", parameters={"__array__": "byte"}),
                parameters={"__array__": "bytestring"},
                form_key=col(path),
            )

        elif isinstance(arrow_type, pyarrow.DataType):
            if arrow_type == pyarrow.null():
                return ak.forms.EmptyForm(form_key=col(path))
            else:
                dtype = np.dtype(arrow_type.to_pandas_dtype())
                return ak.forms.Form.from_numpy(dtype).with_form_key(col(path))

        else:
            raise NotImplementedError(
                "cannot convert {0}.{1} to an equivalent Awkward Form".format(
                    type(arrow_type).__module__, type(arrow_type).__name__
                )
                + ak._util.exception_suffix(__file__)
            )

    contents = []
    for index, name in enumerate(schema.names):
        field = schema.field(index)
        content = maybe_nullable(field, recurse(field.type, (name,)))
        contents.append(ak.forms.VirtualForm(content, has_length=True))
    assert len(contents) != 0
    return ak.forms.RecordForm(contents, schema.names)


class _LazyDatasetGenerator(object):
    def __init__(self, reader):
        self._dataset = reader

    def __call__(self, row_group, columns, length, form, lazy_cache, lazy_cache_key):
        if form.form_key is None:
            if isinstance(form, ak.forms.RecordForm):
                return self._create_record_array(
                    row_group, columns, length, form, lazy_cache, lazy_cache_key
                )

            elif isinstance(form, ak.forms.ListOffsetForm):
                return self._create_list_offset_array(
                    row_group, columns, form, lazy_cache, lazy_cache_key
                )

            else:
                raise AssertionError(
                    "unexpected Form: {0}".format(type(form))
                    + ak._util.exception_suffix(__file__)
                )

        else:
            assert form.form_key.startswith("col:") or form.form_key.startswith("lst:")
            column_name = form.form_key[4:]
            masked = isinstance(form, ak.forms.ByteMaskedForm)
            if masked:
                form = form.content
            table = self._dataset.read_row_group(row_group, [column_name])
            struct_only = [column_name.split(".")[-1]]
            struct_only.extend([x for x in columns[:0:-1] if x is not None])
            return self._convert_arrow_to_awkward(table, struct_only, masked, columns)

    def _convert_arrow_to_awkward(self, table, struct_only, masked, columns):
        out = _from_arrow(table, False, struct_only=struct_only, highlevel=False)
        for column in columns:
            if column is None:
                out = out.content
            else:
                out = out.field(column)
        if masked:
            if isinstance(out, (ak.layout.BitMaskedArray, ak.layout.UnmaskedArray)):
                out = out.toByteMaskedArray()
            elif isinstance(out, ak.layout.ListOffsetArray32) and isinstance(
                out.content, (ak.layout.BitMaskedArray, ak.layout.UnmaskedArray)
            ):
                out = ak.layout.ListOffsetArray32(
                    out.offsets, out.content.toByteMaskedArray()
                )
            elif isinstance(out, ak.layout.ListOffsetArray64) and isinstance(
                out.content, (ak.layout.BitMaskedArray, ak.layout.UnmaskedArray)
            ):
                out = ak.layout.ListOffsetArray64(
                    out.offsets, out.content.toByteMaskedArray()
                )
        return out

    def _create_record_array(
        self, row_group, columns, length, form, lazy_cache, lazy_cache_key
    ):
        contents = []
        recordlookup = []
        for i in range(form.numfields):
            name = form.key(i)
            subform = form.content(i).form
            generator = ak.layout.ArrayGenerator(
                self,
                (
                    row_group,
                    columns + (name,),
                    length,
                    subform,
                    lazy_cache,
                    lazy_cache_key,
                ),
                length=length,
                form=subform,
            )
            if subform.form_key is None:
                field_cache = None
                cache_key = None
            else:
                field_cache = lazy_cache
                cache_key = "{0}:{1}[{2}]".format(
                    lazy_cache_key, subform.form_key, row_group
                )
            contents.append(ak.layout.VirtualArray(generator, field_cache, cache_key))
            recordlookup.append(name)
        return ak.layout.RecordArray(contents, recordlookup, length)

    def _create_list_offset_array(
        self, row_group, columns, form, lazy_cache, lazy_cache_key
    ):
        struct_only = [x for x in columns[:0:-1] if x is not None]
        sampleform = self._find_first_column(form, struct_only)

        assert sampleform.form_key.startswith("col:") or sampleform.form_key.startswith(
            "lst:"
        )
        samplekey = "{0}:off:{1}:{2}[{3}]".format(
            lazy_cache_key,
            sampleform.form_key[4:],
            ".".join("" if x is None else x for x in columns),
            row_group,
        )
        sample = None
        if lazy_cache is not None:
            try:
                sample = lazy_cache.mutablemapping[samplekey]
            except KeyError:
                pass
        if sample is None:
            sample = self.get(row_group, columns, sampleform, struct_only)
        if lazy_cache is not None:
            lazy_cache.mutablemapping[samplekey] = sample

        offsets = [sample.offsets]
        sublength = offsets[-1][-1]
        sample = sample.content
        recordform = form.content
        columns = columns + (None,)
        while not isinstance(recordform, ak.forms.RecordForm):
            offsets.append(sample.offsets)
            sublength = offsets[-1][-1]
            sample = sample.content
            recordform = recordform.content
            columns = columns + (None,)

        out = self(
            row_group, columns, sublength, recordform, lazy_cache, lazy_cache_key
        )
        for off in offsets[::-1]:
            if isinstance(off, ak.layout.Index32):
                out = ak.layout.ListOffsetArray32(off, out)
            elif isinstance(off, ak.layout.Index64):
                out = ak.layout.ListOffsetArray64(off, out)
            else:
                raise AssertionError(
                    "unexpected Index type: {0}".format(off)
                    + ak._util.exception_suffix(__file__)
                )
        return out

    def _find_first_column(self, form, struct_only):
        if isinstance(form, ak.forms.VirtualForm):
            return self._find_first_column(form.form, struct_only)
        elif isinstance(form, ak.forms.RecordForm):
            assert form.numfields != 0
            struct_only.insert(0, form.key(0))
            return self._find_first_column(form.content(0), struct_only)
        elif isinstance(form, ak.forms.ListOffsetForm):
            if form.parameter("__array__") in ("string", "bytestring"):
                return form
            else:
                return self._find_first_column(form.content, struct_only)
        else:
            return form

    def get(self, row_group, unpack, form, struct_only):
        assert form.form_key.startswith("col:") or form.form_key.startswith("lst:")
        column_name = form.form_key[4:]
        masked = isinstance(form, ak.forms.ByteMaskedForm)
        if masked:
            form = form.content
        table = self._dataset.read_row_group(row_group, [column_name])
        return self._convert_arrow_to_awkward(table, struct_only, masked, unpack)


class _Dataset(object):
    def __init__(self, schema, row_groups, columns, partition_columns):
        self.schema = schema
        self.row_groups = row_groups
        self.columns = columns
        self.partition_columns = partition_columns

    @property
    def is_empty(self):
        return len(self.row_groups) == 0

    @property
    def row_group_metadata(self):
        raise NotImplementedError

    def read_row_group(self, row_group, columns=None):
        raise NotImplementedError

    def read_row_group_batches(self, row_groups=None, columns=None):
        if row_groups is None:
            row_groups = self.row_groups

        batches = []
        for i in row_groups:
            batches.extend(self.read_row_group(i, columns).to_batches())

        return batches


class _ParquetFileDataset(_Dataset):
    def __init__(
        self, source, row_groups=None, columns=None, use_threads=True, options=None
    ):
        import pyarrow.parquet

        if options is None:
            options = {}
        self._file = pyarrow.parquet.ParquetFile(source, **options)

        if row_groups is None:
            row_groups = range(self._file.num_row_groups)

        schema = self._file.schema_arrow
        if columns is None:
            columns = schema.names
        schema = _partial_schema_from_columns(schema, columns)

        self._use_threads = use_threads

        super(_ParquetFileDataset, self).__init__(
            schema, row_groups, columns, partition_columns=[]
        )

    @property
    def row_group_metadata(self):
        return [self._file.metadata.row_group(i) for i in self.row_groups]

    def read_row_group(self, row_group, columns=None):
        if columns is None:
            columns = self.columns

        return self._file.read_row_group(
            row_group, columns=columns, use_threads=self._use_threads
        )


class _ParquetDataset(_Dataset):
    def __init__(
        self,
        directory,
        metadata_filename,
        row_groups=None,
        columns=None,
        use_threads=True,
        include_partition_columns=True,
        options=None,
    ):
        import pyarrow.parquet

        if options is None:
            options = {}

        self._metadata_file = pyarrow.parquet.ParquetFile(metadata_filename, **options)

        if row_groups is None:
            row_groups = range(self._metadata_file.num_row_groups)

        schema = self._metadata_file.schema_arrow
        if columns is None:
            columns = schema.names
        schema = _partial_schema_from_columns(schema, columns)

        if include_partition_columns:
            paths_and_counts = self._get_paths_and_counts(
                self._metadata_file, row_groups
            )
            partition_columns = _parquet_partitions_to_awkward(paths_and_counts)
        else:
            partition_columns = []

        self._use_threads = use_threads
        self._lookup = self._build_row_group_lookup(directory, self._metadata_file)
        self._options = options

        super(_ParquetDataset, self).__init__(
            schema, row_groups, columns, partition_columns
        )

    @staticmethod
    def _build_row_group_lookup(directory, metadata_file):
        last_filename = None
        lookup = []
        for i in range(metadata_file.num_row_groups):
            filename = metadata_file.metadata.row_group(i).column(0).file_path
            if last_filename is None:
                if filename == "":
                    raise ValueError(
                        "Parquet _metadata file does not contain file paths "
                        "(e.g. was not made with 'set_file_path')"
                        + ak._util.exception_suffix(__file__)
                    )
                last_filename = filename
                start_i = 0
            elif filename != "" and last_filename != filename:
                last_filename = filename
                start_i = i
            lookup.append((os.path.join(directory, last_filename), i - start_i))
        return lookup

    @staticmethod
    def _get_paths_and_counts(file, row_groups):
        paths_and_counts = []
        last_filename = None
        for i in row_groups:
            filename = file.metadata.row_group(i).column(0).file_path
            if last_filename is None:
                if filename == "":
                    raise ValueError(
                        "Parquet _metadata file does not contain file paths "
                        "(e.g. was not made with 'set_file_path')"
                        + ak._util.exception_suffix(__file__)
                    )
                last_filename = filename
                paths_and_counts.append([filename, 0])
            elif filename != "" and last_filename != filename:
                last_filename = filename
                paths_and_counts.append([filename, 0])
            paths_and_counts[-1][-1] += file.metadata.row_group(i).num_rows
        return paths_and_counts

    @property
    def row_group_metadata(self):
        return [self._metadata_file.metadata.row_group(i) for i in self.row_groups]

    def read_row_group(self, row_group, columns=None):
        if columns is None:
            columns = self.columns

        import pyarrow.parquet

        file_path, local_row_group = self._lookup[row_group]
        local_file = pyarrow.parquet.ParquetFile(file_path, **self._options)

        return local_file.read_row_group(
            local_row_group, columns, use_threads=self._use_threads
        )


class _ParquetMultiFileDataset(_Dataset):
    def __init__(
        self,
        source,
        relative_to,
        row_groups=None,
        columns=None,
        use_threads=True,
        include_partition_columns=True,
        options=None,
    ):
        schema, lookup, paths_and_counts = self._get_dataset_metadata(
            source, relative_to, options
        )
        if row_groups is None:
            row_groups = range(len(lookup))

        if include_partition_columns:
            partition_columns = _parquet_partitions_to_awkward(paths_and_counts)
        else:
            partition_columns = []

        if columns is None:
            columns = schema.names
        schema = _partial_schema_from_columns(schema, columns)

        self._lookup = lookup
        self._use_threads = use_threads
        super(_ParquetMultiFileDataset, self).__init__(
            schema, row_groups, columns, partition_columns
        )

    @staticmethod
    def _get_dataset_metadata(source, relative_to, options):
        import pyarrow.parquet

        schema = None
        lookup = []
        paths_and_counts = []
        for filename in source:
            single_file = pyarrow.parquet.ParquetFile(filename, **options)
            if schema is None:
                schema = single_file.schema_arrow
                first_filename = filename
            elif schema != single_file.schema_arrow:
                raise ValueError(
                    "schema in {0} differs from the first schema (in {1})".format(
                        repr(filename), repr(first_filename)
                    )
                    + ak._util.exception_suffix(__file__)
                )
            for i in range(single_file.num_row_groups):
                lookup.append((single_file, i))
            paths_and_counts.append(
                (os.path.relpath(filename, relative_to), single_file.metadata.num_rows)
            )
        return schema, lookup, paths_and_counts

    @property
    def row_group_metadata(self):
        return [
            f.metadata.row_group(g)
            for f, g in (self._lookup[g] for g in self.row_groups)
        ]

    def read_row_group(self, row_group, columns=None):
        if columns is None:
            columns = self.columns

        single_file, local_row_group = self._lookup[row_group]

        return single_file.read_row_group(
            local_row_group, columns, use_threads=self._use_threads
        )


def _parquet_partition_values(path):
    dirname, filename = os.path.split(path)
    m = re.match("([^=]+)=([^=]*)$", filename)
    if m is None:
        pair = ()
    else:
        pair = (m.groups(),)
    if dirname == "" or dirname == "/":
        return pair
    else:
        return _parquet_partition_values(dirname) + pair


def _parquet_partitions_to_awkward(paths_and_counts):
    path, count = paths_and_counts[0]
    columns = [column for column, value in _parquet_partition_values(path)]
    values = [[] for _ in columns]
    indexes = [[] for _ in columns]
    for path, count in paths_and_counts:
        for i, (column, value) in enumerate(_parquet_partition_values(path)):
            if i >= len(columns) or column != columns[i]:
                raise ValueError(
                    "inconsistent partition column names in Parquet directory paths"
                    + ak._util.exception_suffix(__file__)
                )
            try:
                j = values[i].index(value)
            except ValueError:
                j = len(values[i])
                values[i].append(value)
            indexes[i].append(numpy.full(count, j, np.int32))
    indexedarrays = []
    for column, vals, idx in zip(columns, values, indexes):
        indexedarrays.append(
            (
                column,
                ak.layout.IndexedArray32(
                    ak.layout.Index32(numpy.concatenate(idx)),
                    ak.operations.convert.from_iter(vals, highlevel=False),
                ),
            )
        )
    return indexedarrays


def _partial_schema_from_columns(schema, columns):
    pyarrow = _import_pyarrow("ak.from_parquet")
    pa_fields = []
    for x in columns:
        if x not in schema.names:
            raise ValueError(
                "column {0} not found in schema".format(repr(x))
                + ak._util.exception_suffix(__file__)
            )
        pa_fields.append(schema.field(x))
    return pyarrow.schema(pa_fields)


def _create_partitioned_array_from_form(
    form,
    state,
    row_groups,
    lengths,
    columns,
    all_columns,
    partition_columns,
    lazy_cache,
    lazy_cache_key,
):
    partitions = []
    offsets = [0]
    for length_index, row_group in enumerate(row_groups):
        length = lengths[length_index]
        offsets.append(offsets[-1] + length)

        contents = []
        recordlookup = []
        for column in columns:
            subform = form.contents[column].form
            generator = ak.layout.ArrayGenerator(
                state,
                (
                    row_group,
                    (column,),
                    length,
                    subform,
                    lazy_cache,
                    lazy_cache_key,
                ),
                length=length,
                form=subform,
            )
            if subform.form_key is None:
                field_cache = None
                cache_key = None
            else:
                field_cache = lazy_cache
                cache_key = "{0}:{1}[{2}]".format(
                    lazy_cache_key, subform.form_key, row_group
                )
            contents.append(ak.layout.VirtualArray(generator, field_cache, cache_key))
            recordlookup.append(column)

        if partition_columns == [] and all_columns == [""]:
            partitions.append(contents[0])
        else:
            if partition_columns == []:
                field_names = recordlookup
                fields = contents
            else:
                start, stop = offsets[row_group], offsets[row_group + 1]
                field_names = [x[0] for x in partition_columns] + recordlookup
                fields = [x[1][start:stop] for x in partition_columns] + contents
            recordarray = ak.layout.RecordArray(fields, field_names, length)
            partitions.append(recordarray)

    if len(partitions) == 1:
        return partitions[0]
    else:
        return ak.partition.IrregularlyPartitionedArray(partitions, offsets[1:])


def _regularize_lazy_cache(lazy_cache):
    hold_cache = None
    if lazy_cache is None:
        pass
    # Wrap existing ArrayCache's internal cache
    elif isinstance(lazy_cache, ak.layout.ArrayCache):
        pass
    else:
        if lazy_cache == "new":
            hold_cache = ak._util.MappingProxy({})
        elif lazy_cache == "attach":
            raise TypeError("lazy_cache must be a MutableMapping")
        else:
            hold_cache = ak._util.MappingProxy.maybe_wrap(lazy_cache)
            if not isinstance(hold_cache, MutableMapping):
                raise TypeError("lazy_cache must be a MutableMapping")
        lazy_cache = ak.layout.ArrayCache(hold_cache)
    return lazy_cache, hold_cache


def _regularize_parquet_lazy_cache_key(lazy_cache_key):
    if lazy_cache_key is None:
        lazy_cache_key = "ak.from_parquet:{0}".format(_from_parquet_key())
    return lazy_cache_key


def from_parquet(
    source,
    columns=None,
    row_groups=None,
    use_threads=True,
    include_partition_columns=True,
    lazy=False,
    lazy_cache="new",
    lazy_cache_key=None,
    highlevel=True,
    behavior=None,
    **options  # NOTE: a comma after **options breaks Python 2
):
    """
    Args:
        source (str, Path, file-like object, pyarrow.NativeFile): Where to
            get the Parquet file. If `source` is the name of a local directory
            (str or Path), then it is interpreted as a partitioned Parquet dataset.
        columns (None or list of str): If None, read all columns; otherwise,
            read a specified set of columns.
        row_groups (None, int, or list of int): If None, read all row groups;
            otherwise, read a single or list of row groups.
        use_threads (bool): Passed to the pyarrow.parquet.ParquetFile.read
            functions; if True, do multithreaded reading.
        include_partition_columns (bool): If True and `source` is a partitioned
            Parquet dataset with subdirectory names defining partition names
            and values, include those special columns in the output.
        lazy (bool): If True, read columns in row groups on demand (as
            #ak.layout.VirtualArray, possibly in #ak.partition.PartitionedArray
            if the file has more than one row group); if False, read all
            requested data immediately.
        lazy_cache (None, "new", or MutableMapping): If lazy, pass this
            cache to the VirtualArrays. If "new", a new dict (keep-forever cache)
            is created. If None, no cache is used.
        lazy_cache_key (None or str): If lazy, pass this cache_key to the
            VirtualArrays. If None, a process-unique string is constructed.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        options: All other options are passed to pyarrow.parquet.ParquetFile.

    Reads a Parquet file into an Awkward Array (through pyarrow).

        >>> ak.from_parquet("array1.parquet")
        <Array [[1, 2, 3], [], ... [], [6, 7, 8, 9]] type='6 * var * ?int64'>

    See also #ak.from_arrow, which is used as an intermediate step.
    See also #ak.to_parquet.
    """
    _import_pyarrow("ak.from_parquet")

    if isinstance(row_groups, (numbers.Integral, np.integer)):
        row_groups = [row_groups]

    source = _regularize_path(source)

    if isinstance(source, str) and os.path.isdir(source):
        metadata_filename = os.path.join(source, "_metadata")
        if os.path.exists(metadata_filename):
            dataset = _ParquetDataset(
                source,
                metadata_filename,
                row_groups,
                columns,
                use_threads,
                include_partition_columns,
                options,
            )
        else:
            relative_to = source
            source = [
                _regularize_path(x)
                for x in sorted(glob.glob(source + "/**/*.parquet", recursive=True))
            ]
            dataset = _ParquetMultiFileDataset(
                source,
                relative_to,
                row_groups,
                columns,
                use_threads,
                include_partition_columns,
                options,
            )

    elif (
        not isinstance(source, str)
        and isinstance(source, Iterable)
        and not hasattr(source, "read")
    ):
        source = [_regularize_path(x) for x in source]
        relative_to = os.path.commonpath(source)
        dataset = _ParquetMultiFileDataset(
            source,
            relative_to,
            row_groups,
            columns,
            use_threads,
            include_partition_columns,
            options,
        )

    else:
        dataset = _ParquetFileDataset(source, row_groups, columns, use_threads, options)

    if dataset.is_empty:
        return ak.layout.RecordArray(
            [ak.layout.EmptyArray() for _ in dataset.columns], dataset.columns, 0
        )

    if lazy:
        lazy_cache, hold_cache = _regularize_lazy_cache(lazy_cache)
        lazy_cache_key = _regularize_parquet_lazy_cache_key(lazy_cache_key)

        lengths = [r.num_rows for r in dataset.row_group_metadata]
        state = _LazyDatasetGenerator(dataset)

        form = _parquet_schema_to_form(dataset.schema)
        out = _create_partitioned_array_from_form(
            form,
            state,
            dataset.row_groups,
            lengths,
            dataset.columns,
            dataset.schema.names,
            dataset.partition_columns,
            lazy_cache,
            lazy_cache_key,
        )

    else:
        batches = dataset.read_row_group_batches()
        out = _from_arrow(batches, False, highlevel=False)
        assert isinstance(out, ak.layout.RecordArray) and not out.istuple

        if dataset.partition_columns != []:
            field_names, fields = zip(*dataset.partition_columns)
            out = ak.layout.RecordArray(
                fields + tuple(out.contents), field_names + tuple(out.keys())
            )

        elif dataset.schema.names == [""]:
            out = out[""]

    return ak._util.maybe_wrap(out, behavior, highlevel)


def to_buffers(
    array,
    container=None,
    partition_start=0,
    form_key="node{id}",
    key_format="part{partition}-{form_key}-{attribute}",
    virtual="materialize",
):
    """
    Args:
        array: Data to decompose into named buffers.
        container (None or MutableMapping): The str \u2192 NumPy arrays (or
            Python buffers) that represent the decomposed Awkward Array. This
            `container` is only assumed to have a `__setitem__` method that
            accepts strings as keys.
        partition_start (non-negative int): If `array` is not partitioned, this is
            the partition number that will be used as part of the container
            key. If `array` is partitioned, this is the first partition number.
        form_key (str, callable): Python format string containing
            `"{id}"` or a function that takes non-negative integer as a string
            and the current `layout` as keyword arguments and returns a string,
            for use as a `form_key` on each Form node and in `key_format` (below).
        key_format (str or callable): Python format string containing
            `"{partition}"`, `"{form_key}"`, and/or `"{attribute}"` or a function
            that takes these as keyword arguments and returns a string to use
            as keys for buffers in the `container`. The `partition` is a
            partition number (non-negative integer, passed as a string), the
            `form_key` is the result of applying `form_key` (above), and the
            `attribute` is a hard-coded string representing the buffer's function
            (e.g. `"data"`, `"offsets"`, `"index"`).
        virtual (str): If `"materialize"`, any virtual arrays will be materialized
            and the materialized data will be included in the container. If `"pass"`,
            a virtual array's Form is passed through as a #ak.forms.VirtualForm,
            assuming that it contains `form_keys` that can be found in the
            container (e.g. by a previous pass through this function). No other
            values are allowed for this function argument.

    Decomposes an Awkward Array into a Form and a collection of memory buffers,
    so that data can be losslessly written to file formats and storage devices
    that only map names to binary blobs (such as a filesystem directory).

    This function returns a 3-tuple:

        (form, length, container)

    where the `form` is a #ak.forms.Form (which can be converted to JSON
    with `tojson`), the `length` is either an integer (`len(array)`) or a list
    of the lengths of each partition in `array`, and the `container` is either
    the MutableMapping you passed in or a new dict containing the buffers (as
    NumPy arrays).

    These are also the first three arguments of #ak.from_buffers, so a full
    round-trip is

        >>> reconstituted = ak.from_buffers(*ak.to_buffers(original))

    The `container` argument lets you specify your own MutableMapping, which
    might be an interface to some storage format or device (e.g. h5py). It's
    okay if the `container` drops NumPy's `dtype` and `shape` information,
    leaving raw bytes, since `dtype` and `shape` can be reconstituted from
    the #ak.forms.NumpyForm.

    The `partition_start` argument lets you fill the `container` gradually or
    in parallel. If the `array` is not partitioned, the `partition_start`
    argument sets its partition number (for the container keys, through
    `key_format`). If the `array` is partitioned, the first partition is numbered
    `partition_start` and as many are filled as ar in `array`. See #ak.partitions
    to get the number of partitions in `array`.

    Here is a simple example:

        >>> original = ak.Array([[1, 2, 3], [], [4, 5]])
        >>> form, length, container = ak.to_buffers(original)
        >>> form
        {
            "class": "ListOffsetArray64",
            "offsets": "i64",
            "content": {
                "class": "NumpyArray",
                "itemsize": 8,
                "format": "l",
                "primitive": "int64",
                "form_key": "node1"
            },
            "form_key": "node0"
        }
        >>> length
        3
        >>> container
        {'part0-node0-offsets': array([0, 3, 3, 5], dtype=int64),
         'part0-node1-data': array([1, 2, 3, 4, 5])}

    which may be read back with

        >>> ak.from_buffers(form, length, container)
        <Array [[1, 2, 3], [], [4, 5]] type='3 * var * int64'>

    Here is an example that builds up a partitioned array:

        >>> container = {}
        >>> lengths = []
        >>> form, length, _ = ak.to_buffers(ak.Array([[1, 2, 3], [], [4, 5]]), container, 0)
        >>> lengths.append(length)
        >>> form, length, _ = ak.to_buffers(ak.Array([[6, 7, 8, 9]]), container, 1)
        >>> lengths.append(length)
        >>> form, length, _ = ak.to_buffers(ak.Array([[], [], []]), container, 2)
        >>> lengths.append(length)
        >>> form, length, _ = ak.to_buffers(ak.Array([[10]]), container, 3)
        >>> lengths.append(length)
        >>> form
        {
            "class": "ListOffsetArray64",
            "offsets": "i64",
            "content": {
                "class": "NumpyArray",
                "itemsize": 8,
                "format": "l",
                "primitive": "int64",
                "form_key": "node1"
            },
            "form_key": "node0"
        }
        >>> lengths
        [3, 1, 3, 1]
        >>> container
        {'part0-node0-offsets': array([0, 3, 3, 5], dtype=int64),
         'part0-node1-data': array([1, 2, 3, 4, 5]),
         'part1-node0-offsets': array([0, 4], dtype=int64),
         'part1-node1-data': array([6, 7, 8, 9]),
         'part2-node0-offsets': array([0, 0, 0, 0], dtype=int64),
         'part2-node1-data': array([], dtype=float64),
         'part3-node0-offsets': array([0, 1], dtype=int64),
         'part3-node1-data': array([10])}

    The object returned by #ak.from_buffers is now a partitioned array:

        >>> reconstituted = ak.from_buffers(form, lengths, container)
        >>> reconstituted
        <Array [[1, 2, 3], [], [4, ... [], [], [10]] type='8 * var * int64'>
        >>> ak.partitions(reconstituted)
        [3, 1, 3, 1]

    If you intend to use this function for saving data, you may want to pack it
    first with #ak.packed.

    See also #ak.from_buffers and #ak.packed.
    """
    if container is None:
        container = {}

    def index_form(index):
        if isinstance(index, ak.layout.Index64):
            return "i64"
        elif isinstance(index, ak.layout.Index32):
            return "i32"
        elif isinstance(index, ak.layout.IndexU32):
            return "u32"
        elif isinstance(index, ak.layout.Index8):
            return "i8"
        elif isinstance(index, ak.layout.IndexU8):
            return "u8"
        else:
            raise AssertionError(
                "unrecognized index: "
                + repr(index)
                + ak._util.exception_suffix(__file__)
            )

    if isinstance(form_key, str):

        def generate_form_key(form_key):
            def fk(**v):
                return form_key.format(**v)

            return fk

        form_key = generate_form_key(form_key)

    if isinstance(key_format, str):

        def generate_key_format(key_format):
            def kf(**v):
                return key_format.format(**v)

            return kf

        key_format = generate_key_format(key_format)

    num_form_keys = [0]

    def little_endian(array):
        return array.astype(array.dtype.newbyteorder("<"), copy=False)

    def fill(layout, part):
        has_identities = layout.identities is not None
        parameters = layout.parameters
        key_index = num_form_keys[0]
        num_form_keys[0] += 1

        if has_identities:
            raise NotImplementedError(
                "ak.to_buffers for an array with Identities"
                + ak._util.exception_suffix(__file__)
            )

        if isinstance(layout, ak.layout.EmptyArray):
            fk = form_key(id=str(key_index))
            key = key_format(form_key=fk, attribute="data", partition=str(part))
            container[key] = little_endian(numpy.asarray(layout))
            return ak.forms.EmptyForm(has_identities, parameters, fk)

        elif isinstance(
            layout,
            (
                ak.layout.IndexedArray32,
                ak.layout.IndexedArrayU32,
                ak.layout.IndexedArray64,
            ),
        ):
            fk = form_key(id=str(key_index), layout=layout)
            key = key_format(form_key=fk, attribute="index", partition=str(part))
            container[key] = little_endian(numpy.asarray(layout.index))
            return ak.forms.IndexedForm(
                index_form(layout.index),
                fill(layout.content, part),
                has_identities,
                parameters,
                fk,
            )

        elif isinstance(
            layout, (ak.layout.IndexedOptionArray32, ak.layout.IndexedOptionArray64)
        ):
            fk = form_key(id=str(key_index), layout=layout)
            key = key_format(form_key=fk, attribute="index", partition=str(part))
            container[key] = little_endian(numpy.asarray(layout.index))
            return ak.forms.IndexedOptionForm(
                index_form(layout.index),
                fill(layout.content, part),
                has_identities,
                parameters,
                fk,
            )

        elif isinstance(layout, ak.layout.ByteMaskedArray):
            fk = form_key(id=str(key_index), layout=layout)
            key = key_format(form_key=fk, attribute="mask", partition=str(part))
            container[key] = little_endian(numpy.asarray(layout.mask))
            return ak.forms.ByteMaskedForm(
                index_form(layout.mask),
                fill(layout.content, part),
                layout.valid_when,
                has_identities,
                parameters,
                fk,
            )

        elif isinstance(layout, ak.layout.BitMaskedArray):
            fk = form_key(id=str(key_index), layout=layout)
            key = key_format(form_key=fk, attribute="mask", partition=str(part))
            container[key] = little_endian(numpy.asarray(layout.mask))
            return ak.forms.BitMaskedForm(
                index_form(layout.mask),
                fill(layout.content, part),
                layout.valid_when,
                layout.lsb_order,
                has_identities,
                parameters,
                fk,
            )

        elif isinstance(layout, ak.layout.UnmaskedArray):
            return ak.forms.UnmaskedForm(
                fill(layout.content, part),
                has_identities,
                parameters,
                form_key(id=str(key_index), layout=layout),
            )

        elif isinstance(
            layout,
            (ak.layout.ListArray32, ak.layout.ListArrayU32, ak.layout.ListArray64),
        ):
            fk = form_key(id=str(key_index), layout=layout)
            key = key_format(form_key=fk, attribute="starts", partition=str(part))
            container[key] = little_endian(numpy.asarray(layout.starts))
            key = key_format(form_key=fk, attribute="stops", partition=str(part))
            container[key] = little_endian(numpy.asarray(layout.stops))
            return ak.forms.ListForm(
                index_form(layout.starts),
                index_form(layout.stops),
                fill(layout.content, part),
                has_identities,
                parameters,
                fk,
            )

        elif isinstance(
            layout,
            (
                ak.layout.ListOffsetArray32,
                ak.layout.ListOffsetArrayU32,
                ak.layout.ListOffsetArray64,
            ),
        ):
            fk = form_key(id=str(key_index), layout=layout)
            key = key_format(form_key=fk, attribute="offsets", partition=str(part))
            container[key] = little_endian(numpy.asarray(layout.offsets))
            return ak.forms.ListOffsetForm(
                index_form(layout.offsets),
                fill(layout.content, part),
                has_identities,
                parameters,
                fk,
            )

        elif isinstance(layout, ak.layout.NumpyArray):
            fk = form_key(id=str(key_index), layout=layout)
            key = key_format(form_key=fk, attribute="data", partition=str(part))
            array = numpy.asarray(layout)
            container[key] = little_endian(array)
            form = ak.forms.Form.from_numpy(array.dtype)
            return ak.forms.NumpyForm(
                layout.shape[1:],
                form.itemsize,
                form.format,
                has_identities,
                parameters,
                fk,
            )

        elif isinstance(layout, ak.layout.RecordArray):
            if layout.istuple:
                forms = [fill(x, part) for x in layout.contents]
                keys = None
            else:
                forms = []
                keys = []
                for k in layout.keys():
                    forms.append(fill(layout[k], part))
                    keys.append(k)

            return ak.forms.RecordForm(
                forms,
                keys,
                has_identities,
                parameters,
                form_key(id=str(key_index), layout=layout),
            )

        elif isinstance(layout, ak.layout.RegularArray):
            return ak.forms.RegularForm(
                fill(layout.content, part),
                layout.size,
                has_identities,
                parameters,
                form_key(id=str(key_index), layout=layout),
            )

        elif isinstance(
            layout,
            (
                ak.layout.UnionArray8_32,
                ak.layout.UnionArray8_U32,
                ak.layout.UnionArray8_64,
            ),
        ):
            forms = []
            for x in layout.contents:
                forms.append(fill(x, part))

            fk = form_key(id=str(key_index), layout=layout)
            key = key_format(form_key=fk, attribute="tags", partition=str(part))
            container[key] = little_endian(numpy.asarray(layout.tags))
            key = key_format(form_key=fk, attribute="index", partition=str(part))
            container[key] = little_endian(numpy.asarray(layout.index))
            return ak.forms.UnionForm(
                index_form(layout.tags),
                index_form(layout.index),
                forms,
                has_identities,
                parameters,
                fk,
            )

        elif isinstance(layout, ak.layout.VirtualArray):
            if virtual == "materialize":
                return fill(layout.array, part)
            elif virtual == "pass":
                return ak.forms.VirtualForm(
                    layout.form,
                    layout.generator.length is not None,
                    layout.identities is not None,
                    layout.parameters,
                    None,
                )
            else:
                raise ValueError(
                    "unrecognized value for 'virtual': "
                    + str(virtual)
                    + ak._util.exception_suffix(__file__)
                )

        else:
            raise AssertionError(
                "unrecognized layout node type: "
                + str(type(layout))
                + ak._util.exception_suffix(__file__)
            )

    layout = to_layout(array, allow_record=False, allow_other=False)

    if isinstance(layout, ak.partition.PartitionedArray):
        form = None
        length = []
        for part, content in enumerate(layout.partitions):
            num_form_keys[0] = 0

            f = fill(content, partition_start + part)

            if form is None:
                form = f
            elif form != f:
                raise ValueError(
                    """the Form of partition {0}:

    {1}

differs from the first Form:

    {2}""".format(
                        partition_start + part,
                        f.tojson(True, False),
                        form.tojson(True, False),
                    )
                    + ak._util.exception_suffix(__file__)
                )
            length.append(len(content))

    else:
        form = fill(layout, partition_start)
        length = len(layout)

    return form, length, container


_index_form_to_dtype = _index_form_to_index = _form_to_layout_class = None


def _asbuf(obj):
    try:
        tmp = numpy.asarray(obj)
    except Exception:
        return numpy.frombuffer(obj, np.uint8)
    else:
        return tmp.reshape(-1).view(np.uint8)


def _form_to_layout(
    form,
    container,
    partnum,
    key_format,
    length,
    lazy_cache,
    lazy_cache_key,
):
    global _index_form_to_dtype, _index_form_to_index, _form_to_layout_class

    if _index_form_to_dtype is None:
        _index_form_to_dtype = {
            "i8": np.dtype("<i1"),
            "u8": np.dtype("<u1"),
            "i32": np.dtype("<i4"),
            "u32": np.dtype("<u4"),
            "i64": np.dtype("<i8"),
        }

        _index_form_to_index = {
            "i8": ak.layout.Index8,
            "u8": ak.layout.IndexU8,
            "i32": ak.layout.Index32,
            "u32": ak.layout.IndexU32,
            "i64": ak.layout.Index64,
        }

        _form_to_layout_class = {
            (ak.forms.IndexedForm, "i32"): ak.layout.IndexedArray32,
            (ak.forms.IndexedForm, "u32"): ak.layout.IndexedArrayU32,
            (ak.forms.IndexedForm, "i64"): ak.layout.IndexedArray64,
            (ak.forms.IndexedOptionForm, "i32"): ak.layout.IndexedOptionArray32,
            (ak.forms.IndexedOptionForm, "i64"): ak.layout.IndexedOptionArray64,
            (ak.forms.ListForm, "i32"): ak.layout.ListArray32,
            (ak.forms.ListForm, "u32"): ak.layout.ListArrayU32,
            (ak.forms.ListForm, "i64"): ak.layout.ListArray64,
            (ak.forms.ListOffsetForm, "i32"): ak.layout.ListOffsetArray32,
            (ak.forms.ListOffsetForm, "u32"): ak.layout.ListOffsetArrayU32,
            (ak.forms.ListOffsetForm, "i64"): ak.layout.ListOffsetArray64,
            (ak.forms.UnionForm, "i32"): ak.layout.UnionArray8_32,
            (ak.forms.UnionForm, "u32"): ak.layout.UnionArray8_U32,
            (ak.forms.UnionForm, "i64"): ak.layout.UnionArray8_64,
        }

    if form.has_identities:
        raise NotImplementedError(
            "ak.from_buffers for an array with Identities"
            + ak._util.exception_suffix(__file__)
        )
    else:
        identities = None

    parameters = form.parameters
    fk = form.form_key

    if isinstance(form, ak.forms.BitMaskedForm):
        raw_mask = _asbuf(
            container[key_format(form_key=fk, attribute="mask", partition=partnum)]
        )
        mask = _index_form_to_index[form.mask](
            raw_mask.view(_index_form_to_dtype[form.mask])
        )

        content = _form_to_layout(
            form.content,
            container,
            partnum,
            key_format,
            length,
            lazy_cache,
            lazy_cache_key,
        )

        if length is None:
            length = len(content)
        if length > len(mask) * 8:
            raise ValueError(
                "mask is too short for BitMaskedArray: content length "
                "is {0}, mask length * 8 is {1}".format(length, len(mask) * 8)
                + ak._util.exception_suffix(__file__)
            )

        return ak.layout.BitMaskedArray(
            mask,
            content,
            form.valid_when,
            length,
            form.lsb_order,
            identities,
            parameters,
        )

    elif isinstance(form, ak.forms.ByteMaskedForm):
        raw_mask = _asbuf(
            container[key_format(form_key=fk, attribute="mask", partition=partnum)]
        )
        mask = _index_form_to_index[form.mask](
            raw_mask.view(_index_form_to_dtype[form.mask])
        )

        if length is None:
            length = len(mask)
        elif length > len(mask):
            raise ValueError(
                "mask is too short for ByteMaskedArray: expected {0}, mask length is {1}".format(
                    length, len(mask)
                )
                + ak._util.exception_suffix(__file__)
            )

        content = _form_to_layout(
            form.content,
            container,
            partnum,
            key_format,
            length,
            lazy_cache,
            lazy_cache_key,
        )

        return ak.layout.ByteMaskedArray(
            mask, content, form.valid_when, identities, parameters
        )

    elif isinstance(form, ak.forms.EmptyForm):
        if length is not None and length != 0:
            raise ValueError(
                "EmptyArray found in node with non-zero expected length: expected {0}".format(
                    length
                )
                + ak._util.exception_suffix(__file__)
            )
        return ak.layout.EmptyArray(identities, parameters)

    elif isinstance(form, ak.forms.IndexedForm):
        raw_index = _asbuf(
            container[key_format(form_key=fk, attribute="index", partition=partnum)]
        )
        index = _index_form_to_index[form.index](
            raw_index.view(_index_form_to_dtype[form.index])
        )

        if length is None:
            length = len(index)
        elif length > len(index):
            raise ValueError(
                "index too short for IndexedArray: expected {0}, index length is {1}".format(
                    length, len(index)
                )
                + ak._util.exception_suffix(__file__)
            )

        content = _form_to_layout(
            form.content,
            container,
            partnum,
            key_format,
            0 if len(index) == 0 else numpy.max(index) + 1,
            lazy_cache,
            lazy_cache_key,
        )

        return _form_to_layout_class[type(form), form.index](
            index, content, identities, parameters
        )

    elif isinstance(form, ak.forms.IndexedOptionForm):
        raw_index = _asbuf(
            container[key_format(form_key=fk, attribute="index", partition=partnum)]
        )
        index = _index_form_to_index[form.index](
            raw_index.view(_index_form_to_dtype[form.index])
        )

        if length is None:
            length = len(index)
        elif length > len(index):
            raise ValueError(
                "index too short for IndexedOptionArray: expected {0}, index length is {1}".format(
                    length, len(index)
                )
                + ak._util.exception_suffix(__file__)
            )

        content = _form_to_layout(
            form.content,
            container,
            partnum,
            key_format,
            0 if len(index) == 0 else max(0, numpy.max(index) + 1),
            lazy_cache,
            lazy_cache_key,
        )

        return _form_to_layout_class[type(form), form.index](
            index, content, identities, parameters
        )

    elif isinstance(form, ak.forms.ListForm):
        raw_starts = _asbuf(
            container[key_format(form_key=fk, attribute="starts", partition=partnum)]
        )
        starts = _index_form_to_index[form.starts](
            raw_starts.view(_index_form_to_dtype[form.starts])
        )
        raw_stops = _asbuf(
            container[key_format(form_key=fk, attribute="stops", partition=partnum)]
        )
        stops = _index_form_to_index[form.stops](
            raw_stops.view(_index_form_to_dtype[form.stops])
        )

        if length is None:
            length = len(starts)
        elif length > len(starts):
            raise ValueError(
                "starts too short for ListArray: expected {0}, starts length is {1}".format(
                    length, len(starts)
                )
                + ak._util.exception_suffix(__file__)
            )
        elif length > len(stops):
            raise ValueError(
                "stops too short for ListArray: expected {0}, stops length is {1}".format(
                    length, len(stops)
                )
                + ak._util.exception_suffix(__file__)
            )

        array_starts = numpy.asarray(starts)
        if len(array_starts) != length:
            array_starts = array_starts[:length]
        array_stops = numpy.asarray(stops)
        if len(array_stops) != length:
            array_stops = array_stops[:length]
        array_stops = array_stops[array_starts != array_stops]
        content = _form_to_layout(
            form.content,
            container,
            partnum,
            key_format,
            0 if len(array_stops) == 0 else numpy.max(array_stops),
            lazy_cache,
            lazy_cache_key,
        )

        return _form_to_layout_class[type(form), form.starts](
            starts, stops, content, identities, parameters
        )

    elif isinstance(form, ak.forms.ListOffsetForm):
        raw_offsets = _asbuf(
            container[key_format(form_key=fk, attribute="offsets", partition=partnum)]
        )
        offsets = _index_form_to_index[form.offsets](
            raw_offsets.view(_index_form_to_dtype[form.offsets])
        )

        if length is None:
            length = len(offsets) - 1
        elif length > len(offsets) - 1:
            raise ValueError(
                "offsets too short for ListOffsetArray: expected {0}, offsets length - 1 is {1}".format(
                    length, len(offsets) - 1
                )
                + ak._util.exception_suffix(__file__)
            )

        content = _form_to_layout(
            form.content,
            container,
            partnum,
            key_format,
            offsets[-1],
            lazy_cache,
            lazy_cache_key,
        )

        return _form_to_layout_class[type(form), form.offsets](
            offsets, content, identities, parameters
        )

    elif isinstance(form, ak.forms.NumpyForm):
        raw_array = _asbuf(
            container[key_format(form_key=fk, attribute="data", partition=partnum)]
        )
        dtype_inner_shape = form.to_numpy()
        if dtype_inner_shape.subdtype is None:
            dtype, inner_shape = dtype_inner_shape, ()
        else:
            dtype, inner_shape = dtype_inner_shape.subdtype

        # If the buffer is required to have a length, check that it's not too short.
        # If the array has no itemsize, then the buffer can never be "too short"!
        if (length is not None) and (dtype_inner_shape.itemsize > 0):
            actual = len(raw_array) // dtype_inner_shape.itemsize
            if length > actual:
                raise ValueError(
                    "buffer is too short for NumpyArray: expected {0}, buffer "
                    "has {1} items ({2} bytes)".format(length, actual, len(raw_array))
                    + ak._util.exception_suffix(__file__)
                )

        # NumPy can only infer the length of the array from the buffer
        # if the inner shape has a nonzero item-size.
        if dtype_inner_shape.itemsize:
            leading_shape = (-1,)
        elif length is not None:
            leading_shape = (length,)
        # Given that `length=None` only occurs when reading legacy data
        # This already failed, so we can fail if `length` is not given
        else:
            raise ValueError(
                "buffer is empty, and no length was given. It is not possible to "
                "determine the correct size of the array from the buffer alone."
            )

        array = raw_array.view(dtype).reshape(leading_shape + inner_shape)

        return ak.layout.NumpyArray(array, identities, parameters)

    elif isinstance(form, ak.forms.RecordForm):
        items = list(form.contents.items())
        if form.istuple:
            items.sort(key=lambda x: int(x[0]))
        contents = []
        minlength = None
        keys = []
        for key, content_form in items:
            keys.append(key)
            content = _form_to_layout(
                content_form,
                container,
                partnum,
                key_format,
                length,
                lazy_cache,
                lazy_cache_key,
            )
            if minlength is None:
                minlength = len(content)
            else:
                minlength = min(minlength, len(content))
            contents.append(content)

        if length is None:
            length = minlength
        elif minlength is not None and length > minlength:
            raise ValueError(
                "RecordArray length mismatch: expected {0}, minimum content is {1}".format(
                    length, minlength
                )
                + ak._util.exception_suffix(__file__)
            )

        return ak.layout.RecordArray(
            contents,
            None if form.istuple else keys,
            length,
            identities,
            parameters,
        )

    elif isinstance(form, ak.forms.RegularForm):
        if length is None:
            length = 0

        content = _form_to_layout(
            form.content,
            container,
            partnum,
            key_format,
            length * form.size,
            lazy_cache,
            lazy_cache_key,
        )

        return ak.layout.RegularArray(
            content, form.size, length, identities, parameters
        )

    elif isinstance(form, ak.forms.UnionForm):
        raw_tags = _asbuf(
            container[key_format(form_key=fk, attribute="tags", partition=partnum)]
        )
        tags = _index_form_to_index[form.tags](
            raw_tags.view(_index_form_to_dtype[form.tags])
        )
        raw_index = _asbuf(
            container[key_format(form_key=fk, attribute="index", partition=partnum)]
        )
        index = _index_form_to_index[form.index](
            raw_index.view(_index_form_to_dtype[form.index])
        )

        if length is None:
            length = len(tags)
        elif length > len(tags):
            raise ValueError(
                "tags too short for UnionArray: expected {0}, tags length is {1}".format(
                    length, len(tags)
                )
                + ak._util.exception_suffix(__file__)
            )
        elif length > len(index):
            raise ValueError(
                "index too short for UnionArray: expected {0}, index length is {1}".format(
                    length, len(index)
                )
                + ak._util.exception_suffix(__file__)
            )

        array_tags = numpy.asarray(tags)
        if len(array_tags) != length:
            array_tags = array_tags[:length]
        array_index = numpy.asarray(index)
        if len(array_index) != length:
            array_index = array_index[:length]

        contents = []
        for i, content_form in enumerate(form.contents):
            mine = array_index[numpy.equal(array_tags, i)]
            contents.append(
                _form_to_layout(
                    content_form,
                    container,
                    partnum,
                    key_format,
                    0 if len(mine) == 0 else numpy.max(mine) + 1,
                    lazy_cache,
                    lazy_cache_key,
                )
            )

        return _form_to_layout_class[type(form), form.index](
            tags, index, contents, identities, parameters
        )

    elif isinstance(form, ak.forms.UnmaskedForm):
        content = _form_to_layout(
            form.content,
            container,
            partnum,
            key_format,
            length,
            lazy_cache,
            lazy_cache_key,
        )

        return ak.layout.UnmaskedArray(content, identities, parameters)

    elif isinstance(form, ak.forms.VirtualForm):
        args = (
            form.form,
            container,
            partnum,
            key_format,
            length,
            lazy_cache,
            lazy_cache_key,
        )
        generator = ak.layout.ArrayGenerator(
            _form_to_layout,
            args,
            form=form.form,
            length=length,
        )
        node_cache_key = key_format(
            form_key=form.form.form_key, attribute="virtual", partition=partnum
        )

        if isinstance(form.form, (ak.forms.NumpyForm, ak.forms.EmptyForm)):
            # If it's a leaf node, the node_cache_key completely determines
            # uniqueness of the subtree (because it's the whole subtree).
            nested_cache_key = "{0}({1})".format(lazy_cache_key, node_cache_key)
        else:
            # Otherwise, the node_cache_key for the root of the subtree might
            # be the same in two places whereas the nested content differs.
            nested_cache_key = "{0}({1}:{2})".format(
                lazy_cache_key, node_cache_key, _from_buffers_key()
            )

        return ak.layout.VirtualArray(generator, lazy_cache, nested_cache_key)

    else:
        raise AssertionError(
            "unexpected form node type: "
            + str(type(form))
            + ak._util.exception_suffix(__file__)
        )


_from_buffers_key_number = 0
_from_buffers_key_lock = threading.Lock()


def _from_buffers_key():
    global _from_buffers_key_number
    with _from_buffers_key_lock:
        out = _from_buffers_key_number
        _from_buffers_key_number += 1
    return out


def _wrap_record_with_virtual(input_form):
    def modify(form):
        if form["class"] == "RecordArray":
            for item in form["contents"].values():
                modify(item)
        elif form["class"].startswith("UnionArray"):
            for item in form["contents"]:
                modify(item)
        elif "content" in form:
            modify(form["content"])

        if form["class"] == "RecordArray":
            for key in form["contents"].keys():
                form["contents"][key] = {
                    "class": "VirtualArray",
                    "has_length": True,
                    "form": form["contents"][key],
                }

    form = json.loads(input_form.tojson())
    modify(form)
    return ak.forms.Form.fromjson(json.dumps(form))


def from_buffers(
    form,
    length,
    container,
    partition_start=0,
    key_format="part{partition}-{form_key}-{attribute}",
    lazy=False,
    lazy_cache="new",
    lazy_cache_key=None,
    highlevel=True,
    behavior=None,
):
    """
    Args:
        form (#ak.forms.Form or str/dict equivalent): The form of the Awkward
            Array to reconstitute from named buffers.
        length (int or iterable of int): Length of the array to reconstitute as a
            non-partitioned array or the lengths (plural) of partitions in a
            partitioned array.
        container (Mapping, such as dict): The str \u2192 Python buffers that
            represent the decomposed Awkward Array. This `container` is only
            assumed to have a `__getitem__` method that accepts strings as keys.
        partition_start (int): First (or only) partition number to get from the
            `container`.
        key_format (str or callable): Python format string containing
            `"{partition}"`, `"{form_key}"`, and/or `"{attribute}"` or a function
            that takes these as keyword arguments and returns a string to use
            as keys for buffers in the `container`. The `partition` is a
            partition number (non-negative integer, passed as a string), the
            `form_key` is a string associated with each node in the Form, and the
            `attribute` is a hard-coded string representing the buffer's function
            (e.g. `"data"`, `"offsets"`, `"index"`).
        lazy (bool): If True, read the array or its partitions on demand (as
            #ak.layout.VirtualArray, possibly in #ak.partition.PartitionedArray
            if `num_partitions` is not None); if False, read all requested data
            immediately. Any RecordArray child nodes will additionally be
            read on demand.
        lazy_cache (None, "new", or MutableMapping): If lazy, pass this
            cache to the VirtualArrays. If "new", a new dict (keep-forever cache)
            is created. If None, no cache is used.
        lazy_cache_key (None or str): If lazy, pass this cache_key to the
            VirtualArrays. If None, a process-unique string is constructed.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Reconstitutes an Awkward Array from a Form, length, and a collection of memory
    buffers, so that data can be losslessly read from file formats and storage
    devices that only map names to binary blobs (such as a filesystem directory).

    The first three arguments of this function are the return values of
    #ak.to_buffers, so a full round-trip is

        >>> reconstituted = ak.from_buffers(*ak.to_buffers(original))

    The `container` argument lets you specify your own Mapping, which might be
    an interface to some storage format or device (e.g. h5py). It's okay if
    the `container` dropped NumPy's `dtype` and `shape` information, leaving
    raw bytes, since `dtype` and `shape` can be reconstituted from the
    #ak.forms.NumpyForm.

    The `key_format` should be the same as the one used in #ak.to_buffers.

    The arguments that begin with `lazy_` are only needed if `lazy` is True.
    The `lazy_cache` and `lazy_cache_key` determine how the array or its
    partitions are cached after being read from the `container` (in a no-eviction
    dict attached to the output #ak.Array as `cache` if not specified).

    See #ak.to_buffers for examples.
    """

    if isinstance(form, str) or (ak._util.py27 and isinstance(form, ak._util.unicode)):
        form = ak.forms.Form.fromjson(form)
    elif isinstance(form, dict):
        form = ak.forms.Form.fromjson(json.dumps(form))

    if isinstance(key_format, str):

        def generate_key_format(key_format):
            def kf(**v):
                return key_format.format(**v)

            return kf

        key_format = generate_key_format(key_format)

    hold_cache = None
    if lazy:
        form = _wrap_record_with_virtual(form)

        if lazy_cache == "new":
            hold_cache = ak._util.MappingProxy({})
            lazy_cache = ak.layout.ArrayCache(hold_cache)
        elif lazy_cache is not None and not isinstance(
            lazy_cache, ak.layout.ArrayCache
        ):
            hold_cache = ak._util.MappingProxy.maybe_wrap(lazy_cache)
            if not isinstance(hold_cache, MutableMapping):
                raise TypeError("lazy_cache must be a MutableMapping")
            lazy_cache = ak.layout.ArrayCache(hold_cache)

        if lazy_cache_key is None:
            lazy_cache_key = "ak.from_buffers:{0}".format(_from_buffers_key())

    if length is None or isinstance(length, (numbers.Integral, np.integer)):
        if length is None:
            raise TypeError(
                "length must be an integer or an iterable of integers"
                + ak._util.exception_suffix(__file__)
            )

        args = (form, container, str(partition_start), key_format, length)

        if lazy:
            generator = ak.layout.ArrayGenerator(
                _form_to_layout,
                args + (lazy_cache, lazy_cache_key),
                form=form,
                length=length,
            )
            out = ak.layout.VirtualArray(generator, lazy_cache, lazy_cache_key)

        else:
            out = _form_to_layout(*(args + (None, None)))

    elif isinstance(length, Iterable):
        partitions = []
        offsets = [0]

        for part, partlen in enumerate(length):
            partnum = str(partition_start + part)
            args = (form, container, partnum, key_format)

            if lazy:
                lazy_cache_key_part = "{0}[{1}]".format(lazy_cache_key, partnum)
                generator = ak.layout.ArrayGenerator(
                    _form_to_layout,
                    args + (partlen, lazy_cache, lazy_cache_key_part),
                    form=form,
                    length=length[part],
                )

                partitions.append(
                    ak.layout.VirtualArray(generator, lazy_cache, lazy_cache_key_part)
                )
                offsets.append(offsets[-1] + length[part])

            else:
                partitions.append(_form_to_layout(*(args + (partlen, None, None))))
                offsets.append(offsets[-1] + len(partitions[-1]))

        out = ak.partition.IrregularlyPartitionedArray(partitions, offsets[1:])

    else:
        raise TypeError(
            "length must be an integer or an iterable of integers, not "
            + repr(length)
            + ak._util.exception_suffix(__file__)
        )

    return ak._util.maybe_wrap(out, behavior, highlevel)


def to_pandas(
    array, how="inner", levelname=lambda i: "sub" * i + "entry", anonymous="values"
):
    """
    Args:
        array: Data to convert into one or more Pandas DataFrames.
        how (None or str): Passed to
            [pd.merge](https://pandas.pydata.org/pandas-docs/version/1.0.3/reference/api/pandas.merge.html)
            to combine DataFrames for each multiplicity into one DataFrame. If
            None, a list of Pandas DataFrames is returned.
        levelname (int -> str): Computes a name for each level of the row index
            from the number of levels deep.
        anonymous (str): Column name to use if the `array` does not contain
            records; otherwise, column names are derived from record fields.

    Converts Awkward data structures into Pandas
    [MultiIndex](https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html)
    rows and columns. The resulting DataFrame(s) contains no Awkward structures.

    #ak.Array structures can't be losslessly converted into a single DataFrame;
    different fields in a record structure might have different nested list
    lengths, but a DataFrame can have only one index.

    If `how` is None, this function always returns a list of DataFrames (even
    if it contains only one DataFrame); otherwise `how` is passed to
    [pd.merge](https://pandas.pydata.org/pandas-docs/version/1.0.3/reference/api/pandas.merge.html)
    to merge them into a single DataFrame with the associated loss of data.

    In the following example, nested lists are converted into MultiIndex rows.
    The index level names `"entry"`, `"subentry"` and `"subsubentry"` can be
    controlled with the `levelname` parameter. The column name `"values"` is
    assigned because this array has no fields; it can be controlled with the
    `anonymous` parameter.

        >>> ak.to_pandas(ak.Array([[[1.1, 2.2], [], [3.3]],
        ...                        [],
        ...                        [[4.4], [5.5, 6.6]],
        ...                        [[7.7]],
        ...                        [[8.8]]]))
                                    values
        entry subentry subsubentry
        0     0        0               1.1
                       1               2.2
              2        0               3.3
        2     0        0               4.4
              1        0               5.5
                       1               6.6
        3     0        0               7.7
        4     0        0               8.8

    In this example, nested records are converted into MultiIndex columns.
    (MultiIndex rows and columns can be mixed; these examples are deliberately
    simple.)

        >>> ak.to_pandas(ak.Array([
        ...     {"I": {"a": _, "b": {"i": _}}, "II": {"x": {"y": {"z": _}}}}
        ...     for _ in range(0, 50, 10)]))
                I      II
                a   b   x
                    i   y
                        z
        entry
        0       0   0   0
        1      10  10  10
        2      20  20  20
        3      30  30  30
        4      40  40  40

    The following two examples show how fields of different length lists are
    merged. With `how="inner"` (default), only subentries that exist for all
    fields are preserved; with `how="outer"`, all subentries are preserved at
    the expense of requiring missing values.

        >>> ak.to_pandas(ak.Array([{"x": [], "y": [4.4, 3.3, 2.2, 1.1]},
        ...                        {"x": [1], "y": [3.3, 2.2, 1.1]},
        ...                        {"x": [1, 2], "y": [2.2, 1.1]},
        ...                        {"x": [1, 2, 3], "y": [1.1]},
        ...                        {"x": [1, 2, 3, 4], "y": []}]),
        ...                        how="inner")
                        x    y
        entry subentry
        1     0         1  3.3
        2     0         1  2.2
              1         2  1.1
        3     0         1  1.1

    The same with `how="outer"`:

        >>> ak.to_pandas(ak.Array([{"x": [], "y": [4.4, 3.3, 2.2, 1.1]},
        ...                        {"x": [1], "y": [3.3, 2.2, 1.1]},
        ...                        {"x": [1, 2], "y": [2.2, 1.1]},
        ...                        {"x": [1, 2, 3], "y": [1.1]},
        ...                        {"x": [1, 2, 3, 4], "y": []}]),
        ...                        how="outer")
                          x    y
        entry subentry
        0     0         NaN  4.4
              1         NaN  3.3
              2         NaN  2.2
              3         NaN  1.1
        1     0         1.0  3.3
              1         NaN  2.2
              2         NaN  1.1
        2     0         1.0  2.2
              1         2.0  1.1
        3     0         1.0  1.1
              1         2.0  NaN
              2         3.0  NaN
        4     0         1.0  NaN
              1         2.0  NaN
              2         3.0  NaN
              3         4.0  NaN
    """
    try:
        import pandas
    except ImportError:
        raise ImportError(
            """install the 'pandas' package with:

    pip install pandas --upgrade

or

    conda install pandas"""
        )

    if how is not None:
        out = None
        for df in to_pandas(array, how=None, levelname=levelname, anonymous=anonymous):
            if out is None:
                out = df
            else:
                out = pandas.merge(out, df, how=how, left_index=True, right_index=True)
        return out

    def recurse(layout, row_arrays, col_names):
        if isinstance(layout, ak._util.virtualtypes):
            return recurse(layout.array, row_arrays, col_names)

        elif isinstance(layout, ak._util.indexedtypes):
            return recurse(layout.project(), row_arrays, col_names)

        elif layout.parameter("__array__") in ("string", "bytestring"):
            return [(to_numpy(layout), row_arrays, col_names)]

        elif layout.purelist_depth > 1:
            offsets, flattened = layout.offsets_and_flatten(axis=1)
            offsets = numpy.asarray(offsets)
            starts, stops = offsets[:-1], offsets[1:]
            counts = stops - starts
            if ak._util.win or ak._util.bits32:
                counts = counts.astype(np.int32)
            if len(row_arrays) == 0:
                newrows = [
                    numpy.repeat(numpy.arange(len(counts), dtype=counts.dtype), counts)
                ]
            else:
                newrows = [numpy.repeat(x, counts) for x in row_arrays]
            newrows.append(
                numpy.arange(offsets[-1], dtype=counts.dtype)
                - numpy.repeat(starts, counts)
            )
            return recurse(flattened, newrows, col_names)

        elif isinstance(layout, ak._util.uniontypes):
            layout = ak._util.union_to_record(layout, anonymous)
            if isinstance(layout, ak._util.uniontypes):
                return [(to_numpy(layout), row_arrays, col_names)]
            else:
                return sum(
                    [
                        recurse(layout.field(n), row_arrays, col_names + (n,))
                        for n in layout.keys()
                    ],
                    [],
                )

        elif isinstance(layout, ak.layout.RecordArray):
            return sum(
                [
                    recurse(layout.field(n), row_arrays, col_names + (n,))
                    for n in layout.keys()
                ],
                [],
            )

        else:
            return [(to_numpy(layout), row_arrays, col_names)]

    layout = to_layout(array, allow_record=True, allow_other=False)
    if isinstance(layout, ak.partition.PartitionedArray):
        layout = layout.toContent()

    if isinstance(layout, ak.layout.Record):
        layout2 = layout.array[layout.at : layout.at + 1]
    else:
        layout2 = layout

    tables = []
    last_row_arrays = None
    for column, row_arrays, col_names in recurse(layout2, [], ()):
        if isinstance(layout, ak.layout.Record):
            row_arrays = row_arrays[1:]  # Record --> one-element RecordArray
        if len(col_names) == 0:
            columns = [anonymous]
        else:
            columns = pandas.MultiIndex.from_tuples([col_names])

        if (
            last_row_arrays is not None
            and len(last_row_arrays) == len(row_arrays)
            and all(
                numpy.array_equal(x, y) for x, y in zip(last_row_arrays, row_arrays)
            )
        ):
            oldcolumns = tables[-1].columns
            if isinstance(oldcolumns, pandas.MultiIndex):
                numold = len(oldcolumns.levels)
            else:
                numold = max(len(x) for x in oldcolumns)
            numnew = len(columns.levels)
            maxnum = max(numold, numnew)
            if numold != maxnum:
                oldcolumns = pandas.MultiIndex.from_tuples(
                    [x + ("",) * (maxnum - numold) for x in oldcolumns]
                )
                tables[-1].columns = oldcolumns
            if numnew != maxnum:
                columns = pandas.MultiIndex.from_tuples(
                    [x + ("",) * (maxnum - numnew) for x in columns]
                )

            newframe = pandas.DataFrame(
                data=column, index=tables[-1].index, columns=columns
            )
            tables[-1] = pandas.concat([tables[-1], newframe], axis=1)

        else:
            if len(row_arrays) == 0:
                index = pandas.RangeIndex(len(column), name=levelname(0))
            else:
                index = pandas.MultiIndex.from_arrays(
                    row_arrays, names=[levelname(i) for i in range(len(row_arrays))]
                )
            tables.append(pandas.DataFrame(data=column, index=index, columns=columns))

        last_row_arrays = row_arrays

    for table in tables:
        if isinstance(table.columns, pandas.MultiIndex) and table.columns.nlevels == 1:
            table.columns = table.columns.get_level_values(0)

    return tables


__all__ = [
    x
    for x in list(globals())
    if not x.startswith("_")
    and x
    not in (
        "numbers",
        "json",
        "collections",
        "math",
        "os",
        "threading",
        "distutils",
        "glob",
        "re",
        "Iterable",
        "MutableMapping",
        "numpy",
        "ak",
        "np",
    )
]


def __dir__():
    return __all__
