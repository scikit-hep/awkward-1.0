# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak
import numbers
import json

import jax
import jax.tree_util

np = ak.nplike.NumPyMetadata.instance()
numpy = ak.nplike.NumPy.instance()


def _find_dataptrs(layout):
    def find_nparray_ptrs(node, depth, data_ptrs):
        if isinstance(node, ak.layout.NumpyArray):
            data_ptrs.append(node.ptr)

    data_ptrs = []
    ak._util.recursive_walk(layout, find_nparray_ptrs, args=(data_ptrs,))

    return data_ptrs


def _find_dataptrs_and_map(layout, jaxtracers, isscalar):
    if not isscalar:
        data_ptrs = _find_dataptrs(layout)
        assert len(jaxtracers) == len(data_ptrs)
        map_ptrs_to_tracers = dict(zip(data_ptrs, jaxtracers))
    else:
        # Layout is a scalar
        data_ptrs = None
        map_ptrs_to_tracers = None

    return data_ptrs, map_ptrs_to_tracers


def _jaxtracers_getitem(array, where):
    if array._isscalar:
        raise TypeError("Cannot slice a scalar")

    out = array.layout[where]

    def find_nparray_node_newptr(layout, outlayout):
        def find_nparray_node(
            node, depth, outlayout, fieldloc, shape, nodenum, nodenum_index
        ):
            if isinstance(node, ak.layout.NumpyArray):
                if node.identities.fieldloc == fieldloc:
                    nodenum_index.append(nodenum)
                else:
                    nodenum = nodenum + 1

        outlayout_fieldloc = outlayout.identities.fieldloc
        nodenum_index = []

        ak._util.recursive_walk(
            layout,
            find_nparray_node,
            args=(
                outlayout,
                outlayout_fieldloc,
                numpy.asarray(outlayout.identities).shape,
                0,
                nodenum_index,
            ),
        )
        if len(nodenum_index) == 0:
            raise ValueError("Couldn't find the node in new slice")
        return nodenum_index[0]

    if not isinstance(out, ak.layout.Content):

        def recurse(outarray, recurse_where):

            if isinstance(recurse_where, numbers.Integral) or isinstance(
                recurse_where, str
            ):
                if isinstance(outarray.layout, ak.layout.NumpyArray):
                    return outarray._tracers[0][recurse_where]
            elif isinstance(where, tuple):
                return recurse(array[where[:-1]], where[len(where) - 1])

            else:
                raise ValueError("Can't slice the array with {0}".format(where))

        child = [recurse(array, where)]
        return ak.Array._internal_for_jax(out, child, isscalar=True)

    else:

        def fetch_indices_and_fieldloc_layout(outlayout):
            if isinstance(outlayout, ak.layout.NumpyArray):
                return [
                    (
                        (
                            outlayout.identities.fieldloc,
                            numpy.asarray(outlayout.identities).shape[1],
                        ),
                        numpy.asarray(outlayout.identities),
                    )
                ]
            elif isinstance(outlayout, ak._util.listtypes):
                return fetch_indices_and_fieldloc_layout(outlayout.content)
            elif isinstance(outlayout, ak._util.indexedtypes):
                return fetch_indices_and_fieldloc_layout(outlayout.project())
            elif isinstance(outlayout, ak._util.uniontypes):
                raise ValueError(
                    "Can't differentiate an UnionArray type {0}".format(outlayout)
                )
            elif isinstance(outlayout, ak._util.recordtypes):
                indices = []
                for content in outlayout.contents:
                    indices = indices + fetch_indices_and_fieldloc_layout(content)
                return indices
            elif isinstance(outlayout, ak._util.indexedtypes):
                return fetch_indices_and_fieldloc_layout(outlayout.content)
            elif isinstance(outlayout, ak._util.indexedoptiontypes):
                return fetch_indices_and_fieldloc_layout(outlayout.content)
            elif isinstance(
                outlayout,
                (
                    ak.layout.BitMaskedArray,
                    ak.layout.ByteMaskedArray,
                    ak.layout.UnmaskedArray,
                ),
            ):
                return fetch_indices_and_fieldloc_layout(outlayout.content)
            else:
                raise NotImplementedError

        def fetch_children_tracer(outlayout, preslice_identities):
            if isinstance(outlayout, ak.layout.NumpyArray):

                def find_intersection_indices(
                    postslice_identities, preslice_identities
                ):
                    multiplier = numpy.append(
                        numpy.cumprod(
                            (numpy.max(preslice_identities, axis=0) + 1)[::-1]
                        )[-2::-1],
                        1,
                    )
                    haystack = numpy.sum(preslice_identities * multiplier, axis=1)
                    needle = numpy.sum(postslice_identities * multiplier, axis=1)
                    haystack_argsort = numpy.argsort(haystack)
                    indices = numpy.searchsorted(
                        haystack, needle, sorter=haystack_argsort
                    )
                    final_indices = haystack_argsort[indices]
                    return final_indices

                def find_corresponding_identity(
                    postslice_identities, preslice_identities
                ):
                    for identity in preslice_identities:
                        if identity[0] == postslice_identities:
                            return identity[1]

                    raise ValueError(
                        "Couldn't find postslice identities in preslice identities"
                    )

                indices = find_intersection_indices(
                    numpy.asarray(outlayout.identities),
                    find_corresponding_identity(
                        (
                            outlayout.identities.fieldloc,
                            numpy.asarray(outlayout.identities).shape[1],
                        ),
                        preslice_identities,
                    ),
                )
                if outlayout.ptr in array._map_ptrs_to_tracers:
                    tracer = array._map_ptrs_to_tracers[outlayout.ptr]
                else:
                    tracer = array._tracers[
                        find_nparray_node_newptr(array.layout, outlayout)
                    ]
                return [jax.numpy.take(tracer, indices)]

            elif isinstance(outlayout, ak._util.listtypes):
                return fetch_children_tracer(outlayout.content, preslice_identities)
            elif isinstance(outlayout, ak._util.uniontypes):
                raise ValueError(
                    "Can't differentiate an UnionArray type {0}".format(outlayout)
                )
            elif isinstance(outlayout, ak._util.recordtypes):
                children = []
                for content in outlayout.contents:
                    children = children + fetch_children_tracer(
                        content, preslice_identities
                    )
                return children
            elif isinstance(outlayout, ak._util.indexedtypes):
                return fetch_children_tracer(outlayout.content, preslice_identities)
            elif isinstance(outlayout, ak._util.indexedoptiontypes):
                return fetch_children_tracer(outlayout.content, preslice_identities)
            elif isinstance(
                outlayout,
                (
                    ak.layout.BitMaskedArray,
                    ak.layout.ByteMaskedArray,
                    ak.layout.UnmaskedArray,
                ),
            ):
                return fetch_children_tracer(outlayout.content, preslice_identities)
            else:
                raise NotImplementedError(
                    "fetch_children_tracer not completely implemented yet for {0}".format(
                        outlayout
                    )
                )

        children = fetch_children_tracer(
            out, fetch_indices_and_fieldloc_layout(array.layout)
        )
        out = out.deep_copy()
        out.setidentities()
        return ak.Array._internal_for_jax(out, children, isscalar=False)


def array_ufunc(array, ufunc, method, inputs, kwargs):
    for x in inputs:
        if isinstance(x, ak.Array) and hasattr(x, "_tracers"):
            assert len(x._tracers) == len(array._tracers)

    # ak.Array __add__, etc. map to the NumPy functions, switch to JAX
    import numpy

    for name, np_ufunc in numpy.core.umath.__dict__.items():
        if ufunc is np_ufunc:
            ufunc = getattr(jax.numpy, name)

    # need to apply the ufunc to the same argument list for each tracer separately
    nexttracers = []
    for i in range(len(array._tracers)):
        nextinputs = [
            x._tracers[i] if isinstance(x, ak.Array) and hasattr(x, "_tracers") else x
            for x in inputs
        ]
        nexttracers.append(getattr(ufunc, method)(*nextinputs, **kwargs))

    return ak.Array._internal_for_jax(array.layout, nexttracers, array._isscalar)


class AuxData(object):
    def __init__(self, array):
        self.array = array

    def __eq__(self, other):
        self_form = json.loads(self.array.layout.form.tojson())
        other_form = json.loads(self.array.layout.form.tojson())

        def form_sweep(input_form, blacklist):
            if isinstance(input_form, dict):
                return {
                    k: form_sweep(v, blacklist)
                    for k, v in input_form.items()
                    if k not in blacklist
                }
            else:
                return input_form

        self_form = form_sweep(self_form, ["primitive", "format", "itemsize"])
        other_form = form_sweep(self_form, ["primitive", "format", "itemsize"])

        return self_form == other_form


def special_flatten(array):
    if isinstance(array, ak.Array) and hasattr(array, "_tracers"):
        aux_data, children = AuxData(array), array._tracers
    elif isinstance(array, ak.Array):

        def create_databuffers(node, depth, databuffers):
            if isinstance(node, ak.layout.NumpyArray):
                databuffers.append(node)

        databuffers = []
        ak._util.recursive_walk(array.layout, create_databuffers, args=(databuffers,))

        array.layout.setidentities()
        children = [jax.numpy.asarray(x) for x in databuffers]
        aux_data = AuxData(ak.Array._internal_for_jax(array.layout, children))
    else:
        raise ValueError(
            "Can only differentiate Awkward Arrays, received array of type {0}".format(
                type(array)
            )
        )

    return children, aux_data


def special_unflatten(aux_data, children):
    if any(isinstance(x, jax.core.Tracer) for x in children):
        return ak.Array._internal_for_jax(aux_data.array.layout, children)
    elif all(child is None for child in children):
        return None
    else:
        if aux_data.array._isscalar:
            assert len(children) == 1
            # return children[0]
            import numpy

            return numpy.ndarray.item(numpy.asarray(children[0]))
        children = list(children)

        def function(layout):
            if isinstance(layout, ak.layout.NumpyArray):
                buffer = children[0]
                children.pop(0)
                return lambda: ak.layout.NumpyArray(buffer)

        arr = ak._util.recursively_apply(
            aux_data.array.layout, function, pass_depth=False
        )
        return ak.Array(arr)


jax.tree_util.register_pytree_node(ak.Array, special_flatten, special_unflatten)
