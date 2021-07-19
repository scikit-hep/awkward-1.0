# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import types

import awkward as ak

numpy = ak.nplike.NumPy.instance()

NEP13Box = None


def register():
    import autograd

    global NEP13Box

    if NEP13Box is None:

        class NEP13Box(autograd.extend.Box, ak._connect._numpy.NDArrayOperatorsMixin):
            def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
                import autograd

                if (
                    method != "__call__"
                    or len(inputs) == 0
                    or "out" in kwargs
                    or ufunc.__class__.__module__ != "numpy"
                ):
                    return NotImplemented

                nextinputs = []
                for x in inputs:
                    if isinstance(x, NEP13Box):
                        nextinputs.append(
                            autograd.numpy.numpy_boxes.ArrayBox(
                                numpy.asarray(x._value), x._trace, x._node
                            )
                        )
                    else:
                        nextinputs.append(x)

                out = getattr(autograd.numpy, ufunc.__name__)(*nextinputs, **kwargs)
                return NEP13Box(ak.layout.NumpyArray(out._value), out._trace, out._node)

        NEP13Box.register(ak.layout.NumpyArray)

        autograd.extend.VSpace.register(
            ak.layout.NumpyArray,
            lambda x: autograd.numpy.numpy_vspaces.ArrayVSpace(numpy.asarray(x)),
        )


def elementwise_grad(fun, argnum=0, *nary_op_args, **nary_op_kwargs):
    import autograd

    register()

    gradfun = autograd.elementwise_grad(fun, argnum, *nary_op_args, **nary_op_kwargs)

    def broadcast(*args, **kwargs):
        nextargs = [
            ak.operations.convert.to_layout(x, allow_record=True, allow_other=True)
            for x in args
        ]

        def getfunction(inputs):
            if all(
                isinstance(x, ak.layout.NumpyArray)
                or not isinstance(x, ak.layout.Content)
                for x in inputs
            ):
                return lambda: (ak.layout.NumpyArray(gradfun(*inputs)),)
            else:
                return None

        behavior = ak._util.behaviorof(*args)
        out = ak._util.broadcast_and_apply(
            nextargs, getfunction, behavior, pass_depth=False
        )
        assert isinstance(out, tuple) and len(out) == 1
        return ak._util.wrap(out[0], behavior)

    return broadcast


ak.autograd = types.ModuleType("autograd")
ak.autograd.elementwise_grad = elementwise_grad
