# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def count(array, axis=None, keepdims=False, mask_identity=False):
    """
    Args:
        array: Data in which to count elements.
        axis (None or int): If None, combine all values from the array into
            a single scalar result; if an int, group by that axis: `0` is the
            outermost, `1` is the first level of nested lists, etc., and
            negative `axis` counts from the innermost: `-1` is the innermost,
            `-2` is the next level up, etc.
        keepdims (bool): If False, this reducer decreases the number of
            dimensions by 1; if True, the reduced values are wrapped in a new
            length-1 dimension so that the result of this operation may be
            broadcasted with the original array.
        mask_identity (bool): If True, reducing over empty lists results in
            None (an option type); otherwise, reducing over empty lists
            results in the operation's identity.

    Counts elements of `array` (many types supported, including all
    Awkward Arrays and Records). The identity of counting is `0` and it is
    usually not masked.

    This function has no analog in NumPy because counting values in a
    rectilinear array would only result in elements of the NumPy array's
    [shape](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.shape.html).

    However, for nested lists of variable dimension and missing values, the
    result of counting is non-trivial. For example, with this `array`,

        ak.Array([[ 0.1,  0.2      ],
                  [None, 10.2, None],
                  None,
                  [20.1, 20.2, 20.3],
                  [30.1, 30.2      ]])

    the result of counting over the innermost dimension is

        >>> ak.count(array, axis=-1)
        <Array [2, 1, None, 3, 2] type='5 * ?int64'>

    the outermost dimension is

        >>> ak.count(array, axis=0)
        <Array [3, 4, 1] type='3 * int64'>

    and all dimensions is

        >>> ak.count(array, axis=None)
        8

    The gaps and None values are not counted, and if a None value occurs at
    a higher axis than the one being counted, it is kept as a placeholder
    so that the outer list length does not change.

    See #ak.sum for a more complete description of nested list and missing
    value (None) handling in reducers.

    Note also that this function is different from #ak.num, which counts
    the number of values at a given depth, maintaining structure: #ak.num
    never counts across different lists the way that reducers do (#ak.num
    is not a reducer; #ak.count is). For the same `array`,

        >>> ak.num(array, axis=0)
        5
        >>> ak.num(array, axis=1)
        <Array [2, 3, None, 3, 2] type='5 * ?int64'>

    If it is desirable to include None values in #ak.count, use #ak.fill_none
    to turn the None values into something that would be counted.
    """
    layout = ak.operations.convert.to_layout(
        array, allow_record=False, allow_other=False
    )
    if axis is None:

        def reduce(xs):
            if len(xs) == 1:
                return xs[0]
            else:
                return xs[0] + reduce(xs[1:])

        return reduce(
            [ak.nplike.of(x).size(x) for x in ak._util.completely_flatten(layout)]
        )
    else:
        behavior = ak._util.behaviorof(array)
        return ak._util.wrap(
            layout.count(axis=axis, mask=mask_identity, keepdims=keepdims), behavior
        )


@ak._connect._numpy.implements("count_nonzero")
def count_nonzero(array, axis=None, keepdims=False, mask_identity=False):
    """
    Args:
        array: Data in which to count nonzero elements.
        axis (None or int): If None, combine all values from the array into
            a single scalar result; if an int, group by that axis: `0` is the
            outermost, `1` is the first level of nested lists, etc., and
            negative `axis` counts from the innermost: `-1` is the innermost,
            `-2` is the next level up, etc.
        keepdims (bool): If False, this reducer decreases the number of
            dimensions by 1; if True, the reduced values are wrapped in a new
            length-1 dimension so that the result of this operation may be
            broadcasted with the original array.
        mask_identity (bool): If True, reducing over empty lists results in
            None (an option type); otherwise, reducing over empty lists
            results in the operation's identity.

    Counts nonzero elements of `array` (many types supported, including all
    Awkward Arrays and Records). The identity of counting is `0` and it is
    usually not masked. This operation is the same as NumPy's
    [count_nonzero](https://docs.scipy.org/doc/numpy/reference/generated/numpy.count_nonzero.html)
    if all lists at a given dimension have the same length and no None values,
    but it generalizes to cases where they do not.

    See #ak.sum for a more complete description of nested list and missing
    value (None) handling in reducers.

    Following the same rules as other reducers, #ak.count_nonzero does not
    count None values. If it is desirable to count them, use #ak.fill_none
    to turn them into something that would be counted.
    """
    layout = ak.operations.convert.to_layout(
        array, allow_record=False, allow_other=False
    )
    if axis is None:

        def reduce(xs):
            if len(xs) == 1:
                return xs[0]
            else:
                return xs[0] + reduce(xs[1:])

        return reduce(
            [
                ak.nplike.of(x).count_nonzero(x)
                for x in ak._util.completely_flatten(layout)
            ]
        )
    else:
        behavior = ak._util.behaviorof(array)
        return ak._util.wrap(
            layout.count_nonzero(axis=axis, mask=mask_identity, keepdims=keepdims),
            behavior,
        )


@ak._connect._numpy.implements("sum")
def sum(array, axis=None, keepdims=False, mask_identity=False):
    """
    Args:
        array: Data to sum over.
        axis (None or int): If None, combine all values from the array into
            a single scalar result; if an int, group by that axis: `0` is the
            outermost, `1` is the first level of nested lists, etc., and
            negative `axis` counts from the innermost: `-1` is the innermost,
            `-2` is the next level up, etc.
        keepdims (bool): If False, this reducer decreases the number of
            dimensions by 1; if True, the reduced values are wrapped in a new
            length-1 dimension so that the result of this operation may be
            broadcasted with the original array.
        mask_identity (bool): If True, reducing over empty lists results in
            None (an option type); otherwise, reducing over empty lists
            results in the operation's identity.

    Sums over `array` (many types supported, including all Awkward Arrays
    and Records). The identity of addition is `0` and it is usually not
    masked. This operation is the same as NumPy's
    [sum](https://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html)
    if all lists at a given dimension have the same length and no None values,
    but it generalizes to cases where they do not.

    For example, consider this `array`, in which all lists at a given dimension
    have the same length.

        ak.Array([[ 0.1,  0.2,  0.3],
                  [10.1, 10.2, 10.3],
                  [20.1, 20.2, 20.3],
                  [30.1, 30.2, 30.3]])

    A sum over `axis=-1` combines the inner lists, leaving one value per
    outer list:

        >>> ak.sum(array, axis=-1)
        <Array [0.6, 30.6, 60.6, 90.6] type='4 * float64'>

    while a sum over `axis=0` combines the outer lists, leaving one value
    per inner list:

        >>> ak.sum(array, axis=0)
        <Array [60.4, 60.8, 61.2] type='3 * float64'>

    Now with some values missing,

        ak.Array([[ 0.1,  0.2      ],
                  [10.1            ],
                  [20.1, 20.2, 20.3],
                  [30.1, 30.2      ]])

    The sum over `axis=-1` results in

        >>> ak.sum(array, axis=-1)
        <Array [0.3, 10.1, 60.6, 60.3] type='4 * float64'>

    and the sum over `axis=0` results in

        >>> ak.sum(array, axis=0)
        <Array [60.4, 50.6, 20.3] type='3 * float64'>

    How we ought to sum over the innermost lists is unambiguous, but for all
    other `axis` values, we must choose whether to align contents to the
    left before summing, to the right before summing, or something else.
    As suggested by the way the text has been aligned, we choose the
    left-alignment convention: the first `axis=0` result is the sum of all
    first elements

        60.4 = 0.1 + 10.1 + 20.1 + 30.1

    the second is the sum of all second elements

        50.6 = 0.2 + 20.2 + 30.2

    and the third is the sum of the only third element

        20.3 = 20.3

    The same is true if the values were None, rather than gaps:

        ak.Array([[ 0.1,  0.2, None],
                  [10.1, None, None],
                  [20.1, 20.2, 20.3],
                  [30.1, 30.2, None]])

        >>> ak.sum(array, axis=-1)
        <Array [0.3, 10.1, 60.6, 60.3] type='4 * float64'>
        >>> ak.sum(array, axis=0)
        <Array [60.4, 50.6, 20.3] type='3 * float64'>

    However, the missing value placeholder, None, allows us to align the
    remaining data differently:

        ak.Array([[None,  0.1,  0.2],
                  [None, None, 10.1],
                  [20.1, 20.2, 20.3],
                  [None, 30.1, 30.2]])

    Now the `axis=-1` result is the same but the `axis=0` result has changed:

        >>> ak.sum(array, axis=-1)
        <Array [0.3, 10.1, 60.6, 60.3] type='4 * float64'>
        >>> ak.sum(array, axis=0)
        <Array [20.1, 50.4, 60.8] type='3 * float64'>

    because

        20.1 = 20.1
        50.4 = 0.1 + 20.2 + 30.1
        60.8 = 0.2 + 10.1 + 20.3 + 30.2

    If, instead of missing numbers, we had missing lists,

        ak.Array([[ 0.1,  0.2,  0.3],
                  None,
                  [20.1, 20.2, 20.3],
                  [30.1, 30.2, 30.3]])

    then the placeholder would pass through the `axis=-1` sum because summing
    over the inner dimension shouldn't change the length of the outer
    dimension.

        >>> ak.sum(array, axis=-1)
        <Array [0.6, None, 60.6, 90.6] type='4 * ?float64'>

    However, the `axis=0` sum loses information about the None value.

        >>> ak.sum(array, axis=0)
        <Array [50.3, 50.6, 50.9] type='3 * float64'>

    which is

        50.3 = 0.1 + (None) + 20.1 + 30.1
        50.6 = 0.2 + (None) + 20.2 + 30.2
        50.9 = 0.3 + (None) + 20.3 + 30.3

    An `axis=0` sum would be reducing that information if it had not been
    None, anyway. If the None values were replaced with `0`, the result for
    `axis=0` would be the same. The result for `axis=-1` would not be the
    same because this None is in the `0` axis, not the axis that `axis=-1`
    sums over.

    The `keepdims` parameter ensures that the number of dimensions does not
    change: scalar results are put into new length-1 dimensions:

        >>> ak.sum(array, axis=-1, keepdims=True)
        <Array [[0.6], None, [60.6], [90.6]] type='4 * option[1 * float64]'>
        >>> ak.sum(array, axis=0, keepdims=True)
        <Array [[50.3, 50.6, 50.9]] type='1 * var * float64'>

    and `axis=None` ignores all None values and adds up everything in the
    array (`keepdims` has no effect).

        >>> ak.sum(array, axis=None)
        151.8

    The `mask_identity`, which has no equivalent in NumPy, inserts None in
    the output wherever a reduction takes place over zero elements. This is
    different from reductions that are otherwise equal to the identity or
    are equal to the identity by cancellation.

        >>> array = ak.Array([[2.2, 2.2], [4.4, -2.2, -2.2], [], [0.0]])
        >>> ak.sum(array, axis=-1)
        <Array [4.4, 0, 0, 0] type='4 * float64'>
        >>> ak.sum(array, axis=-1, mask_identity=True)
        <Array [4.4, 0, None, 0] type='4 * ?float64'>

    The third list is reduced to `0` if `mask_identity=False` because `0` is
    the identity of addition, but it is reduced to None if
    `mask_identity=True`.
    """
    layout = ak.operations.convert.to_layout(
        array, allow_record=False, allow_other=False
    )
    if axis is None:

        def reduce(xs):
            if len(xs) == 1:
                return xs[0]
            else:
                return xs[0] + reduce(xs[1:])

        return reduce(
            [ak.nplike.of(x).sum(x) for x in ak._util.completely_flatten(layout)]
        )
    else:
        behavior = ak._util.behaviorof(array)
        return ak._util.wrap(
            layout.sum(axis=axis, mask=mask_identity, keepdims=keepdims), behavior
        )


@ak._connect._numpy.implements("prod")
def prod(array, axis=None, keepdims=False, mask_identity=False):
    """
    Args:
        array: Data to multiply.
        axis (None or int): If None, combine all values from the array into
            a single scalar result; if an int, group by that axis: `0` is the
            outermost, `1` is the first level of nested lists, etc., and
            negative `axis` counts from the innermost: `-1` is the innermost,
            `-2` is the next level up, etc.
        keepdims (bool): If False, this reducer decreases the number of
            dimensions by 1; if True, the reduced values are wrapped in a new
            length-1 dimension so that the result of this operation may be
            broadcasted with the original array.
        mask_identity (bool): If True, reducing over empty lists results in
            None (an option type); otherwise, reducing over empty lists
            results in the operation's identity.

    Multiplies elements of `array` (many types supported, including all
    Awkward Arrays and Records). The identity of multiplication is `1` and it
    is usually not masked. This operation is the same as NumPy's
    [prod](https://docs.scipy.org/doc/numpy/reference/generated/numpy.prod.html)
    if all lists at a given dimension have the same length and no None values,
    but it generalizes to cases where they do not.

    See #ak.sum for a more complete description of nested list and missing
    value (None) handling in reducers.
    """
    layout = ak.operations.convert.to_layout(
        array, allow_record=False, allow_other=False
    )
    if axis is None:

        def reduce(xs):
            if len(xs) == 1:
                return xs[0]
            else:
                return xs[0] * reduce(xs[1:])

        return reduce(
            [ak.nplike.of(x).prod(x) for x in ak._util.completely_flatten(layout)]
        )
    else:
        behavior = ak._util.behaviorof(array)
        return ak._util.wrap(
            layout.prod(axis=axis, mask=mask_identity, keepdims=keepdims), behavior
        )


@ak._connect._numpy.implements("any")
def any(array, axis=None, keepdims=False, mask_identity=False):
    """
    Args:
        array: Data to combine with "logical or."
        axis (None or int): If None, combine all values from the array into
            a single scalar result; if an int, group by that axis: `0` is the
            outermost, `1` is the first level of nested lists, etc., and
            negative `axis` counts from the innermost: `-1` is the innermost,
            `-2` is the next level up, etc.
        keepdims (bool): If False, this reducer decreases the number of
            dimensions by 1; if True, the reduced values are wrapped in a new
            length-1 dimension so that the result of this operation may be
            broadcasted with the original array.
        mask_identity (bool): If True, reducing over empty lists results in
            None (an option type); otherwise, reducing over empty lists
            results in the operation's identity.

    Returns True in each group of elements from `array` (many types supported,
    including all Awkward Arrays and Records) if any values are True; False
    otherwise. Thus, it represents reduction over the "logical or" operation,
    whose identity is False (i.e. asking if there are any True values in an
    empty list results in False). This operation is the same as NumPy's
    [any](https://docs.scipy.org/doc/numpy/reference/generated/numpy.any.html)
    if all lists at a given dimension have the same length and no None values,
    but it generalizes to cases where they do not.

    See #ak.sum for a more complete description of nested list and missing
    value (None) handling in reducers.
    """
    layout = ak.operations.convert.to_layout(
        array, allow_record=False, allow_other=False
    )
    if axis is None:

        def reduce(xs):
            if len(xs) == 1:
                return xs[0]
            else:
                return xs[0] or reduce(xs[1:])

        return reduce(
            [ak.nplike.of(x).any(x) for x in ak._util.completely_flatten(layout)]
        )
    else:
        behavior = ak._util.behaviorof(array)
        return ak._util.wrap(
            layout.any(axis=axis, mask=mask_identity, keepdims=keepdims), behavior
        )


@ak._connect._numpy.implements("all")
def all(array, axis=None, keepdims=False, mask_identity=False):
    """
    Args:
        array: Data to combine with "logical and."
        axis (None or int): If None, combine all values from the array into
            a single scalar result; if an int, group by that axis: `0` is the
            outermost, `1` is the first level of nested lists, etc., and
            negative `axis` counts from the innermost: `-1` is the innermost,
            `-2` is the next level up, etc.
        keepdims (bool): If False, this reducer decreases the number of
            dimensions by 1; if True, the reduced values are wrapped in a new
            length-1 dimension so that the result of this operation may be
            broadcasted with the original array.
        mask_identity (bool): If True, reducing over empty lists results in
            None (an option type); otherwise, reducing over empty lists
            results in the operation's identity.

    Returns True in each group of elements from `array` (many types supported,
    including all Awkward Arrays and Records) if all values are True; False
    otherwise. Thus, it represents reduction over the "logical and" operation,
    whose identity is True (i.e. asking if all the values are True in an
    empty list results in True). This operation is the same as NumPy's
    [all](https://docs.scipy.org/doc/numpy/reference/generated/numpy.all.html)
    if all lists at a given dimension have the same length and no None values,
    but it generalizes to cases where they do not.

    See #ak.sum for a more complete description of nested list and missing
    value (None) handling in reducers.
    """
    layout = ak.operations.convert.to_layout(
        array, allow_record=False, allow_other=False
    )
    if axis is None:

        def reduce(xs):
            if len(xs) == 1:
                return xs[0]
            else:
                return xs[0] and reduce(xs[1:])

        return reduce(
            [ak.nplike.of(x).all(x) for x in ak._util.completely_flatten(layout)]
        )
    else:
        behavior = ak._util.behaviorof(array)
        return ak._util.wrap(
            layout.all(axis=axis, mask=mask_identity, keepdims=keepdims), behavior
        )


@ak._connect._numpy.implements("min")
def min(array, axis=None, keepdims=False, initial=None, mask_identity=True):
    """
    Args:
        array: Data to minimize.
        axis (None or int): If None, combine all values from the array into
            a single scalar result; if an int, group by that axis: `0` is the
            outermost, `1` is the first level of nested lists, etc., and
            negative `axis` counts from the innermost: `-1` is the innermost,
            `-2` is the next level up, etc.
        keepdims (bool): If False, this reducer decreases the number of
            dimensions by 1; if True, the reduced values are wrapped in a new
            length-1 dimension so that the result of this operation may be
            broadcasted with the original array.
        initial (None or number): The maximum value of an output element, as
            an alternative to the numeric type's natural identity (e.g. infinity
            for floating-point types, a maximum integer for integer types).
            If you use `initial`, you might also want `mask_identity=False`.
        mask_identity (bool): If True, reducing over empty lists results in
            None (an option type); otherwise, reducing over empty lists
            results in the operation's identity.

    Returns the minimum value in each group of elements from `array` (many
    types supported, including all Awkward Arrays and Records). The identity
    of minimization is `inf` if floating-point or the largest integer value
    if applied to integers. This identity is usually masked: the minimum of
    an empty list is None, unless `mask_identity=False`.
    This operation is the same as NumPy's
    [amin](https://docs.scipy.org/doc/numpy/reference/generated/numpy.amin.html)
    if all lists at a given dimension have the same length and no None values,
    but it generalizes to cases where they do not.

    See #ak.sum for a more complete description of nested list and missing
    value (None) handling in reducers.
    """
    layout = ak.operations.convert.to_layout(
        array, allow_record=False, allow_other=False
    )
    if axis is None:

        def reduce(xs):
            if len(xs) == 0:
                return None
            elif len(xs) == 1:
                return xs[0]
            else:
                x, y = xs[0], reduce(xs[1:])
                return x if x < y else y

        tmp = ak._util.completely_flatten(layout)
        return reduce([ak.nplike.of(x).min(x) for x in tmp if len(x) > 0])
    else:
        behavior = ak._util.behaviorof(array)
        return ak._util.wrap(
            layout.min(
                axis=axis, mask=mask_identity, keepdims=keepdims, initial=initial
            ),
            behavior,
        )


@ak._connect._numpy.implements("max")
def max(array, axis=None, keepdims=False, initial=None, mask_identity=True):
    """
    Args:
        array: Data to maximize.
        axis (None or int): If None, combine all values from the array into
            a single scalar result; if an int, group by that axis: `0` is the
            outermost, `1` is the first level of nested lists, etc., and
            negative `axis` counts from the innermost: `-1` is the innermost,
            `-2` is the next level up, etc.
        keepdims (bool): If False, this reducer decreases the number of
            dimensions by 1; if True, the reduced values are wrapped in a new
            length-1 dimension so that the result of this operation may be
            broadcasted with the original array.
        initial (None or number): The minimum value of an output element, as
            an alternative to the numeric type's natural identity (e.g. negative
            infinity for floating-point types, a minimum integer for integer types).
            If you use `initial`, you might also want `mask_identity=False`.
        mask_identity (bool): If True, reducing over empty lists results in
            None (an option type); otherwise, reducing over empty lists
            results in the operation's identity.

    Returns the maximum value in each group of elements from `array` (many
    types supported, including all Awkward Arrays and Records). The identity
    of maximization is `-inf` if floating-point or the smallest integer value
    if applied to integers. This identity is usually masked: the maximum of
    an empty list is None, unless `mask_identity=False`.
    This operation is the same as NumPy's
    [amax](https://docs.scipy.org/doc/numpy/reference/generated/numpy.amax.html)
    if all lists at a given dimension have the same length and no None values,
    but it generalizes to cases where they do not.

    See #ak.sum for a more complete description of nested list and missing
    value (None) handling in reducers.
    """
    layout = ak.operations.convert.to_layout(
        array, allow_record=False, allow_other=False
    )
    if axis is None:

        def reduce(xs):
            if len(xs) == 0:
                return None
            elif len(xs) == 1:
                return xs[0]
            else:
                x, y = xs[0], reduce(xs[1:])
                return x if x > y else y

        tmp = ak._util.completely_flatten(layout)
        return reduce([ak.nplike.of(x).max(x) for x in tmp if len(x) > 0])
    else:
        behavior = ak._util.behaviorof(array)
        return ak._util.wrap(
            layout.max(
                axis=axis, mask=mask_identity, keepdims=keepdims, initial=initial
            ),
            behavior,
        )


@ak._connect._numpy.implements("ptp")
def ptp(arr, axis=None, keepdims=False, mask_identity=True):
    """
    Args:
        array: Data from which to find the range of values.
        axis (None or int): If None, combine all values from the array into
            a single scalar result; if an int, group by that axis: `0` is the
            outermost, `1` is the first level of nested lists, etc., and
            negative `axis` counts from the innermost: `-1` is the innermost,
            `-2` is the next level up, etc.
        keepdims (bool): If False, this reducer decreases the number of
            dimensions by 1; if True, the reduced values are wrapped in a new
            length-1 dimension so that the result of this operation may be
            broadcasted with the original array.
        mask_identity (bool): If True, reducing over empty lists results in
            None (an option type); otherwise, reducing over empty lists
            results in the operation's identity of 0.

    Returns the range of values in each group of elements from `array` (many
    types supported, including all Awkward Arrays and Records). The range of
    an empty list is None, unless `mask_identity=False`, in which case it is 0.
    This operation is the same as NumPy's
    [ptp](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ptp.html)
    if all lists at a given dimension have the same length and no None values,
    but it generalizes to cases where they do not.

    For example, with an `array` like

        ak.Array([[0, 1, 2, 3],
                  [          ],
                  [4, 5      ]])

    The range of the innermost lists is

        >>> ak.ptp(array, axis=-1)
        <Array [3, None, 1] type='3 * ?float64'>

    because there are three lists, the first has a range of `3`, the second is
    empty, and the third has a range of `1`.

    See #ak.sum for a more complete description of nested list and missing
    value (None) handling in reducers.
    """
    ptp = ak.max(arr, axis=axis, keepdims=keepdims) - ak.min(
        arr, axis=axis, keepdims=keepdims
    )
    if not mask_identity:
        return ak.fill_none(ptp, 0)
    return ptp


@ak._connect._numpy.implements("argmin")
def argmin(array, axis=None, keepdims=False, mask_identity=True):
    """
    Args:
        array: Data to find the index positions of the minimum values.
        axis (None or int): If None, combine all values from the array into
            a single scalar result; if an int, group by that axis: `0` is the
            outermost, `1` is the first level of nested lists, etc., and
            negative `axis` counts from the innermost: `-1` is the innermost,
            `-2` is the next level up, etc.
        keepdims (bool): If False, this reducer decreases the number of
            dimensions by 1; if True, the reduced values are wrapped in a new
            length-1 dimension so that the result of this operation may be
            broadcasted with the original array.
        mask_identity (bool): If True, reducing over empty lists results in
            None (an option type); otherwise, reducing over empty lists
            results in the operation's identity.

    Returns the index position of the minimum value in each group of elements
    from `array` (many types supported, including all Awkward Arrays and
    Records). The identity of minimization would be infinity, but argmin
    must return the position of the minimum element, which has no value for
    empty lists. Therefore, the identity should be masked: the argmin of
    an empty list is None. If `mask_identity=False`, the result would be `-1`,
    which is distinct from all valid index positions, but care should be taken
    that it is not misinterpreted as "the last element of the list."

    This operation is the same as NumPy's
    [argmin](https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmin.html)
    if all lists at a given dimension have the same length and no None values,
    but it generalizes to cases where they do not.

    See #ak.sum for a more complete description of nested list and missing
    value (None) handling in reducers.
    """
    layout = ak.operations.convert.to_layout(
        array, allow_record=False, allow_other=False
    )

    if axis is None:
        if isinstance(layout, ak.partition.PartitionedArray):
            start = 0
            best_index = None
            best_value = None
            for partition in layout.partitions:
                for tmp in ak._util.completely_flatten(partition):
                    out = ak.nplike.of(tmp).argmin(tmp, axis=None)
                    if best_index is None or tmp[out] < best_value:
                        best_index = start + out
                        best_value = tmp[out]
                start += len(partition)
            return best_index

        else:
            best_index = None
            best_value = None
            for tmp in ak._util.completely_flatten(layout):
                out = ak.nplike.of(tmp).argmin(tmp, axis=None)
                if best_index is None or tmp[out] < best_value:
                    best_index = out
                    best_value = tmp[out]
            return best_index

    else:
        behavior = ak._util.behaviorof(array)
        return ak._util.wrap(
            layout.argmin(axis=axis, mask=mask_identity, keepdims=keepdims), behavior
        )


@ak._connect._numpy.implements("argmax")
def argmax(array, axis=None, keepdims=False, mask_identity=True):
    """
    Args:
        array: Data to find the index positions of the maximum values.
        axis (None or int): If None, combine all values from the array into
            a single scalar result; if an int, group by that axis: `0` is the
            outermost, `1` is the first level of nested lists, etc., and
            negative `axis` counts from the innermost: `-1` is the innermost,
            `-2` is the next level up, etc.
        keepdims (bool): If False, this reducer decreases the number of
            dimensions by 1; if True, the reduced values are wrapped in a new
            length-1 dimension so that the result of this operation may be
            broadcasted with the original array.
        mask_identity (bool): If True, reducing over empty lists results in
            None (an option type); otherwise, reducing over empty lists
            results in the operation's identity.

    Returns the index position of the maximum value in each group of elements
    from `array` (many types supported, including all Awkward Arrays and
    Records). The identity of maximization would be negative infinity, but
    argmax must return the position of the maximum element, which has no value
    for empty lists. Therefore, the identity should be masked: the argmax of
    an empty list is None. If `mask_identity=False`, the result would be `-1`,
    which is distinct from all valid index positions, but care should be taken
    that it is not misinterpreted as "the last element of the list."

    This operation is the same as NumPy's
    [argmax](https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html)
    if all lists at a given dimension have the same length and no None values,
    but it generalizes to cases where they do not.

    See #ak.sum for a more complete description of nested list and missing
    value (None) handling in reducers.
    """
    layout = ak.operations.convert.to_layout(
        array, allow_record=False, allow_other=False
    )

    if axis is None:
        if isinstance(layout, ak.partition.PartitionedArray):
            start = 0
            best_index = None
            best_value = None
            for partition in layout.partitions:
                for tmp in ak._util.completely_flatten(partition):
                    out = ak.nplike.of(tmp).argmax(tmp, axis=None)
                    if best_index is None or tmp[out] > best_value:
                        best_index = start + out
                        best_value = tmp[out]
                start += len(partition)
            return best_index

        else:
            best_index = None
            best_value = None
            for tmp in ak._util.completely_flatten(layout):
                out = ak.nplike.of(tmp).argmax(tmp, axis=None)
                if best_index is None or tmp[out] > best_value:
                    best_index = out
                    best_value = tmp[out]
            return best_index

    else:
        behavior = ak._util.behaviorof(array)
        return ak._util.wrap(
            layout.argmax(axis=axis, mask=mask_identity, keepdims=keepdims), behavior
        )


# The following are not strictly reducers, but are defined in terms of
# reducers and ufuncs.


def moment(x, n, weight=None, axis=None, keepdims=False, mask_identity=True):
    """
    Args:
        x: the data on which to compute the moment.
        n (int): the choice of moment: `0` is a sum of weights, `1` is
            #ak.mean, `2` is #ak.var without subtracting the mean, etc.
        weight: data that can be broadcasted to `x` to give each value a
            weight. Weighting values equally is the same as no weights;
            weighting some values higher increases the significance of those
            values. Weights can be zero or negative.
        axis (None or int): If None, combine all values from the array into
            a single scalar result; if an int, group by that axis: `0` is the
            outermost, `1` is the first level of nested lists, etc., and
            negative `axis` counts from the innermost: `-1` is the innermost,
            `-2` is the next level up, etc.
        keepdims (bool): If False, this function decreases the number of
            dimensions by 1; if True, the output values are wrapped in a new
            length-1 dimension so that the result of this operation may be
            broadcasted with the original array.
        mask_identity (bool): If True, the application of this function on
            empty lists results in None (an option type); otherwise, the
            calculation is followed through with the reducers' identities,
            usually resulting in floating-point `nan`.

    Computes the `n`th moment in each group of elements from `x` (many
    types supported, including all Awkward Arrays and Records). The grouping
    is performed the same way as for reducers, though this operation is not a
    reducer and has no identity.

    This function has no NumPy equivalent.

    Passing all arguments to the reducers, the moment is calculated as

        ak.sum((x*weight)**n) / ak.sum(weight)

    The `n=2` moment differs from #ak.var in that #ak.var also subtracts the
    mean (the `n=1` moment).

    See #ak.sum for a complete description of handling nested lists and
    missing values (None) in reducers, and #ak.mean for an example with another
    non-reducer.
    """
    with np.errstate(invalid="ignore"):
        if weight is None:
            sumw = count(x, axis=axis, keepdims=keepdims, mask_identity=mask_identity)
            sumwxn = sum(
                x ** n, axis=axis, keepdims=keepdims, mask_identity=mask_identity
            )
        else:
            sumw = sum(
                x * 0 + weight,
                axis=axis,
                keepdims=keepdims,
                mask_identity=mask_identity,
            )
            sumwxn = sum(
                (x * weight) ** n,
                axis=axis,
                keepdims=keepdims,
                mask_identity=mask_identity,
            )
        return ak.nplike.of(sumwxn, sumw).true_divide(sumwxn, sumw)


@ak._connect._numpy.implements("mean")
def mean(x, weight=None, axis=None, keepdims=False, mask_identity=True):
    """
    Args:
        x: the data on which to compute the mean.
        weight: data that can be broadcasted to `x` to give each value a
            weight. Weighting values equally is the same as no weights;
            weighting some values higher increases the significance of those
            values. Weights can be zero or negative.
        axis (None or int): If None, combine all values from the array into
            a single scalar result; if an int, group by that axis: `0` is the
            outermost, `1` is the first level of nested lists, etc., and
            negative `axis` counts from the innermost: `-1` is the innermost,
            `-2` is the next level up, etc.
        keepdims (bool): If False, this function decreases the number of
            dimensions by 1; if True, the output values are wrapped in a new
            length-1 dimension so that the result of this operation may be
            broadcasted with the original array.
        mask_identity (bool): If True, the application of this function on
            empty lists results in None (an option type); otherwise, the
            calculation is followed through with the reducers' identities,
            usually resulting in floating-point `nan`.

    Computes the mean in each group of elements from `x` (many
    types supported, including all Awkward Arrays and Records). The grouping
    is performed the same way as for reducers, though this operation is not a
    reducer and has no identity. It is the same as NumPy's
    [mean](https://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html)
    if all lists at a given dimension have the same length and no None values,
    but it generalizes to cases where they do not.

    Passing all arguments to the reducers, the mean is calculated as

        ak.sum(x*weight) / ak.sum(weight)

    For example, with an `array` like

        ak.Array([[0, 1, 2, 3],
                  [          ],
                  [4, 5      ]])

    The mean of the innermost lists is

        >>> ak.mean(array, axis=-1)
        <Array [1.5, None, 4.5] type='3 * ?float64'>

    because there are three lists, the first has mean `1.5`, the second is
    empty, and the third has mean `4.5`.

    The mean of the outermost lists is

        >>> ak.mean(array, axis=0)
        <Array [2, 3, 2, 3] type='4 * ?float64'>

    because the longest list has length 4, the mean of `0` and `4` is `2.0`,
    the mean of `1` and `5` is `3.0`, the mean of `2` (by itself) is `2.0`,
    and the mean of `3` (by itself) is `3.0`. This follows the same grouping
    behavior as reducers.

    See #ak.sum for a complete description of handling nested lists and
    missing values (None) in reducers.
    """
    with np.errstate(invalid="ignore"):
        if weight is None:
            sumw = count(x, axis=axis, keepdims=keepdims, mask_identity=mask_identity)
            sumwx = sum(x, axis=axis, keepdims=keepdims, mask_identity=mask_identity)
        else:
            sumw = sum(
                x * 0 + weight,
                axis=axis,
                keepdims=keepdims,
                mask_identity=mask_identity,
            )
            sumwx = sum(
                x * weight, axis=axis, keepdims=keepdims, mask_identity=mask_identity
            )
        return ak.nplike.of(sumwx, sumw).true_divide(sumwx, sumw)


@ak._connect._numpy.implements("var")
def var(x, weight=None, ddof=0, axis=None, keepdims=False, mask_identity=True):
    """
    Args:
        x: the data on which to compute the variance.
        weight: data that can be broadcasted to `x` to give each value a
            weight. Weighting values equally is the same as no weights;
            weighting some values higher increases the significance of those
            values. Weights can be zero or negative.
        ddof (int): "delta degrees of freedom": the divisor used in the
            calculation is `sum(weights) - ddof`. Use this for "reduced
            variance."
        axis (None or int): If None, combine all values from the array into
            a single scalar result; if an int, group by that axis: `0` is the
            outermost, `1` is the first level of nested lists, etc., and
            negative `axis` counts from the innermost: `-1` is the innermost,
            `-2` is the next level up, etc.
        keepdims (bool): If False, this function decreases the number of
            dimensions by 1; if True, the output values are wrapped in a new
            length-1 dimension so that the result of this operation may be
            broadcasted with the original array.
        mask_identity (bool): If True, the application of this function on
            empty lists results in None (an option type); otherwise, the
            calculation is followed through with the reducers' identities,
            usually resulting in floating-point `nan`.

    Computes the variance in each group of elements from `x` (many
    types supported, including all Awkward Arrays and Records). The grouping
    is performed the same way as for reducers, though this operation is not a
    reducer and has no identity. It is the same as NumPy's
    [var](https://docs.scipy.org/doc/numpy/reference/generated/numpy.var.html)
    if all lists at a given dimension have the same length and no None values,
    but it generalizes to cases where they do not.

    Passing all arguments to the reducers, the variance is calculated as

        ak.sum((x - ak.mean(x))**2 * weight) / ak.sum(weight)

    If `ddof` is not zero, the above is further corrected by a factor of

        ak.sum(weight) / (ak.sum(weight) - ddof)

    Even without `ddof`, #ak.var differs from #ak.moment with `n=2` because
    the mean is subtracted from all points before summing their squares.

    See #ak.sum for a complete description of handling nested lists and
    missing values (None) in reducers, and #ak.mean for an example with another
    non-reducer.
    """
    with np.errstate(invalid="ignore"):
        xmean = mean(
            x, weight=weight, axis=axis, keepdims=keepdims, mask_identity=mask_identity
        )
        if weight is None:
            sumw = count(x, axis=axis, keepdims=keepdims, mask_identity=mask_identity)
            sumwxx = sum(
                (x - xmean) ** 2,
                axis=axis,
                keepdims=keepdims,
                mask_identity=mask_identity,
            )
        else:
            sumw = sum(
                x * 0 + weight,
                axis=axis,
                keepdims=keepdims,
                mask_identity=mask_identity,
            )
            sumwxx = sum(
                (x - xmean) ** 2 * weight,
                axis=axis,
                keepdims=keepdims,
                mask_identity=mask_identity,
            )
        if ddof != 0:
            return ak.nplike.of(sumwxx, sumw).true_divide(sumwxx, sumw) * ak.nplike.of(
                sumw
            ).true_divide(sumw, sumw - ddof)
        else:
            return ak.nplike.of(sumwxx, sumw).true_divide(sumwxx, sumw)


@ak._connect._numpy.implements("std")
def std(x, weight=None, ddof=0, axis=None, keepdims=False, mask_identity=True):
    """
    Args:
        x: the data on which to compute the standard deviation.
        weight: data that can be broadcasted to `x` to give each value a
            weight. Weighting values equally is the same as no weights;
            weighting some values higher increases the significance of those
            values. Weights can be zero or negative.
        ddof (int): "delta degrees of freedom": the divisor used in the
            calculation is `sum(weights) - ddof`. Use this for "reduced
            standard deviation."
        axis (None or int): If None, combine all values from the array into
            a single scalar result; if an int, group by that axis: `0` is the
            outermost, `1` is the first level of nested lists, etc., and
            negative `axis` counts from the innermost: `-1` is the innermost,
            `-2` is the next level up, etc.
        keepdims (bool): If False, this function decreases the number of
            dimensions by 1; if True, the output values are wrapped in a new
            length-1 dimension so that the result of this operation may be
            broadcasted with the original array.
        mask_identity (bool): If True, the application of this function on
            empty lists results in None (an option type); otherwise, the
            calculation is followed through with the reducers' identities,
            usually resulting in floating-point `nan`.

    Computes the standard deviation in each group of elements from `x`
    (many types supported, including all Awkward Arrays and Records). The
    grouping is performed the same way as for reducers, though this operation
    is not a reducer and has no identity. It is the same as NumPy's
    [std](https://docs.scipy.org/doc/numpy/reference/generated/numpy.std.html)
    if all lists at a given dimension have the same length and no None values,
    but it generalizes to cases where they do not.

    Passing all arguments to the reducers, the standard deviation is
    calculated as

        np.sqrt(ak.var(x, weight))

    See #ak.sum for a complete description of handling nested lists and
    missing values (None) in reducers, and #ak.mean for an example with another
    non-reducer.
    """
    with np.errstate(invalid="ignore"):
        return ak.nplike.of(x, weight).sqrt(
            var(
                x,
                weight=weight,
                ddof=ddof,
                axis=axis,
                keepdims=keepdims,
                mask_identity=mask_identity,
            )
        )


def covar(x, y, weight=None, axis=None, keepdims=False, mask_identity=True):
    """
    Args:
        x: one coordinate to use in the covariance calculation.
        y: the other coordinate to use in the covariance calculation.
        weight: data that can be broadcasted to `x` and `y` to give each point
            a weight. Weighting points equally is the same as no weights;
            weighting some points higher increases the significance of those
            points. Weights can be zero or negative.
        axis (None or int): If None, combine all values from the array into
            a single scalar result; if an int, group by that axis: `0` is the
            outermost, `1` is the first level of nested lists, etc., and
            negative `axis` counts from the innermost: `-1` is the innermost,
            `-2` is the next level up, etc.
        keepdims (bool): If False, this function decreases the number of
            dimensions by 1; if True, the output values are wrapped in a new
            length-1 dimension so that the result of this operation may be
            broadcasted with the original array.
        mask_identity (bool): If True, the application of this function on
            empty lists results in None (an option type); otherwise, the
            calculation is followed through with the reducers' identities,
            usually resulting in floating-point `nan`.

    Computes the covariance of `x` and `y` (many types supported, including
    all Awkward Arrays and Records, must be broadcastable to each other).
    The grouping is performed the same way as for reducers, though this
    operation is not a reducer and has no identity.

    This function has no NumPy equivalent.

    Passing all arguments to the reducers, the covariance is calculated as

        ak.sum((x - ak.mean(x))*(y - ak.mean(y))*weight) / ak.sum(weight)

    See #ak.sum for a complete description of handling nested lists and
    missing values (None) in reducers, and #ak.mean for an example with another
    non-reducer.
    """
    with np.errstate(invalid="ignore"):
        xmean = mean(
            x, weight=weight, axis=axis, keepdims=keepdims, mask_identity=mask_identity
        )
        ymean = mean(
            y, weight=weight, axis=axis, keepdims=keepdims, mask_identity=mask_identity
        )
        if weight is None:
            sumw = count(x, axis=axis, keepdims=keepdims, mask_identity=mask_identity)
            sumwxy = sum(
                (x - xmean) * (y - ymean),
                axis=axis,
                keepdims=keepdims,
                mask_identity=mask_identity,
            )
        else:
            sumw = sum(
                x * 0 + weight,
                axis=axis,
                keepdims=keepdims,
                mask_identity=mask_identity,
            )
            sumwxy = sum(
                (x - xmean) * (y - ymean) * weight,
                axis=axis,
                keepdims=keepdims,
                mask_identity=mask_identity,
            )
        return ak.nplike.of(sumwxy, sumw).true_divide(sumwxy, sumw)


def corr(x, y, weight=None, axis=None, keepdims=False, mask_identity=True):
    """
    Args:
        x: one coordinate to use in the correlation.
        y: the other coordinate to use in the correlation.
        weight: data that can be broadcasted to `x` and `y` to give each point
            a weight. Weighting points equally is the same as no weights;
            weighting some points higher increases the significance of those
            points. Weights can be zero or negative.
        axis (None or int): If None, combine all values from the array into
            a single scalar result; if an int, group by that axis: `0` is the
            outermost, `1` is the first level of nested lists, etc., and
            negative `axis` counts from the innermost: `-1` is the innermost,
            `-2` is the next level up, etc.
        keepdims (bool): If False, this function decreases the number of
            dimensions by 1; if True, the output values are wrapped in a new
            length-1 dimension so that the result of this operation may be
            broadcasted with the original array.
        mask_identity (bool): If True, the application of this function on
            empty lists results in None (an option type); otherwise, the
            calculation is followed through with the reducers' identities,
            usually resulting in floating-point `nan`.

    Computes the correlation of `x` and `y` (many types supported, including
    all Awkward Arrays and Records, must be broadcastable to each other).
    The grouping is performed the same way as for reducers, though this
    operation is not a reducer and has no identity.

    This function has no NumPy equivalent.

    Passing all arguments to the reducers, the correlation is calculated as

        ak.sum((x - ak.mean(x))*(y - ak.mean(y))*weight)
            / np.sqrt(ak.sum((x - ak.mean(x))**2))
            / np.sqrt(ak.sum((y - ak.mean(y))**2))

    See #ak.sum for a complete description of handling nested lists and
    missing values (None) in reducers, and #ak.mean for an example with another
    non-reducer.
    """
    with np.errstate(invalid="ignore"):
        xmean = mean(
            x, weight=weight, axis=axis, keepdims=keepdims, mask_identity=mask_identity
        )
        ymean = mean(
            y, weight=weight, axis=axis, keepdims=keepdims, mask_identity=mask_identity
        )
        xdiff = x - xmean
        ydiff = y - ymean
        if weight is None:
            sumwxx = sum(
                xdiff ** 2, axis=axis, keepdims=keepdims, mask_identity=mask_identity
            )
            sumwyy = sum(
                ydiff ** 2, axis=axis, keepdims=keepdims, mask_identity=mask_identity
            )
            sumwxy = sum(
                xdiff * ydiff, axis=axis, keepdims=keepdims, mask_identity=mask_identity
            )
        else:
            sumwxx = sum(
                (xdiff ** 2) * weight,
                axis=axis,
                keepdims=keepdims,
                mask_identity=mask_identity,
            )
            sumwyy = sum(
                (ydiff ** 2) * weight,
                axis=axis,
                keepdims=keepdims,
                mask_identity=mask_identity,
            )
            sumwxy = sum(
                (xdiff * ydiff) * weight,
                axis=axis,
                keepdims=keepdims,
                mask_identity=mask_identity,
            )
        nplike = ak.nplike.of(sumwxy, sumwxx, sumwyy)
        return nplike.true_divide(sumwxy, nplike.sqrt(sumwxx * sumwyy))


def linear_fit(x, y, weight=None, axis=None, keepdims=False, mask_identity=True):
    """
    Args:
        x: one coordinate to use in the linear fit.
        y: the other coordinate to use in the linear fit.
        weight: data that can be broadcasted to `x` and `y` to give each point
            a weight. Weighting points equally is the same as no weights;
            weighting some points higher increases the significance of those
            points. Weights can be zero or negative.
        axis (None or int): If None, combine all values from the array into
            a single scalar result; if an int, group by that axis: `0` is the
            outermost, `1` is the first level of nested lists, etc., and
            negative `axis` counts from the innermost: `-1` is the innermost,
            `-2` is the next level up, etc.
        keepdims (bool): If False, this function decreases the number of
            dimensions by 1; if True, the output values are wrapped in a new
            length-1 dimension so that the result of this operation may be
            broadcasted with the original array.
        mask_identity (bool): If True, the application of this function on
            empty lists results in None (an option type); otherwise, the
            calculation is followed through with the reducers' identities,
            usually resulting in floating-point `nan`.

    Computes the linear fit of `y` with respect to `x` (many types supported,
    including all Awkward Arrays and Records, must be broadcastable to each
    other). The grouping is performed the same way as for reducers, though
    this operation is not a reducer and has no identity.

    This function has no NumPy equivalent.

    Passing all arguments to the reducers, the linear fit is calculated as

        sumw            = ak.sum(weight)
        sumwx           = ak.sum(weight * x)
        sumwy           = ak.sum(weight * y)
        sumwxx          = ak.sum(weight * x**2)
        sumwxy          = ak.sum(weight * x * y)
        delta           = (sumw*sumwxx) - (sumwx*sumwx)

        intercept       = ((sumwxx*sumwy) - (sumwx*sumwxy)) / delta
        slope           = ((sumw*sumwxy) - (sumwx*sumwy))   / delta
        intercept_error = np.sqrt(sumwxx / delta)
        slope_error     = np.sqrt(sumw   / delta)

    The results, `intercept`, `slope`, `intercept_error`, and `slope_error`,
    are given as an #ak.Record with four fields. The values of these fields
    might be arrays or even nested arrays; they match the structure of `x` and
    `y`.

    See #ak.sum for a complete description of handling nested lists and
    missing values (None) in reducers, and #ak.mean for an example with another
    non-reducer.
    """
    with np.errstate(invalid="ignore"):
        nplike = ak.nplike.of(x, y, weight)
        if weight is None:
            sumw = count(x, axis=axis, keepdims=keepdims, mask_identity=mask_identity)
            sumwx = sum(x, axis=axis, keepdims=keepdims, mask_identity=mask_identity)
            sumwy = sum(y, axis=axis, keepdims=keepdims, mask_identity=mask_identity)
            sumwxx = sum(
                x ** 2, axis=axis, keepdims=keepdims, mask_identity=mask_identity
            )
            sumwxy = sum(
                x * y, axis=axis, keepdims=keepdims, mask_identity=mask_identity
            )
        else:
            sumw = sum(
                x * 0 + weight,
                axis=axis,
                keepdims=keepdims,
                mask_identity=mask_identity,
            )
            sumwx = sum(
                x * weight, axis=axis, keepdims=keepdims, mask_identity=mask_identity
            )
            sumwy = sum(
                y * weight, axis=axis, keepdims=keepdims, mask_identity=mask_identity
            )
            sumwxx = sum(
                (x ** 2) * weight,
                axis=axis,
                keepdims=keepdims,
                mask_identity=mask_identity,
            )
            sumwxy = sum(
                x * y * weight,
                axis=axis,
                keepdims=keepdims,
                mask_identity=mask_identity,
            )
        delta = (sumw * sumwxx) - (sumwx * sumwx)
        intercept = nplike.true_divide(((sumwxx * sumwy) - (sumwx * sumwxy)), delta)
        slope = nplike.true_divide(((sumw * sumwxy) - (sumwx * sumwy)), delta)
        intercept_error = nplike.sqrt(nplike.true_divide(sumwxx, delta))
        slope_error = nplike.sqrt(nplike.true_divide(sumw, delta))

        intercept = ak.operations.convert.to_layout(
            intercept, allow_record=True, allow_other=True
        )
        slope = ak.operations.convert.to_layout(
            slope, allow_record=True, allow_other=True
        )
        intercept_error = ak.operations.convert.to_layout(
            intercept_error, allow_record=True, allow_other=True
        )
        slope_error = ak.operations.convert.to_layout(
            slope_error, allow_record=True, allow_other=True
        )

        scalar = False
        if not isinstance(
            intercept,
            (
                ak.layout.Content,
                ak.layout.Record,
                ak.partition.PartitionedArray,
            ),
        ):
            intercept = ak.layout.NumpyArray(nplike.array([intercept]))
            scalar = True
        if not isinstance(
            slope,
            (
                ak.layout.Content,
                ak.layout.Record,
                ak.partition.PartitionedArray,
            ),
        ):
            slope = ak.layout.NumpyArray(nplike.array([slope]))
            scalar = True
        if not isinstance(
            intercept_error,
            (
                ak.layout.Content,
                ak.layout.Record,
                ak.partition.PartitionedArray,
            ),
        ):
            intercept_error = ak.layout.NumpyArray(nplike.array([intercept_error]))
            scalar = True
        if not isinstance(
            slope_error,
            (
                ak.layout.Content,
                ak.layout.Record,
                ak.partition.PartitionedArray,
            ),
        ):
            slope_error = ak.layout.NumpyArray(nplike.array([slope_error]))
            scalar = True

        sample = None
        if isinstance(intercept, ak.partition.PartitionedArray):
            sample = intercept
        elif isinstance(slope, ak.partition.PartitionedArray):
            sample = slope
        elif isinstance(intercept_error, ak.partition.PartitionedArray):
            sample = intercept_error
        elif isinstance(slope_error, ak.partition.PartitionedArray):
            sample = slope_error

        if sample is not None:
            (
                intercept,
                slope,
                intercept_error,
                slope_error,
            ) = ak.partition.partition_as(
                sample, (intercept, slope, intercept_error, slope_error)
            )
            output = []
            for a, b, c, d in ak.partition.iterate(
                sample.numpartitions, (intercept, slope, intercept_error, slope_error)
            ):
                output.append(
                    ak.layout.RecordArray(
                        [a, b, c, d],
                        ["intercept", "slope", "intercept_error", "slope_error"],
                        parameters={"__record__": "LinearFit"},
                    )
                )
            out = ak.partition.IrregularlyPartitionedArray(output)

        else:
            out = ak.layout.RecordArray(
                [intercept, slope, intercept_error, slope_error],
                ["intercept", "slope", "intercept_error", "slope_error"],
                parameters={"__record__": "LinearFit"},
            )
            if scalar:
                out = out[0]

        return ak._util.wrap(out, ak._util.behaviorof(x, y))


def softmax(x, axis=None, keepdims=False, mask_identity=False):
    """
    Args:
        x: the data on which to compute the softmax.
        weight: data that can be broadcasted to `x` to give each value a
            weight. Weighting values equally is the same as no weights;
            weighting some values higher increases the significance of those
            values. Weights can be zero or negative.
        axis (None or int): If None, combine all values from the array into
            a single scalar result; if an int, group by that axis: `0` is the
            outermost, `1` is the first level of nested lists, etc., and
            negative `axis` counts from the innermost: `-1` is the innermost,
            `-2` is the next level up, etc.
        keepdims (bool): If False, this function decreases the number of
            dimensions by 1; if True, the output values are wrapped in a new
            length-1 dimension so that the result of this operation may be
            broadcasted with the original array.
        mask_identity (bool): If True, the application of this function on
            empty lists results in None (an option type); otherwise, the
            calculation is followed through with the reducers' identities,
            usually resulting in floating-point `nan`.

    Computes the softmax in each group of elements from `x` (many
    types supported, including all Awkward Arrays and Records). The grouping
    is performed the same way as for reducers, though this operation is not a
    reducer and has no identity.

    This function has no NumPy equivalent.

    Passing all arguments to the reducers, the softmax is calculated as

        np.exp(x) / ak.sum(np.exp(x))

    See #ak.sum for a complete description of handling nested lists and
    missing values (None) in reducers, and #ak.mean for an example with another
    non-reducer.
    """
    with np.errstate(invalid="ignore"):
        nplike = ak.nplike.of(x)
        expx = nplike.exp(x)
        denom = sum(expx, axis=axis, keepdims=keepdims, mask_identity=mask_identity)
        return nplike.true_divide(expx, denom)


__all__ = [
    x for x in list(globals()) if not x.startswith("_") and x not in ("ak", "np")
]


def __dir__():
    return __all__
