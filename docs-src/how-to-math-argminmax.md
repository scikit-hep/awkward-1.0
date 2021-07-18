---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

How to use argmin and argmax
============================

```{code-cell} ipython3
import awkward as ak
```

The arguments of the maxima - `argmax`, and the arguments of the minima - `argmin` are the index positions of the [ak.Array](https://awkward-array.readthedocs.io/en/latest/_auto/ak.Array.html) maximum or minimum elements over an axis.

The result has the same shape with the dimention along the axis removed.

By default, reducing over empty lists results in `None`.

```{code-cell} ipython3
array = ak.Array([[], [], []])
ak.argmin(array, mask_identity=False, axis=1)
```

```{code-cell} ipython3
ak.argmin(array, mask_identity=True, axis=1)
```

By default, the axis is `None`. In this case the function returns a scalar that is calculated over all the elements in the flattened [ak.Array](https://awkward-array.readthedocs.io/en/latest/_auto/ak.Array.html).

Argmin
------

[ak.argmin](https://awkward-array.readthedocs.io/en/latest/_auto/ak.argmin.html) returns the index position of the minimum value of the [ak.Array](https://awkward-array.readthedocs.io/en/latest/_auto/ak.Array.html) elements over a given axis.

In case of multiple occurencies of the minimum values, the indices corresponding to the first occurence are returned.

The identity of minimization would be infinity, but `argmin` must return the position of the minimum element, which has no value for empty lists. Therefore, the identity should be masked: the `argmin` of an empty list is `None`. If `mask_identity=False`, the result would be `-1`, which is distinct from all valid index positions, but care should be taken that it is not misinterpreted as "the last element of the list."

[![Argmin at axis](img/argmin.svg)](img/argmin.svg)

```{code-cell} ipython3
ak.argmin(array = ak.Array([[2, 3, 5], [], [None, 7], [11]]), axis=0)
```

```{code-cell} ipython3
ak.argmin(array = ak.Array([[2, 3, 5], [], [None, 7], [11]]), axis=1)
```

```{code-cell} ipython3
ak.argmin(array = ak.Array([[2, 3, 5], [], [None, 7], [11]]))
```

This operation is the same as NumPy's [argmin](https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmin.html) if all lists at a given dimension have the same length and no `None` values, but it generalizes to cases where they do not.


Argmax
------

[ak.argmax](https://awkward-array.readthedocs.io/en/latest/_auto/ak.argmax.html) returns the index position of the maximum value of the [ak.Array](https://awkward-array.readthedocs.io/en/latest/_auto/ak.Array.html) elements over a given axis.

In case of multiple occurencies of the maximum values, the indices corresponding to the first occurence are returned.

The identity of maximization would be negative infinity, but `argmax` must return the position of the maximum element, which has no value for empty lists. Therefore, the identity should be masked: the `argmax` of an empty list is `None`. If `mask_identity=False`, the result would be `-1`, which is distinct from all valid index positions, but care should be taken that it is not misinterpreted as "the last element of the list."

[![Argmax at axis](img/argmax.svg)](img/argmax.svg)

```{code-cell} ipython3
ak.argmax(array = ak.Array([[2, 3, 5], [], [None, 7], [11]]), axis=0)
```

```{code-cell} ipython3
ak.argmax(array = ak.Array([[2, 3, 5], [], [None, 7], [11]]), axis=1)
```

```{code-cell} ipython3
ak.argmax(array = ak.Array([[2, 3, 5], [], [None, 7], [11]]))
```

This operation is the same as NumPy's [argmax](https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html) if all lists at a given dimension have the same length and no `None` values, but it generalizes to cases where they do not.

Broadcasting using `argmax` and `argmin` results
------------------------------------------------

The original [ak.Array](https://awkward-array.readthedocs.io/en/latest/_auto/ak.Array.html) can be indexed by the result of an `argmax` or `argmin` operation if the reduced values are wrapped in a new dimention by setting `keepdims=True`.

```{code-cell} ipython3
array = ak.Array([[0.0, 0.8414709848078965, 0.9092974268256817, 0.1411200080598672], [], [None, -0.7568024953079282, -0.9589242746631385, -0.27941549819892586, 0.6569865987187891, None], [0.9893582466233818, 0.4121184852417566, -0.5440211108893699, -0.9999902065507035], None, [-0.5365729180004349, 0.4201670368266409, 0.9906073556948704, 0.6502878401571169, -0.2879033166650653, -0.9613974918795568, -0.750987246771676, 0.14987720966295234]])
ak.to_list(array)
```

```{code-cell} ipython3
index = ak.argmax(array, axis=-1, keepdims=True)
ak.to_list(index)
```

```{code-cell} ipython3
ak.to_list(array[index])
```
