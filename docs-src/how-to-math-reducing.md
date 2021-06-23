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

How to reduce dimensions (sum/min/any/all)
==========================================

```{code-cell} ipython3
import awkward as ak
```

Product
-------

[ak.prod](https://awkward-array.readthedocs.io/en/latest/_auto/ak.prod.html) returns the product of array elements over a given axis.

[![Reducer axis](img/product.svg)](img/product.svg)

```{code-cell} ipython3
ak.prod(array = ak.Array([[2, 3, 5], [], [None, 7], [11]]), axis=0)
```

```{code-cell} ipython3
ak.prod(array = ak.Array([[2, 3, 5], [], [None, 7], [11]]), axis=1)
```

By default `axis=None`. In this case `ak.prod` will calculate the product of all the elements in the input array:

```{code-cell} ipython3
ak.prod(array = ak.Array([[2, 3, 5], [], [None, 7], [11]]))
```

Sum
---

[ak.sum](https://awkward-array.readthedocs.io/en/latest/_auto/ak.sum.html)

Min
---

[ak.min](https://awkward-array.readthedocs.io/en/latest/_auto/ak.min.html)

Max
---

[ak.max](https://awkward-array.readthedocs.io/en/latest/_auto/ak.max.html)

Any
---

[ak.any](https://awkward-array.readthedocs.io/en/latest/_auto/ak.any.html)

All
---

[ak.all](https://awkward-array.readthedocs.io/en/latest/_auto/ak.all.html)

Argmin
------

[ak.argmin](https://awkward-array.readthedocs.io/en/latest/_auto/ak.argmin.html)

Argmax
------

[ak.argmax](https://awkward-array.readthedocs.io/en/latest/_auto/ak.argmax.html)

Count
_____

[ak.count](https://awkward-array.readthedocs.io/en/latest/_auto/ak.count.html)

Count non zero
--------------

[ak.count_nonzero](https://awkward-array.readthedocs.io/en/latest/_auto/ak.count_nonzero.html)

Range of values (maximum - minimum)
-----------------------------------

Range of values (maximum - minimum) along an axis.

[ak.ptp](https://awkward-array.readthedocs.io/en/latest/_auto/ak.ptp.html) returns the range of values in each group of elements from Awkward Array. By default the range of an empty list is `None`, unless `mask_identity=False`, in which case it is 0.

This operation is similar to NumPy's
    [ptp](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ptp.html)

```{code-cell} ipython3
array = ak.Array([[0, 1, 2, 3],
                  [          ],
                  [4, 5      ]])
#ak.ptp(array)
```
