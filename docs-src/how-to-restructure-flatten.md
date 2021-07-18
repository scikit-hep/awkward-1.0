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

How to flatten arrays, especially for plotting
==============================================

In a data analysis, it is important to plot your data frequently, and the interactive nature of array-at-a-time functions facilitate that.

However, plotting views your data as a generic set or sequence—the structure of nested lists and records can't be captured by standard plots. Histograms (including 2-dimensional heatmaps) take input data to be an unordered set, as do scatter plots. Connected-line plots, such as time-series, use the sequential order of the data, but there aren't many visualizations that show nestedness. (Maybe there will be, in the future.)

As such, these standard plotting routines expect simple structures, either a single flat array (in which the order may be relevant or irrelevant) or several same-length arrays (in which the relative or absolute order is relevant). Encountering an Awkward Array, they may try to call `np.asarray` on it, which only works if the array can be made rectilinear or they may try to iterate over it in Python, which can be prohibitively slow if the dataset is large.

+++

Scope of destructuring
----------------------

To destructure an array for plotting, you'll want to

   * remove nested lists, definitely for variable-length ones ("`var *`" in the type string) and possibly for regular ones as well ("`N *`" in the type string, where `N` is an integer),
   * remove record structures,
   * remove missing data

There is a function that does all of these things in one call, [ak.ravel](https://awkward-array.readthedocs.io/en/latest/_auto/ak.ravel.html), but you don't want to apply that without thinking because structure is important to the meaning of your data and you want to be able to interpret the plot. Destructuring is an information-losing operation, so your guidance is required to eliminate exactly the structure you want to eliminate, and there are several ways to do that, depending on what you want to do.

When more precision is required, such as removing only the nested lists and missing data at a specific level of nesting, an alternative function, [ak.flatten](https://awkward-array.readthedocs.io/en/latest/_auto/ak.flatten.html), is more appropriate. Unlike `ak.ravel`, `ak.flatten` accepts an `axis` parameter which restricts the operation to a particular level of nesting, and preserves the record structure.

After destructuring, you might _still_ need to call `np.asarray` on the output because the plotting library might not recognize an [ak.Array](https://awkward-array.readthedocs.io/en/latest/_auto/ak.Array.html) as an array. You'll probably also want to develop your destructuring on a commandline or a different Jupyter cell from the plotting library function call, to understand what structure the output has without the added complication of the plotting library's error messages.

```{code-cell} ipython3
import awkward as ak
import numpy as np
```

ak.ravel vs ak.flatten(axis="records")
--------------------------------------

As mentioned above, [ak.ravel](https://awkward-array.readthedocs.io/en/latest/_auto/ak.ravel.html) is the sledgehammer that turns any array into a 1-dimensional array with no nested lists, no nested records, and no missing data. 

```{code-cell} ipython3
array = ak.Array([[{"x": 1.1, "y": [1]}, {"x": None, "y": [1, 2]}], [], [{"x": 3.3, "y": [1, 2, 3]}]])
array
```

```{code-cell} ipython3
array.type
```

```{code-cell} ipython3
ak.ravel(array)
```

Calling this function on an already flat array does nothing, so you don't have to worry about what state your array had been in before you called it.

```{code-cell} ipython3
ak.ravel(ak.ravel(array))
```

However, there are a few questions you should be asking yourself:

   * Did the nested lists have special meaning? What does the plot represent if I just concatenate them all?
   * Did the record fields have distinct meanings? In this example, what does it mean to put floating-point _x_ values and nested-list _y_ values in the same bucket of numbers to plot? Does it matter that there are more _y_ values than _x_ values? **In most circumstances, you do not want to mix record fields in a plot.**
   * It's likely that we do want to ignore all the missing data, but does dropping them mean that an array representing x-axis values has lost its alignment with an array representing y-axis values?

+++

Selecting record fields
-----------------------

A more controlled way to extract fields from a record is to [project](https://awkward-array.readthedocs.io/en/latest/_auto/ak.Array.html#projection) them by name.

```{code-cell} ipython3
array = ak.Array([[{"x": 1.1, "y": [1], "z": "one"}, {"x": None, "y": [1, 2], "z": "two"}], [], [{"x": 3.3, "y": [1, 2, 3], "z": "three"}]])
array
```

If we want only the _x_ field, we can ask for it as an attribute (because it's a valid Python name) or with a string-valued slice:

```{code-cell} ipython3
array.x
```

```{code-cell} ipython3
array["x"]
```

This controls the biggest deficiency of [ak.ravel](https://awkward-array.readthedocs.io/en/latest/_auto/ak.flatten.html), the mixing of data with different meanings.

```{code-cell} ipython3
ak.ravel(array.x)
```

```{code-cell} ipython3
ak.ravel(array.y)
```

If some of your fields can be safely flattened—together into one set—and others can't, you can use a list of strings to pick just the fields you want.

```{code-cell} ipython3
ak.ravel(array[["x", "y"]])
```

(Careful! A tuple has a special meaning in slices, which doesn't apply here.)

```{code-cell} ipython3
:tags: [raises-exception]

array[("x", "y")]
```

If you have records inside of records, you can extract them with [nested projection](https://awkward-array.readthedocs.io/en/latest/_auto/ak.Array.html#nested-projection) if they have common names.

```{code-cell} ipython3
array = ak.Array([
    {"x": {"up": 1, "down": -1}, "y": {"up": 1.1, "down": -1.1}},
    {"x": {"up": 2, "down": -2}, "y": {"up": 2.2, "down": -2.2}},
    {"x": {"up": 3, "down": -3}, "y": {"up": 3.3, "down": -3.3}},
    {"x": {"up": 4, "down": -4}, "y": {"up": 4.4, "down": -4.4}},
])
array
```

```{code-cell} ipython3
ak.ravel(array[["x", "y"], "up"])
```


ak.flatten(axis="records")
--------------------------

[ak.flatten](https://awkward-array.readthedocs.io/en/latest/_auto/ak.flatten.html) also provides a means of completely flattening an array with `axis="records". Unlike [ak.ravel](https://awkward-array.readthedocs.io/en/latest/_auto/ak.ravel.html), it preserves the record structure.

```{code-cell} ipython3
ak.flatten(array, axis="records")
```

This is often useful when several fields need to be flattened simultaneously. 

```{code-cell} ipython3
array = ak.Array([
    [{"x": 1, "y": 3.3}, {"x": 2, "y": 1.1}, {"x": 3, "y": 2.2}],
    [],
    [{"x": 4, "y": 5.5}, {"x": 5, "y": 4.4}],
    [{"x": 5, "y": 1.1}, {"x": 4, "y": 3.3}, {"x": 2, "y": 5.5}, {"x": 1, "y": 4.4}],
])
array
```

```{code-cell} ipython3
flattened = ak.flatten(array, axis="records")
flattened
```

```{code-cell} ipython3
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.scatter(flattened.x, flattened.y);
```

When there are records-within-records, the flattened array keeps the record structure

```{code-cell} ipython3
array = ak.Array([
    {"x": {"up": 1, "down": -1}, "y": {"up": 1.1, "down": -1.1}},
    {"x": {"up": 2, "down": -2}, "y": {"up": 2.2, "down": -2.2}},
    {"x": {"up": 3, "down": -3}, "y": {"up": 3.3, "down": -3.3}},
    {"x": {"up": 4, "down": -4}, "y": {"up": 4.4, "down": -4.4}},
])
array
```

```{code-cell} ipython3
ak.flatten(array, axis="records")
```

There is considerable overlap between `ak.ravel`, and `ak.flatten(axis="records")`. In most cases, `ak.flatten` is the most appropriate function, because the fields of an array are often conceptually distinct and should therefore not be intermixed (e.g. the components of a position vector). Only when you need a NumPy-convertible, structureless array is `ak.ravel` the right choice.

+++

ak.flatten for one axis
-----------------------

The default axis of [ak.flatten](https://awkward-array.readthedocs.io/en/latest/_auto/ak.flatten.html) is `axis=1`. This flattens only the first nested dimension.

```{code-cell} ipython3
ak.flatten(ak.Array([[0, 1, 2], [], [3, 4], [5], [6, 7, 8, 9]]))
```

It also removes missing values _in the axis that is being flattened_ because flattening considers a missing list like an empty list.

```{code-cell} ipython3
ak.flatten(ak.Array([[0, 1, 2], None, [3, 4], [5], [6, 7, 8, 9]]))
```

It does not flatten or remove missing values from any other axis.

```{code-cell} ipython3
ak.flatten(ak.Array([[[0, 1, 2, 3, 4]], [], [[5], [6, 7, 8, 9]]]))
```

```{code-cell} ipython3
ak.flatten(ak.Array([[[0, 1, 2, None]], [], [[5], [6, 7, 8, 9]]]))
```

Moreover, you can't flatten already-flat data because a 1-dimensional array does not have an `axis=1`. (`axis` starts counting at `0`.)

```{code-cell} ipython3
:tags: [raises-exception]

ak.flatten(ak.Array([1, 2, 3, 4, 5]))
```

`axis=0` is a valid option for [ak.flatten](https://awkward-array.readthedocs.io/en/latest/_auto/ak.flatten.html), but since there can't be any lists at this level, it only removes missing values.

```{code-cell} ipython3
ak.flatten(ak.Array([1, 2, 3, None, None, 4, 5]), axis=0)
```

Selecting one element from each list
------------------------------------

Flattening removes list structure without removing values. Often, you want to do the opposite of that: you want to plot one element from each list. This makes the plot "aware" of your list structure.

This kind of operation is usually just a slice.

```{code-cell} ipython3
array = ak.Array([[0, 1, 2], [3, 4], [5], [6, 7, 8, 9]])
array
```

```{code-cell} ipython3
array[:, 0]
```

The above syntax selects all lists from the array (`axis=0`) and the first element from each list (`axis=1`). We could have as easily selected the last:

```{code-cell} ipython3
array[:, -1]
```

A plot made from `ak.flatten(array)` would be a plot of all numbers with no knowledge of lists; a plot made from `array[:, 0]` would be a plot of lists, as represented by the first element in each. It depends on what you want to plot.

+++

What if you get this error?

```{code-cell} ipython3
array = ak.Array([[0, 1, 2], [], [3, 4], [5], [6, 7, 8, 9]])
array
```

```{code-cell} ipython3
:tags: [raises-exception]

array[:, 0]
```

It says that it can't get element `0` of one of the lists, and that's because this `array` contains an empty list.

One way to deal with that is to take a range-slice, rather than ask for an individual element from each list.

```{code-cell} ipython3
array[:, :1]
```

But this array still has structure, so you can flatten it _as an additional step_.

```{code-cell} ipython3
ak.flatten(array[:, :1])
```

Alternatively, you may want to attack the problem head-on: the issue is that some lists have too few elements, so why not remove those lists with an explicit slice? The [ak.num](https://awkward-array.readthedocs.io/en/latest/_auto/ak.num.html) function tells us the length of each nested list.

```{code-cell} ipython3
ak.num(array)
```

```{code-cell} ipython3
ak.num(array) > 0
```

Slicing the first dimension with this would ensure that the second dimension always has the element we seek.

```{code-cell} ipython3
array[ak.num(array) > 0, 0]
```

The same applies if we're taking the last element:

```{code-cell} ipython3
array[ak.num(array) > 0, -1]
```

You can also do fancy things, requesting both the first and last element of each list, as long as it doesn't run afoul of slicing rules (which were constrained to match NumPy's in cases that overlap).

```{code-cell} ipython3
:tags: [raises-exception]

array[ak.num(array) > 0, [0, -1]]   # these two arrays have different lengths, can't be broadcasted as in NumPy advanced slicing
```

```{code-cell} ipython3
array[ak.num(array) > 0][:, [0, -1]]   # so just put them in different slices
```

And then flatten the result (if necessary—the shape is regular; some plotting libraries would interpret it as a single set of numbers).

```{code-cell} ipython3
ak.flatten(array[ak.num(array) > 0][:, [0, -1]])
```

Aggregating each list
---------------------

Reductions should be familiar to users of SQL and Pandas; after grouping data by some quantity, one must apply some aggregating operation on each group to get one number for each group. The one-element slices of the previous section are like SQL's `FIRST_VALUE` and `LAST_VALUE`, which is a special case of reducing.

The architypical aggregation function is "sum," which reduces a list by adding up its values. [ak.sum](https://awkward-array.readthedocs.io/en/latest/_auto/ak.sum.html) and its relatives, [ak.prod](https://awkward-array.readthedocs.io/en/latest/_auto/ak.prod.html) (product/multiplication), [ak.mean](https://awkward-array.readthedocs.io/en/latest/_auto/ak.mean.html), etc., are all reducers in Awkward Array.

Following NumPy, their default `axis` is `None`, but for this application, you'll need to specify an explicit axis.

```{code-cell} ipython3
array = ak.Array([[0, 1, 2], [], [3, 4], [5], [6, 7, 8, 9]])
array
```

```{code-cell} ipython3
ak.sum(array, axis=1)
```

Some of these are not defined for empty lists, so you'll need to either replace the missing values with [ak.fill_none](https://awkward-array.readthedocs.io/en/latest/_auto/ak.fill_none.html) or flatten them.

```{code-cell} ipython3
ak.mean(array, axis=1)
```

```{code-cell} ipython3
ak.fill_none(ak.mean(array, axis=1), 0)   # fill with zero
```

```{code-cell} ipython3
ak.fill_none(ak.mean(array, axis=1), ak.mean(array))   # fill with the mean of all
```

```{code-cell} ipython3
ak.flatten(ak.mean(array, axis=1), axis=0)
```

Each of these has a different effect: filling with `0` puts an identifiable value in the plot (a peak at `0` if it's a histogram), filling with the overall mean imputes a value in missing cases, flattening away the missing values reduces the number of entries in the plot. Each of these has a different meaning when interpreting your plot!

+++

Minimizing/maximizing over each list
------------------------------------

Minimizing and maximizing are also reducers, [ak.min](https://awkward-array.readthedocs.io/en/latest/_auto/ak.min.html) and [ak.max](https://awkward-array.readthedocs.io/en/latest/_auto/ak.max.html) (and [ak.ptp](https://awkward-array.readthedocs.io/en/latest/_auto/ak.ptp.html) for the peak-to-peak difference between the minimum and maximum).

They deserve their own section because they are an important case.

```{code-cell} ipython3
array = ak.Array([[0, 2, 1], [], [4, 3], [5], [8, 6, 7, 9]])
array
```

```{code-cell} ipython3
ak.min(array, axis=1)
```

```{code-cell} ipython3
ak.max(array, axis=1)
```

As before, they aren't defined for empty lists, so you'll have to _choose_ a method to eliminate the missing values.

+++

Sometimes, you want the "top N" elements from each list, rather than the "top 1." Awkward Array doesn't ([yet](https://github.com/scikit-hep/awkward-1.0/issues/554)) have a function for the "top N" elements, but it can be done with [ak.sort](https://awkward-array.readthedocs.io/en/latest/_auto/ak.sort.html) and a slice.

```{code-cell} ipython3
ak.sort(array, axis=1)
```

```{code-cell} ipython3
ak.sort(array, axis=1)[:, -2:]
```

We still have work to do: some of these lists are shorter than the 2 elements we asked for. What should be done with them? Eliminate all lists with fewer than two elements?

```{code-cell} ipython3
ak.sort(array[ak.num(array) >= 2], axis=1)[:, -2:]
```

Or just concatenate everything so that we don't lose the lists with only one value (`5` in this example)?

```{code-cell} ipython3
ak.flatten(ak.sort(array, axis=1)[:, -2:])
```

Minimizing/maximizing lists of records
--------------------------------------

Unlike numbers, records do not have an ordering: you cannot call [ak.min](https://awkward-array.readthedocs.io/en/latest/_auto/ak.min.html) on an array of records. But usually, what you want to do instead is to find the minimum or maximum of some quantity calculated from the records and pick records (or record fields) from that.

```{code-cell} ipython3
array = ak.Array([
    [{"x": 2, "y": 2, "z": 2.2}, {"x": 1, "y": 1, "z": 1.1}, {"x": 3, "y": 3, "z": 3.3}],
    [],
    [{"x": 5, "y": 5, "z": 5.5}, {"x": 4, "y": 4, "z": 4.4}],
    [{"x": 7, "y": 7, "z": 7.7}, {"x": 9, "y": 9, "z": 9.9}, {"x": 8, "y": 8, "z": 8.8}, {"x": 6, "y": 6, "z": 6.6}],
])
array
```

The [ak.argmin](https://awkward-array.readthedocs.io/en/latest/_auto/ak.argmin.html) and [ak.argmax](https://awkward-array.readthedocs.io/en/latest/_auto/ak.argmax.html) functions return the integer index where the minimum or maximum of some numeric formula can be found.

```{code-cell} ipython3
np.sqrt(array.x**2 + array.y**2)
```

```{code-cell} ipython3
ak.argmax(np.sqrt(array.x**2 + array.y**2), axis=1)
```

These integer indexes can be used as slices if they don't eliminate a dimension, which can be requested via `keepdims=True`. This makes a length-1 list for each reduced output.

```{code-cell} ipython3
maximize_by = ak.argmax(np.sqrt(array.x**2 + array.y**2), axis=1, keepdims=True)
maximize_by
```

Applying this to the original `array`, we get the "best" record in each list, according to `maximize_by`.

```{code-cell} ipython3
array[maximize_by]
```

```{code-cell} ipython3
array[maximize_by].tolist()
```

This still has list structures and missing values, so it's ready for [ak.flatten](https://awkward-array.readthedocs.io/en/latest/_auto/ak.flatten.html), assuming that we extract the appropriate record field to plot.

```{code-cell} ipython3
ak.flatten(array[maximize_by].z, axis="records")
```

Concatenating independently restructured arrays
-----------------------------------------------

Sometimes, what you want to do can't be a single expression. Suppose we have this data:

```{code-cell} ipython3
array = ak.Array([[{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}], [], [{"x": 3.3, "y": [1, 2, 3]}]])
array
```

and we want to combine all _x_ values and the maximum _y_ value in a plot. This requires a different expression on `array.x` from `array.y`.

```{code-cell} ipython3
ak.flatten(array.x)
```

```{code-cell} ipython3
ak.flatten(ak.max(array.y, axis=2), axis="records")
```

To get all of these into one array (because the plotting function only accepts one argument), you'll need to [ak.concatenate](https://awkward-array.readthedocs.io/en/latest/_auto/ak.concatenate.html) them.

```{code-cell} ipython3
ak.concatenate([
    ak.flatten(array.x),
    ak.flatten(ak.max(array.y, axis=2), axis="records"),
])
```

Maintaining alignment between arrays with missing values
--------------------------------------------------------

Dropping missing values with [ak.flatten](https://awkward-array.readthedocs.io/en/latest/_auto/ak.flatten.html) doesn't keep track of where they were removed. This is a problem if the plotting library takes separate sequences for the x-axis and y-axis, and these must be aligned.

Instead of [ak.flatten](https://awkward-array.readthedocs.io/en/latest/_auto/ak.flatten.html), you can use [ak.is_none](https://awkward-array.readthedocs.io/en/latest/_auto/ak.is_none.html).

```{code-cell} ipython3
array = ak.Array([
    {"x": 1, "y": 5.5},
    {"x": 2, "y": 3.3},
    {"x": None, "y": 2.2},
    {"x": 4, "y": None},
    {"x": 5, "y": 1.1},
])
array
```

```{code-cell} ipython3
ak.is_none(array.x)
```

```{code-cell} ipython3
ak.is_none(array.y)
```

```{code-cell} ipython3
to_keep = ~(ak.is_none(array.x) | ak.is_none(array.y))
to_keep
```

```{code-cell} ipython3
array.x[to_keep], array.y[to_keep]
```

Actually drawing structure
--------------------------

If need be, you can change the plotter to match the data.

```{code-cell} ipython3
array = ak.Array([
    [{"x": 1, "y": 3.3}, {"x": 2, "y": 1.1}, {"x": 3, "y": 2.2}],
    [],
    [{"x": 4, "y": 5.5}, {"x": 5, "y": 4.4}],
    [{"x": 5, "y": 1.1}, {"x": 4, "y": 3.3}, {"x": 2, "y": 5.5}, {"x": 1, "y": 4.4}],
])
array
```

```{code-cell} ipython3
import matplotlib.pyplot as plt
import matplotlib.path
import matplotlib.patches

fig, ax = plt.subplots()

for line in array:
    if len(line) > 0:
        vertices = np.dstack([np.asarray(line.x), np.asarray(line.y)])[0]
        codes = [matplotlib.path.Path.MOVETO] + [matplotlib.path.Path.LINETO] * (len(line) - 1)
        path = matplotlib.path.Path(vertices, codes)
        ax.add_patch(matplotlib.patches.PathPatch(path, facecolor="none"))

ax.set_xlim(0, 6)
ax.set_ylim(0, 6);
```

(The above example assumes that `len(array)` is small enough to iterate over in Python, but vectorizes over each list in the `array`. It was adapted from the [Matplotlib path tutorial](https://matplotlib.org/stable/tutorials/advanced/path_tutorial.html).)
