# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_bool_sort():
    array = ak.layout.NumpyArray(np.array([True, False, True, False, False]))
    assert ak.to_list(ak.sort(array, axis=0, ascending=True, stable=False)) == [
        False,
        False,
        False,
        True,
        True,
    ]


def test_keep_None_in_place_test():
    array = ak.Array([[3, 2, 1], [], None, [4, 5]])

    assert ak.to_list(ak.argsort(array, axis=1)) == [
        [2, 1, 0],
        [],
        None,
        [0, 1],
    ]

    assert ak.to_list(ak.sort(array, axis=1)) == [
        [1, 2, 3],
        [],
        None,
        [4, 5],
    ]

    assert ak.to_list(array[ak.argsort(array, axis=1)]) == ak.to_list(
        ak.sort(array, axis=1)
    )


def test_EmptyArray():
    array = ak.layout.EmptyArray()
    assert ak.to_list(ak.sort(array)) == []
    assert ak.to_list(ak.argsort(array)) == []
    assert str(ak.type(ak.sort(array))) == "0 * float64"
    assert str(ak.type(ak.argsort(array))) == "0 * int64"

    array2 = ak.Array([[], [], []])
    assert ak.to_list(ak.argsort(array2)) == [[], [], []]
    assert str(ak.type(ak.argsort(array2))) == "3 * var * int64"


def test_NumpyArray():
    array = ak.layout.NumpyArray(np.array([3.3, 2.2, 1.1, 5.5, 4.4]))
    assert ak.to_list(ak.argsort(array, axis=0, ascending=True, stable=False)) == [
        2,
        1,
        0,
        4,
        3,
    ]
    assert ak.to_list(ak.argsort(array, axis=0, ascending=False, stable=False)) == [
        3,
        4,
        0,
        1,
        2,
    ]

    assert ak.to_list(ak.sort(array, axis=0, ascending=True, stable=False)) == [
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
    ]
    assert ak.to_list(ak.sort(array, axis=0, ascending=False, stable=False)) == [
        5.5,
        4.4,
        3.3,
        2.2,
        1.1,
    ]

    array2 = ak.layout.NumpyArray(np.array([[3.3, 2.2, 4.4], [1.1, 5.5, 3.3]]))

    assert ak.to_list(
        ak.sort(array2, axis=1, ascending=True, stable=False)
    ) == ak.to_list(np.sort(array2, axis=1))
    assert ak.to_list(
        ak.sort(array2, axis=0, ascending=True, stable=False)
    ) == ak.to_list(np.sort(array2, axis=0))

    assert ak.to_list(
        ak.argsort(array2, axis=1, ascending=True, stable=False)
    ) == ak.to_list(np.argsort(array2, 1))
    assert ak.to_list(
        ak.argsort(array2, axis=0, ascending=True, stable=False)
    ) == ak.to_list(np.argsort(array2, 0))

    with pytest.raises(ValueError) as err:
        ak.sort(array2, axis=2, ascending=True, stable=False)
    assert str(err.value).startswith(
        "axis=2 exceeds the depth of the nested list structure (which is 2)"
    )


def test_IndexedOptionArray():
    array = ak.Array(
        [
            [None, None, 2.2, 1.1, 3.3],
            [None, None, None],
            [4.4, None, 5.5],
            [5.5, None, None],
            [-4.4, -5.5, -6.6],
        ]
    )

    assert ak.to_list(ak.sort(array, axis=0, ascending=True, stable=False)) == [
        [-4.4, -5.5, -6.6, 1.1, 3.3],
        [4.4, None, 2.2],
        [5.5, None, 5.5],
        [None, None, None],
        [None, None, None],
    ]

    assert ak.to_list(ak.sort(array, axis=1, ascending=True, stable=False)) == [
        [1.1, 2.2, 3.3, None, None],
        [None, None, None],
        [4.4, 5.5, None],
        [5.5, None, None],
        [-6.6, -5.5, -4.4],
    ]

    assert ak.to_list(ak.sort(array, axis=1, ascending=False, stable=True)) == [
        [3.3, 2.2, 1.1, None, None],
        [None, None, None],
        [5.5, 4.4, None],
        [5.5, None, None],
        [-4.4, -5.5, -6.6],
    ]

    assert ak.to_list(ak.sort(array, axis=1, ascending=False, stable=False)) == [
        [3.3, 2.2, 1.1, None, None],
        [None, None, None],
        [5.5, 4.4, None],
        [5.5, None, None],
        [-4.4, -5.5, -6.6],
    ]

    assert ak.to_list(ak.argsort(array, axis=0, ascending=True, stable=True)) == [
        [4, 4, 4, 0, 0],
        [2, 0, 0],
        [3, 1, 2],
        [0, 2, 1],
        [1, 3, 3],
    ]

    assert ak.to_list(ak.argsort(array, axis=0, ascending=True, stable=False)) == [
        [4, 4, 4, 0, 0],
        [2, 0, 0],
        [3, 1, 2],
        [0, 2, 1],
        [1, 3, 3],
    ]

    assert ak.to_list(ak.argsort(array, axis=0, ascending=False, stable=True)) == [
        [3, 4, 2, 0, 0],
        [2, 0, 0],
        [4, 1, 4],
        [0, 2, 1],
        [1, 3, 3],
    ]
    assert ak.to_list(ak.argsort(array, axis=0, ascending=False, stable=False)) == [
        [3, 4, 2, 0, 0],
        [2, 0, 0],
        [4, 1, 4],
        [0, 2, 1],
        [1, 3, 3],
    ]

    assert ak.to_list(ak.argsort(array, axis=1, ascending=True, stable=True)) == [
        [3, 2, 4, 0, 1],
        [0, 1, 2],
        [0, 2, 1],
        [0, 1, 2],
        [2, 1, 0],
    ]

    assert ak.to_list(ak.argsort(array, axis=1, ascending=True, stable=False)) == [
        [3, 2, 4, 0, 1],
        [0, 1, 2],
        [0, 2, 1],
        [0, 1, 2],
        [2, 1, 0],
    ]

    assert ak.to_list(ak.argsort(array, axis=1, ascending=False, stable=True)) == [
        [4, 2, 3, 0, 1],
        [0, 1, 2],
        [2, 0, 1],
        [0, 1, 2],
        [0, 1, 2],
    ]

    array2 = ak.Array([None, None, 1, -1, 30])
    assert ak.to_list(ak.argsort(array2, axis=0, ascending=True, stable=True)) == [
        3,
        2,
        4,
        0,
        1,
    ]

    array3 = ak.Array(
        [[2.2, 1.1, 3.3], [], [4.4, 5.5], [5.5], [-4.4, -5.5, -6.6]]
    ).layout

    assert ak.to_list(ak.sort(array3, axis=1, ascending=False, stable=False)) == [
        [3.3, 2.2, 1.1],
        [],
        [5.5, 4.4],
        [5.5],
        [-4.4, -5.5, -6.6],
    ]

    assert ak.to_list(ak.sort(array3, axis=0, ascending=True, stable=False)) == [
        [-4.4, -5.5, -6.6],
        [],
        [2.2, 1.1],
        [4.4],
        [5.5, 5.5, 3.3],
    ]

    # FIXME: Based on NumPy list sorting:
    #
    # array([list([2.2, 1.1, 3.3]), list([]), list([4.4, 5.5]), list([5.5]),
    #        list([-4.4, -5.5, -6.6])], dtype=object)
    # np.sort(array, axis=0)
    # array([list([]), list([-4.4, -5.5, -6.6]), list([2.2, 1.1, 3.3]),
    #        list([4.4, 5.5]), list([5.5])], dtype=object)
    #
    # the result should be:
    #
    # [[ -4.4, -5.5, -6.6 ],
    #  [  2.2,  1.1,  3.3 ],
    #  [  4.4,  5.5 ],
    #  [  5.5 ],
    #  []]

    # This can be done following the steps: pad, sort,
    # and dropna to strip off the None's
    #
    array4 = array3.rpad(3, 1)
    assert ak.to_list(array4) == [
        [2.2, 1.1, 3.3],
        [None, None, None],
        [4.4, 5.5, None],
        [5.5, None, None],
        [-4.4, -5.5, -6.6],
    ]

    array5 = ak.sort(array4, axis=0, ascending=True, stable=False)
    assert ak.to_list(array5) == [
        [-4.4, -5.5, -6.6],
        [2.2, 1.1, 3.3],
        [4.4, 5.5, None],
        [5.5, None, None],
        [None, None, None],
    ]

    array4 = array3.rpad(5, 1)
    assert ak.to_list(array4) == [
        [2.2, 1.1, 3.3, None, None],
        [None, None, None, None, None],
        [4.4, 5.5, None, None, None],
        [5.5, None, None, None, None],
        [-4.4, -5.5, -6.6, None, None],
    ]

    array5 = ak.sort(array4, axis=0, ascending=True, stable=False)
    assert ak.to_list(array5) == [
        [-4.4, -5.5, -6.6, None, None],
        [2.2, 1.1, 3.3, None, None],
        [4.4, 5.5, None, None, None],
        [5.5, None, None, None, None],
        [None, None, None, None, None],
    ]

    array5 = ak.argsort(array4, axis=0, ascending=True, stable=False)
    assert ak.to_list(array5) == [
        [4, 4, 4, 0, 0],
        [0, 0, 0, 1, 1],
        [2, 2, 1, 2, 2],
        [3, 1, 2, 3, 3],
        [1, 3, 3, 4, 4],
    ]

    # FIXME: implement dropna to strip off the None's
    #
    # array6 = array5.dropna(0)
    # assert ak.to_list(array6) == [
    #     [ -4.4, -5.5, -6.6 ],
    #     [  2.2,  1.1,  3.3 ],
    #     [  4.4,  5.5 ],
    #     [  5.5 ],
    #     []]

    content = ak.layout.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5]))
    index1 = ak.layout.Index32(np.array([1, 2, 3, 4], dtype=np.int32))
    indexedarray1 = ak.layout.IndexedArray32(index1, content)
    assert ak.to_list(
        ak.argsort(indexedarray1, axis=0, ascending=True, stable=False)
    ) == [0, 1, 2, 3]

    index2 = ak.layout.Index64(np.array([1, 2, 3], dtype=np.int64))
    indexedarray2 = ak.layout.IndexedArray64(index2, indexedarray1)
    assert ak.to_list(
        ak.sort(indexedarray2, axis=0, ascending=False, stable=False)
    ) == [5.5, 4.4, 3.3]

    index3 = ak.layout.Index32(np.array([1, 2], dtype=np.int32))
    indexedarray3 = ak.layout.IndexedArray32(index3, indexedarray2)
    assert ak.to_list(ak.sort(indexedarray3, axis=0, ascending=True, stable=False)) == [
        4.4,
        5.5,
    ]


def test_3d():
    array = ak.layout.NumpyArray(
        np.array(
            [
                # axis 2:    0       1       2       3       4         # axis 1:
                [
                    [1.1, 2.2, 3.3, 4.4, 5.5],  # 0
                    [6.6, 7.7, 8.8, 9.9, 10.10],  # 1
                    [11.11, 12.12, 13.13, 14.14, 15.15],
                ],  # 2
                [
                    [-1.1, -2.2, -3.3, -4.4, -5.5],  # 3
                    [-6.6, -7.7, -8.8, -9.9, -10.1],  # 4
                    [-11.11, -12.12, -13.13, -14.14, -15.15],
                ],
            ]
        )
    )  # 5
    assert ak.to_list(
        ak.argsort(array, axis=2, ascending=True, stable=False)
    ) == ak.to_list(np.argsort(array, 2))
    assert ak.to_list(
        ak.sort(array, axis=2, ascending=True, stable=False)
    ) == ak.to_list(np.sort(array, 2))
    assert ak.to_list(
        ak.argsort(array, axis=1, ascending=True, stable=False)
    ) == ak.to_list(np.argsort(array, 1))
    assert ak.to_list(
        ak.sort(array, axis=1, ascending=True, stable=False)
    ) == ak.to_list(np.sort(array, 1))

    assert ak.to_list(ak.sort(array, axis=1, ascending=False, stable=False)) == [
        [
            [11.11, 12.12, 13.13, 14.14, 15.15],
            [6.6, 7.7, 8.8, 9.9, 10.1],
            [1.1, 2.2, 3.3, 4.4, 5.5],
        ],
        [
            [-1.1, -2.2, -3.3, -4.4, -5.5],
            [-6.6, -7.7, -8.8, -9.9, -10.1],
            [-11.11, -12.12, -13.13, -14.14, -15.15],
        ],
    ]

    assert ak.to_list(
        ak.sort(array, axis=0, ascending=True, stable=False)
    ) == ak.to_list(np.sort(array, 0))
    assert ak.to_list(
        ak.argsort(array, axis=0, ascending=True, stable=False)
    ) == ak.to_list(np.argsort(array, 0))


def test_RecordArray():
    array = ak.Array(
        [
            {"x": 0.0, "y": []},
            {"x": 1.1, "y": [1]},
            {"x": 2.2, "y": [2, 2]},
            {"x": 3.3, "y": [3, 3, 3]},
            {"x": 4.4, "y": [4, 4, 4, 4]},
            {"x": 5.5, "y": [5, 5, 5]},
            {"x": 6.6, "y": [6, 6]},
            {"x": 7.7, "y": [7]},
            {"x": 8.8, "y": []},
        ]
    )
    assert ak.to_list(array) == [
        {"x": 0.0, "y": []},
        {"x": 1.1, "y": [1]},
        {"x": 2.2, "y": [2, 2]},
        {"x": 3.3, "y": [3, 3, 3]},
        {"x": 4.4, "y": [4, 4, 4, 4]},
        {"x": 5.5, "y": [5, 5, 5]},
        {"x": 6.6, "y": [6, 6]},
        {"x": 7.7, "y": [7]},
        {"x": 8.8, "y": []},
    ]

    assert ak.to_list(array.layout.sort(-1, True, False)) == {
        "x": [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8],
        "y": [[], [1], [2, 2], [3, 3, 3], [4, 4, 4, 4], [5, 5, 5], [6, 6], [7], []],
    }

    assert ak.to_list(array.layout.sort(-1, False, False)) == {
        "x": [8.8, 7.7, 6.6, 5.5, 4.4, 3.3, 2.2, 1.1, 0.0],
        "y": [[], [1], [2, 2], [3, 3, 3], [4, 4, 4, 4], [5, 5, 5], [6, 6], [7], []],
    }

    assert ak.to_list(array.layout.argsort(-1, True, False)) == {
        "x": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        "y": [[], [0], [0, 1], [0, 1, 2], [0, 1, 2, 3], [0, 1, 2], [0, 1], [0], []],
    }

    assert ak.to_list(array.x.layout.argsort(0, True, False)) == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
    ]
    assert ak.to_list(array.x.layout.argsort(0, False, False)) == [
        8,
        7,
        6,
        5,
        4,
        3,
        2,
        1,
        0,
    ]

    array_y = array.y
    assert ak.to_list(array_y) == [
        [],
        [1],
        [2, 2],
        [3, 3, 3],
        [4, 4, 4, 4],
        [5, 5, 5],
        [6, 6],
        [7],
        [],
    ]
    assert ak.to_list(array.y.layout.argsort(0, True, False)) == [
        # FIXME?
        [],
        [1],
        [2, 2],
        [3, 3, 3],
        [4, 4, 4, 4],
        [5, 5, 5],
        [6, 6],
        [7],
        []
        # [],
        # [0],
        # [1, 0],
        # [2, 1, 0],
        # [3, 2, 1, 0],
        # [4, 3, 2],
        # [5, 4],
        # [6],
        # [],
    ]

    assert ak.to_list(array.y.layout.argsort(1, True, True)) == [
        [],
        [0],
        [0, 1],
        [0, 1, 2],
        [0, 1, 2, 3],
        [0, 1, 2],
        [0, 1],
        [0],
        [],
    ]


def test_ByteMaskedArray():
    content = ak.from_iter(
        [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )
    mask = ak.layout.Index8(np.array([0, 0, 1, 1, 0], dtype=np.int8))
    array = ak.layout.ByteMaskedArray(mask, content, valid_when=False)
    assert ak.to_list(ak.argsort(array, axis=0, ascending=True, stable=False)) == [
        [0, 0, 0],
        [],
        [2, 2, 2, 2],
        None,
        None,
    ]

    assert ak.to_list(ak.sort(array, axis=0, ascending=True, stable=False)) == [
        [0.0, 1.1, 2.2],
        [],
        [6.6, 7.7, 8.8, 9.9],
        None,
        None,
    ]

    assert ak.to_list(ak.sort(array, axis=0, ascending=False, stable=False)) == [
        [6.6, 7.7, 8.8],
        [],
        [0.0, 1.1, 2.2, 9.9],
        None,
        None,
    ]

    assert ak.to_list(ak.argsort(array, axis=1, ascending=True, stable=False)) == [
        [0, 1, 2],
        [],
        None,
        None,
        [0, 1, 2, 3],
    ]

    assert ak.to_list(array.sort(1, False, False)) == [
        [2.2, 1.1, 0.0],
        [],
        None,
        None,
        [9.9, 8.8, 7.7, 6.6],
    ]


def test_UnionArray():
    content0 = ak.from_iter([[1.1, 2.2, 3.3], [], [4.4, 5.5]], highlevel=False)
    content1 = ak.from_iter(
        [["one"], ["two"], ["three"], ["four"], ["five"]], highlevel=False
    )
    tags = ak.layout.Index8(np.array([1, 1, 0, 0, 1, 0, 1, 1], dtype=np.int8))
    index = ak.layout.Index32(np.array([0, 1, 0, 1, 2, 2, 4, 3], dtype=np.int32))
    array = ak.layout.UnionArray8_32(tags, index, [content0, content1])

    with pytest.raises(ValueError) as err:
        ak.sort(array, axis=1, ascending=True, stable=False)
    assert str(err.value).startswith("cannot sort UnionArray8_32")


def test_sort_strings():
    content1 = ak.from_iter(["one", "two", "three", "four", "five"], highlevel=False)
    assert ak.to_list(content1) == ["one", "two", "three", "four", "five"]

    assert ak.to_list(ak.sort(content1, axis=0, ascending=True, stable=False)) == [
        "five",
        "four",
        "one",
        "three",
        "two",
    ]
    assert ak.to_list(ak.sort(content1, axis=0, ascending=False, stable=False)) == [
        "two",
        "three",
        "one",
        "four",
        "five",
    ]


def test_sort_bytestrings():
    array = ak.from_iter(
        [b"one", b"two", b"three", b"two", b"two", b"one", b"three"], highlevel=False
    )
    assert ak.to_list(array) == [
        b"one",
        b"two",
        b"three",
        b"two",
        b"two",
        b"one",
        b"three",
    ]

    assert ak.to_list(ak.sort(array, axis=0, ascending=True, stable=False)) == [
        b"one",
        b"one",
        b"three",
        b"three",
        b"two",
        b"two",
        b"two",
    ]

    assert ak.to_list(ak.argsort(array, axis=0, ascending=True, stable=True)) == [
        0,
        5,
        2,
        6,
        1,
        3,
        4,
    ]


def test_sort_zero_length_arrays():
    array = ak.layout.IndexedArray64(
        ak.layout.Index64([]), ak.layout.NumpyArray([1, 2, 3])
    )
    assert ak.to_list(array) == []
    assert ak.to_list(ak.sort(array)) == []
    assert ak.to_list(ak.argsort(array)) == []

    content0 = ak.from_iter([[1.1, 2.2, 3.3], [], [4.4, 5.5]], highlevel=False)
    content1 = ak.from_iter(["one", "two", "three", "four", "five"], highlevel=False)
    tags = ak.layout.Index8([])
    index = ak.layout.Index32([])
    array = ak.layout.UnionArray8_32(tags, index, [content0, content1])
    assert ak.to_list(array) == []
    assert ak.to_list(ak.sort(array)) == []
    assert ak.to_list(ak.argsort(array)) == []

    content = ak.from_iter(
        [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )
    mask = ak.layout.Index8([])
    array = ak.layout.ByteMaskedArray(mask, content, valid_when=False)
    assert ak.to_list(array) == []
    assert ak.to_list(ak.sort(array)) == []
    assert ak.to_list(ak.argsort(array)) == []

    array = ak.layout.NumpyArray([])
    assert ak.to_list(array) == []
    assert ak.to_list(ak.sort(array)) == []
    assert ak.to_list(ak.argsort(array)) == []

    array = ak.layout.RecordArray([])
    assert ak.to_list(array) == []
    assert ak.to_list(ak.sort(array)) == []
    assert ak.to_list(ak.argsort(array)) == []

    content = ak.layout.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    starts1 = ak.layout.Index64([])
    stops1 = ak.layout.Index64([])
    offsets1 = ak.layout.Index64(np.array([0]))
    array = ak.layout.ListArray64(starts1, stops1, content)
    assert ak.to_list(array) == []
    assert ak.to_list(ak.sort(array)) == []
    assert ak.to_list(ak.argsort(array)) == []

    array = ak.layout.ListOffsetArray64(offsets1, content)
    assert ak.to_list(array) == []
    assert ak.to_list(ak.sort(array)) == []
    assert ak.to_list(ak.argsort(array)) == []
