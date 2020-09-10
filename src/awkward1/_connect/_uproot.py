# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import json

# don't import awkward1._connect._uproot in awkward1/__init__.py!
import uproot4


def can_optimize(interpretation, form):
    if isinstance(interpretation, uproot4.interpretation.objects.AsObjects):
        jsonform = json.loads(form.tojson(verbose=True))
        if (
            jsonform["class"] == "ListOffsetArray64" and
            jsonform["parameters"].get("uproot") == {
                "as": "array", "header": True, "speedbump": False
            } and
            jsonform["content"]["class"] == "ListOffsetArray64" and
            jsonform["content"]["parameters"].get("uproot") == {
                "as": "vector", "header": False
            } and
            jsonform["content"]["content"]["class"] == "NumpyArray" and
            jsonform["content"]["content"]["inner_shape"] == [] and
            (
                jsonform["content"]["content"].get("primitive") == "float64" or
                jsonform["content"]["content"].get("primitive") == "int32"
            )
        ):
            return True

    return False


def basket_array(form, data, byte_offsets, extra):
    return "YOWZA"