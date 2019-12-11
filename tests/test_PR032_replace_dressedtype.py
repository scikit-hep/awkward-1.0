# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys
import itertools
import pickle

import pytest
import numpy

import awkward1

def test_types_with_parameters():
    t = awkward1.layout.UnknownType()
    assert t.parameters == {}
    t.parameters = {"key": ["val", "ue"]}
    assert t.parameters == {"key": ["val", "ue"]}
    t = awkward1.layout.UnknownType(parameters={"key": ["val", "ue"]})
    assert t.parameters == {"key": ["val", "ue"]}

    t = awkward1.layout.PrimitiveType("int32", parameters={"key": ["val", "ue"]}); assert t.parameters == {"key": ["val", "ue"]}
    t = awkward1.layout.PrimitiveType("float64", parameters={"key": ["val", "ue"]}); assert t.parameters == {"key": ["val", "ue"]}
    t = awkward1.layout.ArrayType(awkward1.layout.PrimitiveType("int32"), 100, parameters={"key": ["val", "ue"]}); assert t.parameters == {"key": ["val", "ue"]}
    t = awkward1.layout.ListType(awkward1.layout.PrimitiveType("int32"), parameters={"key": ["val", "ue"]}); assert t.parameters == {"key": ["val", "ue"]}
    t = awkward1.layout.RegularType(awkward1.layout.PrimitiveType("int32"), 5, parameters={"key": ["val", "ue"]}); assert t.parameters == {"key": ["val", "ue"]}
    t = awkward1.layout.OptionType(awkward1.layout.PrimitiveType("int32"), parameters={"key": ["val", "ue"]}); assert t.parameters == {"key": ["val", "ue"]}
    t = awkward1.layout.UnionType((awkward1.layout.PrimitiveType("int32"), awkward1.layout.PrimitiveType("float64")), parameters={"key": ["val", "ue"]}); assert t.parameters == {"key": ["val", "ue"]}
    t = awkward1.layout.RecordType({"one": awkward1.layout.PrimitiveType("int32"), "two": awkward1.layout.PrimitiveType("float64")}, parameters={"key": ["val", "ue"]}); assert t.parameters == {"key": ["val", "ue"]}

    t = awkward1.layout.UnknownType(parameters={"key1": ["val", "ue"], "key2": u"one \u2192 two"})
    assert t.parameters == {"key2": u"one \u2192 two", "key1": ["val", "ue"]}
