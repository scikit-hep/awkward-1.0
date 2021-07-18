from hypothesis import strategies as st, given, settings
from hypothesis.strategies import composite
import os
import json
from shutil import copy
import copy
import os
from collections import OrderedDict
from itertools import product
import yaml
from numpy import uint8  # noqa: F401 (used in evaluated strings)

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


# Class to create an object of the various arguments of a kernel function
class Argument(object):
    __slots__ = ("name", "typename", "direction", "role")

    def __init__(self, name, typename, direction, role="default"):
        self.name = name
        self.typename = typename
        self.direction = direction
        self.role = role


# Class to create an object of various specifications of a kernel function
class Specification(object):
    def __init__(self, spec, testdata, blacklisted):
        self.name = spec["name"]
        self.args = []
        for arg in spec["args"]:
            self.args.append(
                Argument(
                    arg["name"],
                    arg["type"],
                    arg["dir"],
                    arg["role"] if "role" in arg.keys() else "default",
                )
            )
        if blacklisted:
            self.tests = []

    def validateoverflow(self, testvals):
        flag = True
        for arg in self.args:
            if "uint" in arg.typename and (
                any(n < 0 for n in testvals["inargs"][arg.name])
                or (
                    "outargs" in testvals.keys()
                    and arg.name in testvals["outargs"].keys()
                    and any(n < 0 for n in testvals["outargs"][arg.name])
                )
            ):
                flag = False
        return flag

    def dicttolist(self, outputdict, typename):
        typeval = gettypeval(typename)
        vallist = []
        count = 0
        for num in sorted(outputdict):
            if num == count:
                vallist.append(outputdict[num])
            else:
                while num != count:
                    count += 1
                    vallist.append(typeval)
                vallist.append(outputdict[num])
            count += 1
        return vallist

    def getdummyvalue(self, typename, length):
        return [gettypeval(typename)] * length

    def typevalidates(self, testdict, arglist):
        for arg in arglist:
            if isinstance(testdict[arg.name], list):
                if testdict[arg.name] == []:
                    return False
                if not isinstance(
                    testdict[arg.name][0], type(gettypeval(arg.typename))
                ):
                    return False
            else:
                if not isinstance(testdict[arg.name], type(gettypeval(arg.typename))):
                    return False
        return True


# Reads the kernel specification file into a dict
# Creates an object of each specification and stores it in the dict
def readspec():
    genpykernels()
    specdict = {}
    with open(
        os.path.join(CURRENT_DIR, "..", "kernel-specification.yml"), "r"
    ) as specfile:
        loadfile = yaml.safe_load(specfile)
        indspec = loadfile["kernels"]
        data = loadfile["tests"]
        for spec in indspec:
            if "def " in spec["definition"]:
                for childfunc in spec["specializations"]:
                    specdict[childfunc["name"]] = Specification(
                        childfunc,
                        data,
                        not spec["automatic-tests"],
                    )
    return specdict


def wrap_exec(string, globs, locs):
    exec(string, globs, locs)


def wrap_eval(string, globs, locs):
    return eval(string, globs, locs)


def gettypename(spectype):
    typename = spectype.replace("List", "").replace("[", "").replace("]", "")
    if typename.endswith("_t"):
        typename = typename[:-2]
    return typename


def getfuncnames():
    funcs = {}
    with open(os.path.join(CURRENT_DIR, "..", "kernel-specification.yml")) as specfile:
        indspec = yaml.safe_load(specfile)["kernels"]
        for spec in indspec:
            funcs[spec["name"]] = []
            for childfunc in spec["specializations"]:
                funcs[spec["name"]].append(childfunc["name"])
    return funcs


# Generates the kernel functions in kernels.py in tests-spec folder
def genpykernels():
    print("Generating Python kernels")
    prefix = """
from numpy import uint8
kMaxInt64  = 9223372036854775806
kSliceNone = kMaxInt64 + 1
"""
    with open(
        os.path.join(CURRENT_DIR, "..", "hypothesis-tests-spec", "kernels.py"), "w"
    ) as outfile:
        outfile.write(prefix)
        with open(
            os.path.join(CURRENT_DIR, "..", "kernel-specification.yml")
        ) as specfile:
            indspec = yaml.safe_load(specfile)["kernels"]
            for spec in indspec:
                if "def " in spec["definition"]:
                    outfile.write(spec["definition"] + "\n")
                    for childfunc in spec["specializations"]:
                        outfile.write(
                            "{0} = {1}\n".format(childfunc["name"], spec["name"])
                        )
                    outfile.write("\n\n")


def gettypeval(typename):
    if "int" in typename:
        typeval = 123
    elif "bool" in typename:
        typeval = True
    elif "double" in typename or "float" in typename:
        typeval = 123.0
    else:
        raise ValueError("Unknown type encountered")
    return typeval


def main():
    print("Generating tests")
    with open(
        os.path.join(
            CURRENT_DIR,
            "..",
            "hypothesis-tests-spec",
            "test_awkward_IndexedArray_validity.py",
        ),
        "w",
    ) as wr:
        wr.write("import kernels")
        with open(os.path.join(CURRENT_DIR, "..", "inputs.json"), "r") as js:
            data = json.load(js)
            i = 0
            indexArray = data["Inputs"][0]["cases"]
            for d in indexArray:
                i = i + 1
                wr.write(
                    "\n\n\ndef test_pyawkward_IndexedArray32_validity_"
                    + str(i)
                    + "():\n\t"
                )
                wr.write("index=" + str(d["inputs"][0]))
                wr.write("\n\tlength=" + str(d["inputs"][1]))
                wr.write("\n\tlencontent=" + str(d["inputs"][2]))
                wr.write("\n\tisoption=" + str(d["inputs"][3]))
                if d["except"] == True:
                    wr.write("\n\ttry:\n\t\t")
                    wr.write(
                        "result=kernels.awkward_IndexedArray_validity("
                        + str(d["inputs"][0])
                        + ","
                        + str(d["inputs"][1])
                        + ","
                        + str(d["inputs"][2])
                        + ","
                        + str(d["inputs"][3])
                        + ")"
                    )
                    wr.write("\n\texcept " + str(d["output"]) + ":")
                    wr.write("\n\t\tassert True")
                else:
                    wr.write(
                        "\n\tresult=kernels.awkward_IndexedArray_validity("
                        + str(d["inputs"][0])
                        + ","
                        + str(d["inputs"][1])
                        + ","
                        + str(d["inputs"][2])
                        + ","
                        + str(d["inputs"][3])
                        + ")"
                    )
                    wr.write("\n\tassert result==None")


if __name__ == "__main__":
    readspec()
    main()
