#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import argparse
import subprocess
import shutil
import os
import sys
import json
import glob

arguments = argparse.ArgumentParser()
arguments.add_argument("--clean", default=False, action="store_true")
arguments.add_argument("--release", action="store_true")
arguments.add_argument("--no-ctest", action="store_true")
arguments.add_argument("--no-buildpython", action="store_true")
arguments.add_argument("--pytest", default=None)
args = arguments.parse_args()

args.ctest = not args.no_ctest
args.buildpython = not args.no_buildpython

try:
    git_config = open(".git/config").read()
except:
    git_config = ""

if "url = https://github.com/scikit-hep/awkward-1.0.git" not in git_config:
    arguments.error("localbuild must be executed in the head of the awkward-1.0 tree")

if args.clean:
    for x in ("localbuild", "awkward1", ".pytest_cache", "tests/__pycache__"):
        if os.path.exists(x):
            shutil.rmtree(x)
    sys.exit()

thisstate = {"release": args.release,
             "ctest": args.ctest,
             "buildpython": args.buildpython,
             "python_executable": sys.executable}

try:
    localbuild_time = os.stat("localbuild").st_mtime
except:
    localbuild_time = 0
try:
    laststate = json.load(open("localbuild/lastargs.json"))
except:
    laststate = None

# Refresh the directory if any configuration has changed.
if (# os.stat("CMakeLists.txt").st_mtime >= localbuild_time or
    # os.stat("localbuild.py").st_mtime >= localbuild_time or
    # os.stat("setup.py").st_mtime >= localbuild_time or
    thisstate != laststate):

    subprocess.check_call(["pip", "install",
                           "-r", "requirements.txt",
                           "-r", "requirements-test.txt",
                           "-r", "requirements-docs.txt",
                           "-r", "requirements-dev.txt"])

    if os.path.exists("localbuild"):
        shutil.rmtree("localbuild")

    newdir_args = ["cmake", "-S", ".", "-B", "localbuild"]

    if args.release:
        newdir_args.append("-DCMAKE_BUILD_TYPE=Release")
    else:
        newdir_args.append("-DCMAKE_BUILD_TYPE=Debug")

    if args.ctest:
        newdir_args.append("-DBUILD_TESTING=ON")

    if args.buildpython:
        newdir_args.extend(["-DPYTHON_EXECUTABLE=" + thisstate["python_executable"], "-DPYBUILD=ON"])

    subprocess.check_call(newdir_args)
    json.dump(thisstate, open("localbuild/lastargs.json", "w"))

# Build C++ normally; this might be a no-op if make/ninja determines that the build is up-to-date.
subprocess.check_call(["cmake", "--build", "localbuild"])

if args.ctest:
    subprocess.check_call(["cmake", "--build", "localbuild", "--target", "test", "--", "CTEST_OUTPUT_ON_FAILURE=1"])

# Build Python (copy sources to executable tree).
if args.buildpython:
    if os.path.exists("awkward1"):
        shutil.rmtree("awkward1")

    # Maybe someday they can be symlinks.
    for x in glob.glob("src/awkward1/**", recursive=True):
        olddir, oldfile = os.path.split(x)
        newdir  = olddir[3 + len(os.sep):]
        newfile = x[3 + len(os.sep):]
        if not os.path.exists(newdir):
            os.mkdir(newdir)
        if not os.path.isdir(x):
            # os.symlink(x, newfile)
            shutil.copyfile(x, newfile)

    # The extension modules must be copied over.
    for x in glob.glob("localbuild/layout*") + glob.glob("localbuild/types*") + glob.glob("localbuild/_io*") + glob.glob("localbuild/libawkward*"):
        shutil.copyfile(x, os.path.join("awkward1", os.path.split(x)[1]))

    # localbuild must be in the library path for some operations.
    env = dict(os.environ)
    reminder = False
    if "awkward1" not in env.get("LD_LIBRARY_PATH", ""):
        env["LD_LIBRARY_PATH"] = "awkward1:" + env.get("LD_LIBRARY_PATH", "")
        reminder = True

    # Run pytest on all or a subset of tests.
    if args.pytest is not None:
        subprocess.check_call(["python", "-m", "pytest", "-vv", "-rs", args.pytest], env=env)

    # If you'll be using it interactively, you'll need awkward1 in the library path (for some operations).
    if reminder:
        print("")
        print("Remember to")
        print("")
        print("    export LD_LIBRARY_PATH=awkward1:$LD_LIBRARY_PATH")
        print("")
