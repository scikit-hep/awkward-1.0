// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARDPY_VIRTUAL_H_
#define AWKWARDPY_VIRTUAL_H_

#include <pybind11/pybind11.h>

#include "awkward/virtual/ArrayGenerator.h"
#include "awkward/virtual/ArrayCache.h"

namespace py = pybind11;
namespace ak = awkward;

class PyArrayGenerator: public ak::ArrayGenerator {
public:
  PyArrayGenerator(const ak::FormPtr& form,
                   const py::object& callable,
                   const py::tuple& args,
                   const py::dict& kwargs);

  const ak::ContentPtr
    generate() const override;

private:
  const py::object callable_;
  const py::tuple args_;
  const py::dict kwargs_;
};

py::class_<PyArrayGenerator, std::shared_ptr<PyArrayGenerator>>
make_PyArrayGenerator(const py::handle& m, const std::string& name);

class PyArrayCache: public ak::ArrayCache {
public:
  PyArrayCache(const py::object& mutablemapping);

  ak::ContentPtr
    get(const std::string& key) const override;

  void
    set(const std::string& key, const ak::ContentPtr& value) override;

private:
  const py::object mutablemapping_;
};

py::class_<PyArrayCache, std::shared_ptr<PyArrayCache>>
make_PyArrayCache(const py::handle& m, const std::string& name);

#endif // AWKWARDPY_VIRTUAL_H_
