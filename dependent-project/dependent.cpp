// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#include <iostream>

#include <pybind11/pybind11.h>

#include "awkward/Content.h"
#include "awkward/builder/ArrayBuilder.h"
#include "awkward/builder/ArrayBuilderOptions.h"
#include "awkward/kernels.h"

namespace py = pybind11;
namespace ak = awkward;

std::shared_ptr<ak::Content> producer() {
  ak::ArrayBuilder builder(ak::ArrayBuilderOptions(1024, 2.0));

  builder.real(1.1);
  builder.real(2.2);
  builder.real(3.3);

  builder.beginlist();
  builder.integer(1);
  builder.integer(2);
  builder.integer(3);
  builder.endlist();

  builder.beginlist();
  builder.endlist();

  builder.beginlist();
  builder.integer(4);
  builder.integer(5);
  builder.endlist();

  builder.beginrecord();
  builder.field_fast("x");
  builder.real(12.3);
  builder.field_fast("y");
  builder.string("wow");
  builder.endrecord();

  return builder.snapshot();
}

std::string consumer(const std::shared_ptr<ak::Content>& array) {
  return array.get()->tojson(false, 10);
}

PYBIND11_MODULE(dependent, m) {
  // Ensure dependencies are loaded.
  py::module::import("awkward");

  m.def("producer", &producer);
  m.def("consumer", &consumer);
}
