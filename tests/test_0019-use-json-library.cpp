// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#include <memory>

#include "awkward/Slice.h"
#include "awkward/builder/ArrayBuilder.h"
#include "awkward/builder/ArrayBuilderOptions.h"
#include "awkward/array/NumpyArray.h"

namespace ak = awkward;

void fill(ak::ArrayBuilder& builder, int64_t x) {
  builder.integer(x);
}

void fill(ak::ArrayBuilder& builder, double x) {
  builder.real(x);
}

void fill(ak::ArrayBuilder& builder, const char* x) {
  builder.bytestring(x);
}

void fill(ak::ArrayBuilder& builder, const std::string& x) {
  builder.bytestring(x.c_str());
}

template <typename T>
void fill(ak::ArrayBuilder& builder, const std::vector<T>& vector) {
  builder.beginlist();
  for (auto x : vector) {
    fill(builder, x);
  }
  builder.endlist();
}

int main(int, char**) {
  std::vector<std::vector<std::vector<double>>> vector =
    {{{0.0, 1.1, 2.2}, {}, {3.3, 4.4}}, {{5.5}}, {}, {{6.6, 7.7, 8.8, 9.9}}};

  ak::ArrayBuilder builder(ak::ArrayBuilderOptions(1024, 2.0));
  for (auto x : vector) fill(builder, x);
  std::shared_ptr<ak::Content> array = builder.builder().snapshot();

  // array[-1][0][1] == 7.7
  std::shared_ptr<ak::NumpyArray> scalar = std::dynamic_pointer_cast<ak::NumpyArray>(array.get()->getitem_at(-1).get()->getitem_at(0).get()->getitem_at(1));
  if (*reinterpret_cast<double*>(scalar.get()->data()) != 7.7)
    return -1;

  ak::Slice slice;
  slice.append(ak::SliceRange(ak::Slice::none(), ak::Slice::none(), -1));
  slice.append(ak::SliceRange(ak::Slice::none(), ak::Slice::none(), 2));
  slice.append(ak::SliceRange(1, ak::Slice::none(), ak::Slice::none()));

  if (array.get()->getitem(slice).get()->tojson(false, 1) !=
         "[[[7.7,8.8,9.9]],[],[[]],[[1.1,2.2],[4.4]]]")
    return -1;

  return 0;
}
