// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <stdexcept>

#include "awkward/Identity.h"
#include "awkward/fillable/OptionFillable.h"
#include "awkward/fillable/BoolFillable.h"
#include "awkward/fillable/Int64Fillable.h"

#include "awkward/fillable/UnknownFillable.h"

namespace awkward {
  int64_t UnknownFillable::length() const {
    return nullcount_;
  }

  void UnknownFillable::clear() {
    nullcount_ = 0;
  }

  const std::shared_ptr<Type> UnknownFillable::type() const {
    return std::shared_ptr<Type>(nullptr);
  }

  const std::shared_ptr<Content> UnknownFillable::snapshot() const {
    throw std::runtime_error("UnknownFillable::snapshot() needs MaskedArray");
  }

  Fillable* UnknownFillable::null() {
    nullcount_++;
    return this;
  }

  Fillable* UnknownFillable::boolean(bool x) {
    Fillable* out = prepare<BoolFillable>();
    out->boolean(x);
    return out;
  }

  Fillable* UnknownFillable::integer(int64_t x) {
    Fillable* out = prepare<Int64Fillable>();
    out->integer(x);
    return out;
  }

  template <typename T>
  Fillable* UnknownFillable::prepare() const {
    Fillable* out = new T(options_);
    if (nullcount_ != 0) {
      out = OptionFillable::fromnulls(options_, nullcount_, out);
    }
    return out;
  }

}
