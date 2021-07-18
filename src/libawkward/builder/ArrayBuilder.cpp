// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/builder/ArrayBuilder.cpp", line)

#include <sstream>
#include <stdexcept>

#include "awkward/common.h"
#include "awkward/Content.h"
#include "awkward/type/Type.h"
#include "awkward/builder/ArrayBuilderOptions.h"
#include "awkward/builder/Builder.h"
#include "awkward/builder/UnknownBuilder.h"

#include "awkward/builder/ArrayBuilder.h"

#include "awkward/refactoring.h"

namespace awkward {
  ArrayBuilder::ArrayBuilder(const ArrayBuilderOptions& options)
      : builder_(UnknownBuilder::fromempty(options)) { }

  const std::string
  ArrayBuilder::tostring() const {
    util::TypeStrs typestrs;
    typestrs["char"] = "char";
    typestrs["string"] = "string";
    std::stringstream out;
    out << "<ArrayBuilder length=\"" << length() << "\" type=\""
        << type(typestrs).get()->tostring() << "\"/>";
    return out.str();
  }

  int64_t
  ArrayBuilder::length() const {
    return builder_.get()->length();
  }

  void
  ArrayBuilder::clear() {
    builder_.get()->clear();
  }

  const TypePtr
  ArrayBuilder::type(const util::TypeStrs& typestrs) const {
    return builder_.get()->snapshot().get()->type(typestrs);
  }

  const ContentPtr
  ArrayBuilder::snapshot() const {
    // FIXME: refactor
    return ::builder_snapshot(builder());
  }

  const ContentPtr
  ArrayBuilder::getitem_at(int64_t at) const {
    throw std::invalid_argument(
      std::string("called obsolete 'ArrayBuilder::getitem_at'")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  ArrayBuilder::getitem_range(int64_t start, int64_t stop) const {
    throw std::invalid_argument(
      std::string("called obsolete 'ArrayBuilder::getitem_range'")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  ArrayBuilder::getitem_field(const std::string& key) const {
    throw std::invalid_argument(
      std::string("called obsolete 'ArrayBuilder::getitem_field'")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  ArrayBuilder::getitem_fields(const std::vector<std::string>& keys) const {
    throw std::invalid_argument(
      std::string("called obsolete 'ArrayBuilder::getitem_fields'")
      + FILENAME(__LINE__));
  }

  const ContentPtr
  ArrayBuilder::getitem(const Slice& where) const {
    throw std::invalid_argument(
      std::string("called obsolete 'ArrayBuilder::getitem'")
      + FILENAME(__LINE__));
  }

  void
  ArrayBuilder::null() {
    maybeupdate(builder_.get()->null());
  }

  void
  ArrayBuilder::boolean(bool x) {
    maybeupdate(builder_.get()->boolean(x));
  }

  void
  ArrayBuilder::integer(int64_t x) {
    maybeupdate(builder_.get()->integer(x));
  }

  void
  ArrayBuilder::real(double x) {
    maybeupdate(builder_.get()->real(x));
  }

  void
  ArrayBuilder::complex(std::complex<double> x) {
    maybeupdate(builder_.get()->complex(x));
  }

  void
  ArrayBuilder::datetime(int64_t x, const std::string& unit) {
    maybeupdate(builder_.get()->datetime(x, unit));
  }

  void
  ArrayBuilder::timedelta(int64_t x, const std::string& unit) {
    maybeupdate(builder_.get()->timedelta(x, unit));
  }

  void
  ArrayBuilder::bytestring(const char* x) {
    maybeupdate(builder_.get()->string(x, -1, no_encoding));
  }

  void
  ArrayBuilder::bytestring(const char* x, int64_t length) {
    maybeupdate(builder_.get()->string(x, length, no_encoding));
  }

  void
  ArrayBuilder::bytestring(const std::string& x) {
    bytestring(x.c_str(), (int64_t)x.length());
  }

  void
  ArrayBuilder::string(const char* x) {
    maybeupdate(builder_.get()->string(x, -1, utf8_encoding));
  }

  void
  ArrayBuilder::string(const char* x, int64_t length) {
    maybeupdate(builder_.get()->string(x, length, utf8_encoding));
  }

  void
  ArrayBuilder::string(const std::string& x) {
    string(x.c_str(), (int64_t)x.length());
  }

  void
  ArrayBuilder::beginlist() {
    maybeupdate(builder_.get()->beginlist());
  }

  void
  ArrayBuilder::endlist() {
    BuilderPtr tmp = builder_.get()->endlist();
    if (tmp.get() == nullptr) {
      throw std::invalid_argument(
        std::string("endlist doesn't match a corresponding beginlist")
        + FILENAME(__LINE__));
    }
    maybeupdate(tmp);
  }

  void
  ArrayBuilder::begintuple(int64_t numfields) {
    maybeupdate(builder_.get()->begintuple(numfields));
  }

  void
  ArrayBuilder::index(int64_t index) {
    maybeupdate(builder_.get()->index(index));
  }

  void
  ArrayBuilder::endtuple() {
    maybeupdate(builder_.get()->endtuple());
  }

  void
  ArrayBuilder::beginrecord() {
    beginrecord_fast(nullptr);
  }

  void
  ArrayBuilder::beginrecord_fast(const char* name) {
    maybeupdate(builder_.get()->beginrecord(name, false));
  }

  void
  ArrayBuilder::beginrecord_check(const char* name) {
    maybeupdate(builder_.get()->beginrecord(name, true));
  }

  void
  ArrayBuilder::beginrecord_check(const std::string& name) {
    beginrecord_check(name.c_str());
  }

  void
  ArrayBuilder::field_fast(const char* key) {
    maybeupdate(builder_.get()->field(key, false));
  }

  void
  ArrayBuilder::field_check(const char* key) {
    maybeupdate(builder_.get()->field(key, true));
  }

  void
  ArrayBuilder::field_check(const std::string& key) {
    field_check(key.c_str());
  }

  void
  ArrayBuilder::endrecord() {
    maybeupdate(builder_.get()->endrecord());
  }

  void
  ArrayBuilder::append(const ContentPtr& array, int64_t at) {
    int64_t length = array.get()->length();
    int64_t regular_at = at;
    if (regular_at < 0) {
      regular_at += length;
    }
    if (!(0 <= regular_at  &&  regular_at < length)) {
      throw std::invalid_argument(
        std::string("'append' index (")
        + std::to_string(at) + std::string(") out of bounds (")
        + std::to_string(length) + std::string(")")
        + FILENAME(__LINE__));
    }
    return append_nowrap(array, regular_at);
  }

  void
  ArrayBuilder::append_nowrap(const ContentPtr& array, int64_t at) {
    maybeupdate(builder_.get()->append(array, at));
  }

  void
  ArrayBuilder::extend(const ContentPtr& array) {
    BuilderPtr tmp = builder_;
    for (int64_t i = 0;  i < array.get()->length();  i++) {
      tmp = builder_.get()->append(array, i);
      maybeupdate(tmp);
    }
  }

  void
  ArrayBuilder::maybeupdate(const BuilderPtr& tmp) {
    if (tmp.get() != builder_.get()) {
      builder_ = tmp;
    }
  }

  const char* ArrayBuilder::no_encoding = nullptr;
  const char* ArrayBuilder::utf8_encoding = "utf-8";
}

////////// extern C interface

uint8_t awkward_ArrayBuilder_length(void* arraybuilder,
                                    int64_t* result) {
  awkward::ArrayBuilder* obj =
    reinterpret_cast<awkward::ArrayBuilder*>(arraybuilder);
  try {
    *result = obj->length();
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_ArrayBuilder_clear(void* arraybuilder) {
  awkward::ArrayBuilder* obj =
    reinterpret_cast<awkward::ArrayBuilder*>(arraybuilder);
  try {
    obj->clear();
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_ArrayBuilder_null(void* arraybuilder) {
  awkward::ArrayBuilder* obj =
    reinterpret_cast<awkward::ArrayBuilder*>(arraybuilder);
  try {
    obj->null();
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_ArrayBuilder_boolean(void* arraybuilder,
                                     bool x) {
  awkward::ArrayBuilder* obj =
    reinterpret_cast<awkward::ArrayBuilder*>(arraybuilder);
  try {
    obj->boolean(x);
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_ArrayBuilder_integer(void* arraybuilder,
                                     int64_t x) {
  awkward::ArrayBuilder* obj =
    reinterpret_cast<awkward::ArrayBuilder*>(arraybuilder);
  try {
    obj->integer(x);
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_ArrayBuilder_real(void* arraybuilder,
                                  double x) {
  awkward::ArrayBuilder* obj =
    reinterpret_cast<awkward::ArrayBuilder*>(arraybuilder);
  try {
    obj->real(x);
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_ArrayBuilder_bytestring(void* arraybuilder,
                                        const char* x) {
  awkward::ArrayBuilder* obj =
    reinterpret_cast<awkward::ArrayBuilder*>(arraybuilder);
  try {
    obj->bytestring(x);
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_ArrayBuilder_bytestring_length(void* arraybuilder,
                                               const char* x,
                                               int64_t length) {
  awkward::ArrayBuilder* obj =
    reinterpret_cast<awkward::ArrayBuilder*>(arraybuilder);
  try {
    obj->bytestring(x, length);
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_ArrayBuilder_string(void* arraybuilder,
                                    const char* x) {
  awkward::ArrayBuilder* obj =
    reinterpret_cast<awkward::ArrayBuilder*>(arraybuilder);
  try {
    obj->string(x);
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_ArrayBuilder_string_length(void* arraybuilder,
                                           const char* x,
                                           int64_t length) {
  awkward::ArrayBuilder* obj =
    reinterpret_cast<awkward::ArrayBuilder*>(arraybuilder);
  try {
    obj->string(x, length);
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_ArrayBuilder_beginlist(void* arraybuilder) {
  awkward::ArrayBuilder* obj =
    reinterpret_cast<awkward::ArrayBuilder*>(arraybuilder);
  try {
    obj->beginlist();
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_ArrayBuilder_endlist(void* arraybuilder) {
  awkward::ArrayBuilder* obj =
    reinterpret_cast<awkward::ArrayBuilder*>(arraybuilder);
  try {
    obj->endlist();
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_ArrayBuilder_begintuple(void* arraybuilder,
                                        int64_t numfields) {
  awkward::ArrayBuilder* obj =
    reinterpret_cast<awkward::ArrayBuilder*>(arraybuilder);
  try {
    obj->begintuple(numfields);
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_ArrayBuilder_index(void* arraybuilder,
                                   int64_t index) {
  awkward::ArrayBuilder* obj =
    reinterpret_cast<awkward::ArrayBuilder*>(arraybuilder);
  try {
    obj->index(index);
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_ArrayBuilder_endtuple(void* arraybuilder) {
  awkward::ArrayBuilder* obj =
    reinterpret_cast<awkward::ArrayBuilder*>(arraybuilder);
  try {
    obj->endtuple();
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_ArrayBuilder_beginrecord(void* arraybuilder) {
  awkward::ArrayBuilder* obj =
    reinterpret_cast<awkward::ArrayBuilder*>(arraybuilder);
  try {
    obj->beginrecord();
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_ArrayBuilder_beginrecord_fast(void* arraybuilder,
                                              const char* name) {
  awkward::ArrayBuilder* obj =
    reinterpret_cast<awkward::ArrayBuilder*>(arraybuilder);
  try {
    obj->beginrecord_fast(name);
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_ArrayBuilder_beginrecord_check(void* arraybuilder,
                                               const char* name) {
  awkward::ArrayBuilder* obj =
    reinterpret_cast<awkward::ArrayBuilder*>(arraybuilder);
  try {
    obj->beginrecord_check(name);
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_ArrayBuilder_field_fast(void* arraybuilder,
                                        const char* key) {
  awkward::ArrayBuilder* obj =
    reinterpret_cast<awkward::ArrayBuilder*>(arraybuilder);
  try {
    obj->field_fast(key);
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_ArrayBuilder_field_check(void* arraybuilder,
                                         const char* key) {
  awkward::ArrayBuilder* obj =
    reinterpret_cast<awkward::ArrayBuilder*>(arraybuilder);
  try {
    obj->field_check(key);
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_ArrayBuilder_endrecord(void* arraybuilder) {
  awkward::ArrayBuilder* obj =
    reinterpret_cast<awkward::ArrayBuilder*>(arraybuilder);
  try {
    obj->endrecord();
  }
  catch (...) {
    return 1;
  }
  return 0;
}

uint8_t awkward_ArrayBuilder_append_nowrap(void* arraybuilder,
                                           const void* shared_ptr_ptr,
                                           int64_t at) {
  awkward::ArrayBuilder* obj =
    reinterpret_cast<awkward::ArrayBuilder*>(arraybuilder);
  const std::shared_ptr<awkward::Content>* array =
    reinterpret_cast<const std::shared_ptr<awkward::Content>*>(shared_ptr_ptr);
  try {
    obj->append_nowrap(*array, at);
  }
  catch (...) {
    return 1;
  }
  return 0;
}
