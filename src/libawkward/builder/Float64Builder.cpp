// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/builder/Float64Builder.cpp", line)

#include <stdexcept>

#include "awkward/builder/ArrayBuilderOptions.h"
#include "awkward/builder/Complex128Builder.h"
#include "awkward/builder/OptionBuilder.h"
#include "awkward/builder/UnionBuilder.h"

#include "awkward/builder/Float64Builder.h"

namespace awkward {
  const BuilderPtr
  Float64Builder::fromempty(const ArrayBuilderOptions& options) {
    return std::make_shared<Float64Builder>(options,
                                            GrowableBuffer<double>::empty(options));
  }

  const BuilderPtr
  Float64Builder::fromint64(const ArrayBuilderOptions& options,
                            const GrowableBuffer<int64_t>& old) {
    GrowableBuffer<double> buffer =
      GrowableBuffer<double>::empty(options, old.reserved());
    int64_t* oldraw = old.ptr().get();
    double* newraw = buffer.ptr().get();
    for (int64_t i = 0;  i < old.length();  i++) {
      newraw[i] = (double)oldraw[i];
    }
    buffer.set_length(old.length());
    return std::make_shared<Float64Builder>(options, buffer);
  }

  Float64Builder::Float64Builder(const ArrayBuilderOptions& options,
                                 const GrowableBuffer<double>& buffer)
      : options_(options)
      , buffer_(buffer) { }

  const GrowableBuffer<double>
  Float64Builder::buffer() const {
    return buffer_;
  }

  const std::string
  Float64Builder::classname() const {
    return "Float64Builder";
  }

  int64_t
  Float64Builder::length() const {
    return buffer_.length();
  }

  void
  Float64Builder::clear() {
    buffer_.clear();
  }

  const ContentPtr
  Float64Builder::snapshot() const {
    throw std::invalid_argument(
      std::string("called obsolete 'Float64Builder::snapshot'")
      + FILENAME(__LINE__));
  }

  bool
  Float64Builder::active() const {
    return false;
  }

  const BuilderPtr
  Float64Builder::null() {
    BuilderPtr out = OptionBuilder::fromvalids(options_, shared_from_this());
    out.get()->null();
    return out;
  }

  const BuilderPtr
  Float64Builder::boolean(bool x) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->boolean(x);
    return out;
  }

  const BuilderPtr
  Float64Builder::integer(int64_t x) {
    buffer_.append((double)x);
    return shared_from_this();
  }

  const BuilderPtr
  Float64Builder::real(double x) {
    buffer_.append(x);
    return shared_from_this();
  }

  const BuilderPtr
  Float64Builder::complex(std::complex<double> x) {
    BuilderPtr out = Complex128Builder::fromfloat64(options_, buffer_);
    out.get()->complex(x);
    return out;
  }

  const BuilderPtr
  Float64Builder::datetime(int64_t x, const std::string& unit) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->datetime(x, unit);
    return out;
  }

  const BuilderPtr
  Float64Builder::timedelta(int64_t x, const std::string& unit) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->timedelta(x, unit);
    return out;
  }

  const BuilderPtr
  Float64Builder::string(const char* x, int64_t length, const char* encoding) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->string(x, length, encoding);
    return out;
  }

  const BuilderPtr
  Float64Builder::beginlist() {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->beginlist();
    return out;
  }

  const BuilderPtr
  Float64Builder::endlist() {
    throw std::invalid_argument(
      std::string("called 'end_list' without 'begin_list' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  Float64Builder::begintuple(int64_t numfields) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->begintuple(numfields);
    return out;
  }

  const BuilderPtr
  Float64Builder::index(int64_t index) {
    throw std::invalid_argument(
      std::string("called 'index' without 'begin_tuple' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  Float64Builder::endtuple() {
    throw std::invalid_argument(
      std::string("called 'end_tuple' without 'begin_tuple' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  Float64Builder::beginrecord(const char* name, bool check) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->beginrecord(name, check);
    return out;
  }

  const BuilderPtr
  Float64Builder::field(const char* key, bool check) {
    throw std::invalid_argument(
      std::string("called 'field' without 'begin_record' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  Float64Builder::endrecord() {
    throw std::invalid_argument(
      std::string("called 'end_record' without 'begin_record' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  Float64Builder::append(const ContentPtr& array, int64_t at) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->append(array, at);
    return out;
  }
}
