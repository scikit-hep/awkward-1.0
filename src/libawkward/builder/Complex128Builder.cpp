// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/builder/Complex128Builder.cpp", line)

#include <stdexcept>

#include "awkward/builder/OptionBuilder.h"
#include "awkward/builder/UnionBuilder.h"

#include "awkward/builder/Complex128Builder.h"

namespace awkward {
  const BuilderPtr
  Complex128Builder::fromempty(const ArrayBuilderOptions& options) {
    return std::make_shared<Complex128Builder>(options,
                                              GrowableBuffer<std::complex<double>>::empty(options));
  }

  const BuilderPtr
  Complex128Builder::fromint64(const ArrayBuilderOptions& options,
                               const GrowableBuffer<int64_t>& old) {
    GrowableBuffer<std::complex<double>> buffer =
      GrowableBuffer<std::complex<double>>::empty(options, old.reserved());
    int64_t* oldraw = old.ptr().get();
    std::complex<double>* newraw = buffer.ptr().get();
    for (int64_t i = 0;  i < 2*old.length();  i++) {
      newraw[i] = {static_cast<double>(oldraw[i]), 0};
    }
    buffer.set_length(old.length());
    return std::make_shared<Complex128Builder>(options, buffer);
  }

  const BuilderPtr
  Complex128Builder::fromfloat64(const ArrayBuilderOptions& options,
                                 const GrowableBuffer<double>& old) {
    GrowableBuffer<std::complex<double>> buffer =
      GrowableBuffer<std::complex<double>>::empty(options, old.reserved());
    double* oldraw = old.ptr().get();
    std::complex<double>* newraw = buffer.ptr().get();
    for (int64_t i = 0;  i < old.length();  i++) {
      newraw[i] = std::complex<double>(oldraw[i], 0);
    }
    buffer.set_length(old.length());
    return std::make_shared<Complex128Builder>(options, buffer);
  }

  Complex128Builder::Complex128Builder(const ArrayBuilderOptions& options,
                                       const GrowableBuffer<std::complex<double>>& buffer)
      : options_(options)
      , buffer_(buffer) { }

  const std::string
  Complex128Builder::classname() const {
    return "Complex128Builder";
  }

  int64_t
  Complex128Builder::length() const {
    return buffer_.length();
  }

  void
  Complex128Builder::clear() {
    buffer_.clear();
  }

  const ContentPtr
  Complex128Builder::snapshot() const {
    throw std::invalid_argument(
      std::string("called obsolete 'Complex128Builder::snapshot'")
      + FILENAME(__LINE__));
  }

  bool
  Complex128Builder::active() const {
    return false;
  }

  const BuilderPtr
  Complex128Builder::null() {
    BuilderPtr out = OptionBuilder::fromvalids(options_, shared_from_this());
    out.get()->null();
    return out;
  }

  const BuilderPtr
  Complex128Builder::boolean(bool x) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->boolean(x);
    return out;
  }

  const BuilderPtr
  Complex128Builder::integer(int64_t x) {
    buffer_.append(std::complex<double>((double)x, 0));
    return shared_from_this();
  }

  const BuilderPtr
  Complex128Builder::real(double x) {
    buffer_.append(std::complex<double>((double)x, 0));
    return shared_from_this();
  }

  const BuilderPtr
  Complex128Builder::complex(std::complex<double> x) {
    buffer_.append(x);
    return shared_from_this();
  }

  const BuilderPtr
  Complex128Builder::datetime(int64_t x, const std::string& unit) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->datetime(x, unit);
    return out;
  }

  const BuilderPtr
  Complex128Builder::timedelta(int64_t x, const std::string& unit) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->timedelta(x, unit);
    return out;
  }

  const BuilderPtr
  Complex128Builder::string(const char* x, int64_t length, const char* encoding) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->string(x, length, encoding);
    return out;
  }

  const BuilderPtr
  Complex128Builder::beginlist() {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->beginlist();
    return out;
  }

  const BuilderPtr
  Complex128Builder::endlist() {
    throw std::invalid_argument(
      std::string("called 'end_list' without 'begin_list' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  Complex128Builder::begintuple(int64_t numfields) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->begintuple(numfields);
    return out;
  }

  const BuilderPtr
  Complex128Builder::index(int64_t index) {
    throw std::invalid_argument(
      std::string("called 'index' without 'begin_tuple' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  Complex128Builder::endtuple() {
    throw std::invalid_argument(
      std::string("called 'end_tuple' without 'begin_tuple' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  Complex128Builder::beginrecord(const char* name, bool check) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->beginrecord(name, check);
    return out;
  }

  const BuilderPtr
  Complex128Builder::field(const char* key, bool check) {
    throw std::invalid_argument(
      std::string("called 'field' without 'begin_record' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  Complex128Builder::endrecord() {
    return shared_from_this();
  }

  const BuilderPtr
  Complex128Builder::append(const ContentPtr& array, int64_t at) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->append(array, at);
    return out;
  }
}
