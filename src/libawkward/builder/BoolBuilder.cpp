// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/builder/BoolBuilder.cpp", line)

#include <stdexcept>

#include "awkward/builder/ArrayBuilderOptions.h"
#include "awkward/builder/OptionBuilder.h"
#include "awkward/builder/UnionBuilder.h"

#include "awkward/builder/BoolBuilder.h"

namespace awkward {
  const BuilderPtr
  BoolBuilder::fromempty(const ArrayBuilderOptions& options) {
    return std::make_shared<BoolBuilder>(options,
                                         GrowableBuffer<uint8_t>::empty(options));
  }

  BoolBuilder::BoolBuilder(const ArrayBuilderOptions& options,
                           const GrowableBuffer<uint8_t>& buffer)
      : options_(options)
      , buffer_(buffer) { }

  const std::string
  BoolBuilder::classname() const {
    return "BoolBuilder";
  };

  int64_t
  BoolBuilder::length() const {
    return buffer_.length();
  }

  void
  BoolBuilder::clear() {
    buffer_.clear();
  }

  const ContentPtr
  BoolBuilder::snapshot() const {
    throw std::invalid_argument(
      std::string("called obsolete 'BoolBuilder::snapshot'")
      + FILENAME(__LINE__));
  }

  bool
  BoolBuilder::active() const {
    return false;
  }

  const BuilderPtr
  BoolBuilder::null() {
    BuilderPtr out = OptionBuilder::fromvalids(options_, shared_from_this());
    out.get()->null();
    return out;
  }

  const BuilderPtr
  BoolBuilder::boolean(bool x) {
    buffer_.append(x);
    return shared_from_this();
  }

  const BuilderPtr
  BoolBuilder::integer(int64_t x) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->integer(x);
    return out;
  }

  const BuilderPtr
  BoolBuilder::real(double x) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->real(x);
    return out;
  }

  const BuilderPtr
  BoolBuilder::complex(std::complex<double> x) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->complex(x);
    return out;
  }

  const BuilderPtr
  BoolBuilder::datetime(int64_t x, const std::string& unit) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->datetime(x, unit);
    return out;
  }

  const BuilderPtr
  BoolBuilder::timedelta(int64_t x, const std::string& unit) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->timedelta(x, unit);
    return out;
  }

  const BuilderPtr
  BoolBuilder::string(const char* x, int64_t length, const char* encoding) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->string(x, length, encoding);
    return out;
  }

  const BuilderPtr
  BoolBuilder::beginlist() {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->beginlist();
    return out;
  }

  const BuilderPtr
  BoolBuilder::endlist() {
    throw std::invalid_argument(
      std::string("called 'end_list' without 'begin_list' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  BoolBuilder::begintuple(int64_t numfields) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->begintuple(numfields);
    return out;
  }

  const BuilderPtr
  BoolBuilder::index(int64_t index) {
    throw std::invalid_argument(
      std::string("called 'index' without 'begintuple' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  BoolBuilder::endtuple() {
    throw std::invalid_argument(
      std::string("called 'endtuple' without 'begintuple' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  BoolBuilder::beginrecord(const char* name, bool check) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->beginrecord(name, check);
    return out;
  }

  const BuilderPtr
  BoolBuilder::field(const char* key, bool check) {
    throw std::invalid_argument(
      std::string("called 'field' without 'beginrecord' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  BoolBuilder::endrecord() {
    throw std::invalid_argument(
      std::string("called 'endrecord' without 'beginrecord' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  BoolBuilder::append(const ContentPtr& array, int64_t at) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->append(array, at);
    return out;
  }

}
