// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/builder/DatetimeBuilder.cpp", line)

#include <stdexcept>

#include "awkward/builder/ArrayBuilderOptions.h"
#include "awkward/builder/Complex128Builder.h"
#include "awkward/builder/Float64Builder.h"
#include "awkward/builder/OptionBuilder.h"
#include "awkward/builder/UnionBuilder.h"

#include "awkward/builder/DatetimeBuilder.h"
#include "awkward/datetime_util.h"

namespace awkward {
  const BuilderPtr
  DatetimeBuilder::fromempty(const ArrayBuilderOptions& options, const std::string& units) {
    GrowableBuffer<int64_t> content = GrowableBuffer<int64_t>::empty(options);
    return std::make_shared<DatetimeBuilder>(options,
                                             content,
                                             units);
  }

  DatetimeBuilder::DatetimeBuilder(const ArrayBuilderOptions& options,
                                   const GrowableBuffer<int64_t>& content,
                                   const std::string& units)
      : options_(options)
      , content_(content)
      , units_(units) { }

  const std::string
  DatetimeBuilder::classname() const {
    return "DatetimeBuilder";
  };

  int64_t
  DatetimeBuilder::length() const {
    return content_.length();
  }

  void
  DatetimeBuilder::clear() {
    content_.clear();
  }

  const ContentPtr
  DatetimeBuilder::snapshot() const {
    throw std::invalid_argument(
      std::string("called obsolete 'DatetimeBuilder::snapshot'")
      + FILENAME(__LINE__));
  }

  bool
  DatetimeBuilder::active() const {
    return false;
  }

  const BuilderPtr
  DatetimeBuilder::null() {
    BuilderPtr out = OptionBuilder::fromvalids(options_, shared_from_this());
    out.get()->null();
    return out;
  }

  const BuilderPtr
  DatetimeBuilder::boolean(bool x) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->boolean(x);
    return out;
  }

  const BuilderPtr
  DatetimeBuilder::integer(int64_t x) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->integer(x);
    return out;
  }

  const BuilderPtr
  DatetimeBuilder::real(double x) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->real(x);
    return out;
  }

  const BuilderPtr
  DatetimeBuilder::complex(std::complex<double> x) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->complex(x);
    return out;
  }

  const BuilderPtr
  DatetimeBuilder::datetime(int64_t x, const std::string& unit) {
    if (unit == units_) {
      content_.append(x);
      return shared_from_this();
    }
    else {
      BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
      out.get()->datetime(x, unit);
      return out;
    }
  }

  const BuilderPtr
  DatetimeBuilder::timedelta(int64_t x, const std::string& unit) {
    if (unit == units_) {
      content_.append(x);
      return shared_from_this();
    }
    else {
      BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
      out.get()->timedelta(x, unit);
      return out;
    }
  }

  const BuilderPtr
  DatetimeBuilder::string(const char* x, int64_t length, const char* encoding) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->string(x, length, encoding);
    return out;
  }

  const BuilderPtr
  DatetimeBuilder::beginlist() {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->beginlist();
    return out;
  }

  const BuilderPtr
  DatetimeBuilder::endlist() {
    throw std::invalid_argument(
      std::string("called 'end_list' without 'begin_list' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  DatetimeBuilder::begintuple(int64_t numfields) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->begintuple(numfields);
    return out;
  }

  const BuilderPtr
  DatetimeBuilder::index(int64_t index) {
    throw std::invalid_argument(
      std::string("called 'index' without 'begin_tuple' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  DatetimeBuilder::endtuple() {
    throw std::invalid_argument(
      std::string("called 'end_tuple' without 'begin_tuple' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  DatetimeBuilder::beginrecord(const char* name, bool check) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->beginrecord(name, check);
    return out;
  }

  const BuilderPtr
  DatetimeBuilder::field(const char* key, bool check) {
    throw std::invalid_argument(
      std::string("called 'field' without 'begin_record' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  DatetimeBuilder::endrecord() {
    throw std::invalid_argument(
      std::string("called 'end_record' without 'begin_record' at the same level before it")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  DatetimeBuilder::append(const ContentPtr& array, int64_t at) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->append(array, at);
    return out;
  }

  const std::string&
  DatetimeBuilder::units() const {
    return units_;
  }

}
