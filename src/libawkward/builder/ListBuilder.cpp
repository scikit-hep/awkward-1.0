// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/builder/ListBuilder.cpp", line)

#include <stdexcept>

#include "awkward/builder/ArrayBuilderOptions.h"
#include "awkward/builder/OptionBuilder.h"
#include "awkward/builder/UnionBuilder.h"
#include "awkward/builder/UnknownBuilder.h"

#include "awkward/builder/ListBuilder.h"

namespace awkward {
  const BuilderPtr
  ListBuilder::fromempty(const ArrayBuilderOptions& options) {
    GrowableBuffer<int64_t> offsets = GrowableBuffer<int64_t>::empty(options);
    offsets.append(0);
    return std::make_shared<ListBuilder>(options,
                                         offsets,
                                         UnknownBuilder::fromempty(options),
                                         false);
  }

  ListBuilder::ListBuilder(const ArrayBuilderOptions& options,
                           const GrowableBuffer<int64_t>& offsets,
                           const BuilderPtr& content,
                           bool begun)
      : options_(options)
      , offsets_(offsets)
      , content_(content)
      , begun_(begun) { }

  const std::string
  ListBuilder::classname() const {
    return "ListBuilder";
  };

  int64_t
  ListBuilder::length() const {
    return offsets_.length() - 1;
  }

  void
  ListBuilder::clear() {
    offsets_.clear();
    offsets_.append(0);
    content_.get()->clear();
  }

  const ContentPtr
  ListBuilder::snapshot() const {
    throw std::invalid_argument(
      std::string("called obsolete 'ListBuilder::snapshot'")
      + FILENAME(__LINE__));
  }

  bool
  ListBuilder::active() const {
    return begun_;
  }

  const BuilderPtr
  ListBuilder::null() {
    if (!begun_) {
      BuilderPtr out = OptionBuilder::fromvalids(options_, shared_from_this());
      out.get()->null();
      return out;
    }
    else {
      maybeupdate(content_.get()->null());
      return shared_from_this();
    }
  }

  const BuilderPtr
  ListBuilder::boolean(bool x) {
    if (!begun_) {
      BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
      out.get()->boolean(x);
      return out;
    }
    else {
      maybeupdate(content_.get()->boolean(x));
      return shared_from_this();
    }
  }

  const BuilderPtr
  ListBuilder::integer(int64_t x) {
    if (!begun_) {
      BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
      out.get()->integer(x);
      return out;
    }
    else {
      maybeupdate(content_.get()->integer(x));
      return shared_from_this();
    }
  }

  const BuilderPtr
  ListBuilder::real(double x) {
    if (!begun_) {
      BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
      out.get()->real(x);
      return out;
    }
    else {
      maybeupdate(content_.get()->real(x));
      return shared_from_this();
    }
  }

  const BuilderPtr
  ListBuilder::complex(std::complex<double> x) {
    if (!begun_) {
      BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
      out.get()->complex(x);
      return out;
    }
    else {
      maybeupdate(content_.get()->complex(x));
      return shared_from_this();
    }
  }

  const BuilderPtr
  ListBuilder::datetime(int64_t x, const std::string& unit) {
    if (!begun_) {
      BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
      out.get()->datetime(x, unit);
      return out;
    }
    else {
      maybeupdate(content_.get()->datetime(x, unit));
      return shared_from_this();
    }
  }

  const BuilderPtr
  ListBuilder::timedelta(int64_t x, const std::string& unit) {
    if (!begun_) {
      BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
      out.get()->timedelta(x, unit);
      return out;
    }
    else {
      maybeupdate(content_.get()->timedelta(x, unit));
      return shared_from_this();
    }
  }

  const BuilderPtr
  ListBuilder::string(const char* x, int64_t length, const char* encoding) {
    if (!begun_) {
      BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
      out.get()->string(x, length, encoding);
      return out;
    }
    else {
      maybeupdate(content_.get()->string(x, length, encoding));
      return shared_from_this();
    }
  }

  const BuilderPtr
  ListBuilder::beginlist() {
    if (!begun_) {
      begun_ = true;
    }
    else {
      maybeupdate(content_.get()->beginlist());
    }
    return shared_from_this();
  }

  const BuilderPtr
  ListBuilder::endlist() {
    if (!begun_) {
      throw std::invalid_argument(
        std::string("called 'end_list' without 'begin_list' at the same level before it")
        + FILENAME(__LINE__));
    }
    else if (!content_.get()->active()) {
      offsets_.append(content_.get()->length());
      begun_ = false;
    }
    else {
      maybeupdate(content_.get()->endlist());
    }
    return shared_from_this();
  }

  const BuilderPtr
  ListBuilder::begintuple(int64_t numfields) {
    if (!begun_) {
      BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
      out.get()->begintuple(numfields);
      return out;
    }
    else {
      maybeupdate(content_.get()->begintuple(numfields));
      return shared_from_this();
    }
  }

  const BuilderPtr
  ListBuilder::index(int64_t index) {
    if (!begun_) {
      throw std::invalid_argument(
        std::string("called 'index' without 'begin_tuple' at the same level before it")
        + FILENAME(__LINE__));
    }
    else {
      content_.get()->index(index);
      return shared_from_this();
    }
  }

  const BuilderPtr
  ListBuilder::endtuple() {
    if (!begun_) {
      throw std::invalid_argument(
        std::string("called 'end_tuple' without 'begin_tuple' at the same level before it")
        + FILENAME(__LINE__));
    }
    else {
      content_.get()->endtuple();
      return shared_from_this();
    }
  }

  const BuilderPtr
  ListBuilder::beginrecord(const char* name, bool check) {
    if (!begun_) {
      BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
      out.get()->beginrecord(name, check);
      return out;
    }
    else {
      maybeupdate(content_.get()->beginrecord(name, check));
      return shared_from_this();
    }
  }

  const BuilderPtr
  ListBuilder::field(const char* key, bool check) {
    if (!begun_) {
      throw std::invalid_argument(
        std::string("called 'field' without 'begin_record' at the same level before it")
        + FILENAME(__LINE__));
    }
    else {
      content_.get()->field(key, check);
      return shared_from_this();
    }
  }

  const BuilderPtr
  ListBuilder::endrecord() {
    if (!begun_) {
      throw std::invalid_argument(
        std::string("called 'end_record' without 'begin_record' at the same level "
                    "before it") + FILENAME(__LINE__));
    }
    else {
      content_.get()->endrecord();
      return shared_from_this();
    }
  }

  const BuilderPtr
  ListBuilder::append(const ContentPtr& array, int64_t at) {
    if (!begun_) {
      BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
      out.get()->append(array, at);
      return out;
    }
    else {
      maybeupdate(content_.get()->append(array, at));
      return shared_from_this();
    }
  }

  void
  ListBuilder::maybeupdate(const BuilderPtr& tmp) {
    if (tmp.get() != content_.get()) {
      content_ = tmp;
    }
  }
}
