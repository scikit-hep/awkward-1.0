// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/libawkward/builder/IndexedBuilder.cpp", line)

#include <stdexcept>

#include "awkward/builder/ArrayBuilderOptions.h"
#include "awkward/builder/UnionBuilder.h"

#include "awkward/builder/IndexedBuilder.h"

namespace awkward {
  template <typename T>
  IndexedBuilder<T>::IndexedBuilder(const ArrayBuilderOptions& options,
                                    const GrowableBuffer<int64_t>& index,
                                    const std::shared_ptr<T>& array,
                                    bool hasnull)
      : options_(options)
      , index_(index)
      , array_(array)
      , hasnull_(hasnull) { }

  template <typename T>
  const Content*
  IndexedBuilder<T>::arrayptr() const {
    return array_.get();
  }

  template <typename T>
  int64_t
  IndexedBuilder<T>::length() const {
    return index_.length();
  }

  template <typename T>
  void
  IndexedBuilder<T>::clear() {
    index_.clear();
  }

  template <typename T>
  bool
  IndexedBuilder<T>::active() const {
    return false;
  }

  template <typename T>
  const BuilderPtr
  IndexedBuilder<T>::null() {
    index_.append(-1);
    hasnull_ = true;
    return shared_from_this();
  }

  template <typename T>
  const BuilderPtr
  IndexedBuilder<T>::boolean(bool x) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->boolean(x);
    return out;
  }

  template <typename T>
  const BuilderPtr
  IndexedBuilder<T>::integer(int64_t x) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->integer(x);
    return out;
  }

  template <typename T>
  const BuilderPtr
  IndexedBuilder<T>::real(double x) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->real(x);
    return out;
  }

  template <typename T>
  const BuilderPtr
  IndexedBuilder<T>::complex(std::complex<double> x) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->complex(x);
    return out;
  }

  template <typename T>
  const BuilderPtr
  IndexedBuilder<T>::datetime(int64_t x, const std::string& unit) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->datetime(x, unit);
    return out;
  }

  template <typename T>
  const BuilderPtr
  IndexedBuilder<T>::timedelta(int64_t x, const std::string& unit) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->timedelta(x, unit);
    return out;
  }

  template <typename T>
  const BuilderPtr
  IndexedBuilder<T>::string(const char* x,
                            int64_t length,
                            const char* encoding) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->string(x, length, encoding);
    return out;
  }

  template <typename T>
  const BuilderPtr
  IndexedBuilder<T>::beginlist() {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->beginlist();
    return out;
  }

  template <typename T>
  const BuilderPtr
  IndexedBuilder<T>::endlist() {
    throw std::invalid_argument(
      std::string("called 'end_list' without 'begin_list' at the same level before it")
      + FILENAME(__LINE__));
  }

  template <typename T>
  const BuilderPtr
  IndexedBuilder<T>::begintuple(int64_t numfields) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->begintuple(numfields);
    return out;
  }

  template <typename T>
  const BuilderPtr
  IndexedBuilder<T>::index(int64_t index) {
    throw std::invalid_argument(
      std::string("called 'index' without 'begin_tuple' at the same level before it")
      + FILENAME(__LINE__));
  }

  template <typename T>
  const BuilderPtr
  IndexedBuilder<T>::endtuple() {
    throw std::invalid_argument(
      std::string("called 'end_tuple' without 'begin_tuple' at the same level before it")
      + FILENAME(__LINE__));
  }

  template <typename T>
  const BuilderPtr
  IndexedBuilder<T>::beginrecord(const char* name, bool check) {
    BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
    out.get()->beginrecord(name, check);
    return out;
  }

  template <typename T>
  const BuilderPtr
  IndexedBuilder<T>::field(const char* key, bool check) {
    throw std::invalid_argument(
      std::string("called 'field' without 'begin_record' at the same level before it")
      + FILENAME(__LINE__));
  }

  template <typename T>
  const BuilderPtr
  IndexedBuilder<T>::endrecord() {
    throw std::invalid_argument(
      std::string("called 'end_record' without 'begin_record' at the same level before it")
      + FILENAME(__LINE__));
  }

  ////////// IndexedGenericBuilder

  template class EXPORT_TEMPLATE_INST IndexedBuilder<Content>;

  const BuilderPtr
  IndexedGenericBuilder::fromnulls(const ArrayBuilderOptions& options,
                                   int64_t nullcount,
                                   const ContentPtr& array) {
    GrowableBuffer<int64_t> index =
      GrowableBuffer<int64_t>::full(options, -1, nullcount);
    if (std::shared_ptr<IndexedArray32> ptr =
        std::dynamic_pointer_cast<IndexedArray32>(array)) {
      return std::make_shared<IndexedI32Builder>(options,
                                                 index,
                                                 ptr,
                                                 nullcount != 0);
    }
    else if (std::shared_ptr<IndexedArrayU32> ptr =
             std::dynamic_pointer_cast<IndexedArrayU32>(array)) {
      return std::make_shared<IndexedIU32Builder>(options,
                                                  index,
                                                  ptr,
                                                  nullcount != 0);
    }
    else if (std::shared_ptr<IndexedArray64> ptr =
             std::dynamic_pointer_cast<IndexedArray64>(array)) {
      return std::make_shared<IndexedI64Builder>(options,
                                                 index,
                                                 ptr,
                                                 nullcount != 0);
    }
    else if (std::shared_ptr<IndexedOptionArray32> ptr =
             std::dynamic_pointer_cast<IndexedOptionArray32>(array)) {
      return std::make_shared<IndexedIO32Builder>(options,
                                                  index,
                                                  ptr,
                                                  nullcount != 0);
    }
    else if (std::shared_ptr<IndexedOptionArray64> ptr =
             std::dynamic_pointer_cast<IndexedOptionArray64>(array)) {
      return std::make_shared<IndexedIO64Builder>(options,
                                                  index,
                                                  ptr,
                                                  nullcount != 0);
    }
    else {
      return std::make_shared<IndexedGenericBuilder>(options,
                                                     index,
                                                     array,
                                                     nullcount != 0);
    }
  }

  IndexedGenericBuilder::IndexedGenericBuilder(
    const ArrayBuilderOptions& options,
    const GrowableBuffer<int64_t>& index,
    const ContentPtr& array,
    bool hasnull)
      : IndexedBuilder<Content>(options, index, array, hasnull) { }

  const std::string
  IndexedGenericBuilder::classname() const {
    return "IndexedGenericBuilder";
  };

  const ContentPtr
  IndexedGenericBuilder::snapshot() const {
    throw std::invalid_argument(
      std::string("called obsolete 'Float64Builder::snapshot'")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  IndexedGenericBuilder::append(const ContentPtr& array, int64_t at) {
    if (array.get() == array_.get()) {
      index_.append(at);
    }
    else {
      BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
      out.get()->append(array, at);
      return out;
    }
    return shared_from_this();
  }

  ////////// IndexedI32Builder (makes IndexedArray32)

  template class EXPORT_TEMPLATE_INST IndexedBuilder<IndexedArray32>;

  IndexedI32Builder::IndexedI32Builder(
    const ArrayBuilderOptions& options,
    const GrowableBuffer<int64_t>& index,
    const std::shared_ptr<IndexedArray32>& array,
    bool hasnull)
      : IndexedBuilder<IndexedArray32>(options, index, array, hasnull) { }

  const std::string
  IndexedI32Builder::classname() const {
    return "IndexedI32Builder";
  };

  const ContentPtr
  IndexedI32Builder::snapshot() const {
    throw std::invalid_argument(
      std::string("called obsolete 'IndexedI32Builder::snapshot'")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  IndexedI32Builder::append(const ContentPtr& array, int64_t at) {
    if (array.get() == array_.get()) {
      index_.append((int64_t)array_.get()->index_at_nowrap(at));
    }
    else {
      BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
      out.get()->append(array, at);
      return out;
    }
    return shared_from_this();
  }

  ////////// IndexedIU32Builder (makes IndexedArrayU32)

  template class EXPORT_TEMPLATE_INST IndexedBuilder<IndexedArrayU32>;

  IndexedIU32Builder::IndexedIU32Builder(
    const ArrayBuilderOptions& options,
    const GrowableBuffer<int64_t>& index,
    const std::shared_ptr<IndexedArrayU32>& array,
    bool hasnull)
      : IndexedBuilder<IndexedArrayU32>(options, index, array, hasnull) { }

  const std::string
  IndexedIU32Builder::classname() const {
    return "IndexedIU32Builder";
  };

  const ContentPtr
  IndexedIU32Builder::snapshot() const {
    throw std::invalid_argument(
      std::string("called obsolete 'IndexedIU32Builder::snapshot'")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  IndexedIU32Builder::append(const ContentPtr& array, int64_t at) {
    if (array.get() == array_.get()) {
      index_.append((int64_t)array_.get()->index_at_nowrap(at));
    }
    else {
      BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
      out.get()->append(array, at);
      return out;
    }
    return shared_from_this();
  }

  ////////// IndexedI64Builder (makes IndexedArray64)

  template class EXPORT_TEMPLATE_INST IndexedBuilder<IndexedArray64>;

  IndexedI64Builder::IndexedI64Builder(
    const ArrayBuilderOptions& options,
    const GrowableBuffer<int64_t>& index,
    const std::shared_ptr<IndexedArray64>& array,
    bool hasnull)
      : IndexedBuilder<IndexedArray64>(options, index, array, hasnull) { }

  const std::string
  IndexedI64Builder::classname() const {
    return "IndexedI64Builder";
  };

  const ContentPtr
  IndexedI64Builder::snapshot() const {
    throw std::invalid_argument(
      std::string("called obsolete 'IndexedI64Builder::snapshot'")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  IndexedI64Builder::append(const ContentPtr& array, int64_t at) {
    if (array.get() == array_.get()) {
      index_.append(array_.get()->index_at_nowrap(at));
    }
    else {
      BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
      out.get()->append(array, at);
      return out;
    }
    return shared_from_this();
  }

  ////////// IndexedIO32Builder (makes IndexedOptionArray32)

  template class EXPORT_TEMPLATE_INST IndexedBuilder<IndexedOptionArray32>;

  IndexedIO32Builder::IndexedIO32Builder(
    const ArrayBuilderOptions& options,
    const GrowableBuffer<int64_t>& index,
    const std::shared_ptr<IndexedOptionArray32>& array,
    bool hasnull)
      : IndexedBuilder<IndexedOptionArray32>(options,
                                             index,
                                             array,
                                             hasnull) { }

  const std::string
  IndexedIO32Builder::classname() const {
    return "IndexedIO32Builder";
  };

  const ContentPtr
  IndexedIO32Builder::snapshot() const {
    throw std::invalid_argument(
      std::string("called obsolete 'IndexedIO32Builder::snapshot'")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  IndexedIO32Builder::append(const ContentPtr& array, int64_t at) {
    if (array.get() == array_.get()) {
      index_.append((int64_t)array_.get()->index_at_nowrap(at));
    }
    else {
      BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
      out.get()->append(array, at);
      return out;
    }
    return shared_from_this();
  }

  ////////// IndexedIO64Builder (makes IndexedOptionArray64)

  template class EXPORT_TEMPLATE_INST IndexedBuilder<IndexedOptionArray64>;

  IndexedIO64Builder::IndexedIO64Builder(
    const ArrayBuilderOptions& options,
    const GrowableBuffer<int64_t>& index,
    const std::shared_ptr<IndexedOptionArray64>& array, bool hasnull)
      : IndexedBuilder<IndexedOptionArray64>(options,
                                             index,
                                             array,
                                             hasnull) { }

  const std::string
  IndexedIO64Builder::classname() const {
    return "IndexedIO64Builder";
  };

  const ContentPtr
  IndexedIO64Builder::snapshot() const {
    throw std::invalid_argument(
      std::string("called obsolete 'IndexedIO64Builder::snapshot'")
      + FILENAME(__LINE__));
  }

  const BuilderPtr
  IndexedIO64Builder::append(const ContentPtr& array, int64_t at) {
    if (array.get() == array_.get()) {
      index_.append(array_.get()->index_at_nowrap(at));
    }
    else {
      BuilderPtr out = UnionBuilder::fromsingle(options_, shared_from_this());
      out.get()->append(array, at);
      return out;
    }
    return shared_from_this();
  }

}
