// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_INDEXEDBUILDER_H_
#define AWKWARD_INDEXEDBUILDER_H_

#include <vector>

#include "awkward/common.h"
#include "awkward/array/IndexedArray.h"
#include "awkward/builder/GrowableBuffer.h"
#include "awkward/builder/Builder.h"

namespace awkward {
  class ArrayBuilderOptions;

  /// @class IndexedBuilder
  ///
  /// @brief Builder node for accumulated data that come from an existing
  /// Content array.
  template <typename T>
  class LIBAWKWARD_EXPORT_SYMBOL IndexedBuilder: public Builder {
  public:
    /// @brief Create an IndexedBuilder from a full set of parameters.
    ///
    /// @param options Configuration options for building an array;
    /// these are passed to every Builder's constructor.
    /// @param index Contains the accumulated index (like
    /// {@link IndexedArrayOf#index IndexedArray::index}).
    /// @param array The original Content array from which the new accumulated
    /// data are drawn.
    /// @param hasnull If `true`, some of the accumulated data are missing
    /// and a #snapshot should produce an
    /// {@link IndexedArrayOf IndexedOptionArray}, rather than an
    /// {@link IndexedArrayOf IndexedArray}.
    IndexedBuilder(const ArrayBuilderOptions& options,
                   const GrowableBuffer<int64_t>& index,
                   const std::shared_ptr<T>& array,
                   bool hasnull);

    /// @brief Raw pointer to the original Content `array`.
    const Content*
      arrayptr() const;

    int64_t
      length() const override;

    void
      clear() override;

    /// An IndexedBuilder is never active.
    bool
      active() const override;

    const BuilderPtr
      null() override;

    const BuilderPtr
      boolean(bool x) override;

    const BuilderPtr
      integer(int64_t x) override;

    const BuilderPtr
      real(double x) override;

    const BuilderPtr
      complex(std::complex<double> x) override;

    const BuilderPtr
      datetime(int64_t x, const std::string& unit) override;

    const BuilderPtr
      timedelta(int64_t x, const std::string& unit) override;

    const BuilderPtr
      string(const char* x, int64_t length, const char* encoding) override;

    const BuilderPtr
      beginlist() override;

    const BuilderPtr
      endlist() override;

    const BuilderPtr
      begintuple(int64_t numfields) override;

    const BuilderPtr
      index(int64_t index) override;

    const BuilderPtr
      endtuple() override;

    const BuilderPtr
      beginrecord(const char* name, bool check) override;

    const BuilderPtr
      field(const char* key, bool check) override;

    const BuilderPtr
      endrecord() override;

    // FIXME: refactor
    const GrowableBuffer<int64_t>& buffer() const { return index_; }

    bool has_null() const { return hasnull_; }

    const std::shared_ptr<T> array() const { return array_; }

  protected:
    const ArrayBuilderOptions options_;
    GrowableBuffer<int64_t> index_;
    const std::shared_ptr<T> array_;
    bool hasnull_;
  };

  class IndexedGenericBuilder: public IndexedBuilder<Content> {
  public:
    static const BuilderPtr
    fromnulls(const ArrayBuilderOptions& options,
              int64_t nullcount,
              const ContentPtr& array);

    IndexedGenericBuilder(const ArrayBuilderOptions& options,
                          const GrowableBuffer<int64_t>& index,
                          const ContentPtr& array,
                          bool hasnull);

    /// @brief User-friendly name of this class: `"IndexedGenericBuilder"`.
    const std::string
      classname() const override;

    const ContentPtr
      snapshot() const override;

    const BuilderPtr
      append(const ContentPtr& array, int64_t at) override;

    const std::shared_ptr<Content> array() const { return array_; }
  };

  class IndexedI32Builder: public IndexedBuilder<IndexedArray32> {
  public:
    IndexedI32Builder(const ArrayBuilderOptions& options,
                      const GrowableBuffer<int64_t>& index,
                      const std::shared_ptr<IndexedArray32>& array,
                      bool hasnull);

    /// @brief User-friendly name of this class: `"IndexedI32Builder"`.
    const std::string
      classname() const override;

    const ContentPtr
      snapshot() const override;

    const BuilderPtr
      append(const ContentPtr& array, int64_t at) override;
  };

  class IndexedIU32Builder: public IndexedBuilder<IndexedArrayU32> {
  public:
    IndexedIU32Builder(const ArrayBuilderOptions& options,
                       const GrowableBuffer<int64_t>& index,
                       const std::shared_ptr<IndexedArrayU32>& array,
                       bool hasnull);

    /// @brief User-friendly name of this class: `"IndexedIU32Builder"`.
    const std::string
      classname() const override;

    const ContentPtr
      snapshot() const override;

    const BuilderPtr
      append(const ContentPtr& array, int64_t at) override;
  };

  class IndexedI64Builder: public IndexedBuilder<IndexedArray64> {
  public:
    IndexedI64Builder(const ArrayBuilderOptions& options,
                      const GrowableBuffer<int64_t>& index,
                      const std::shared_ptr<IndexedArray64>& array,
                      bool hasnull);

    /// @brief User-friendly name of this class: `"IndexedI64Builder"`.
    const std::string
      classname() const override;

    const ContentPtr
      snapshot() const override;

    const BuilderPtr
      append(const ContentPtr& array, int64_t at) override;
  };

  class IndexedIO32Builder: public IndexedBuilder<IndexedOptionArray32> {
  public:
    IndexedIO32Builder(const ArrayBuilderOptions& options,
                       const GrowableBuffer<int64_t>& index,
                       const std::shared_ptr<IndexedOptionArray32>& array,
                       bool hasnull);

    /// @brief User-friendly name of this class: `"IndexedIO32Builder"`.
    const std::string
      classname() const override;

    const ContentPtr
      snapshot() const override;

    const BuilderPtr
      append(const ContentPtr& array, int64_t at) override;
  };

  class IndexedIO64Builder: public IndexedBuilder<IndexedOptionArray64> {
  public:
    IndexedIO64Builder(const ArrayBuilderOptions& options,
                       const GrowableBuffer<int64_t>& index,
                       const std::shared_ptr<IndexedOptionArray64>& array,
                       bool hasnull);

    /// @brief User-friendly name of this class: `"IndexedIO64Builder"`.
    const std::string
      classname() const override;

    const ContentPtr
      snapshot() const override;

    const BuilderPtr
      append(const ContentPtr& array, int64_t at) override;
  };

}

#endif // AWKWARD_INDEXEDBUILDER_H_
