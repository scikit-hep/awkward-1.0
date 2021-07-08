// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_UNIONBUILDER_H_
#define AWKWARD_UNIONBUILDER_H_

#include <vector>

#include "awkward/common.h"
#include "awkward/builder/GrowableBuffer.h"
#include "awkward/builder/Builder.h"

namespace awkward {
  class ArrayBuilderOptions;
  class TupleBuilder;
  class RecordBuilder;

  /// @class UnionBuilder
  ///
  /// @brief Builder node for accumulated heterogeneous data.
  class LIBAWKWARD_EXPORT_SYMBOL UnionBuilder: public Builder {
  public:
    static const BuilderPtr
      fromsingle(const ArrayBuilderOptions& options,
                 const BuilderPtr& firstcontent);

    /// @brief Create a UnionBuilder from a full set of parameters.
    ///
    /// @param options Configuration options for building an array;
    /// these are passed to every Builder's constructor.
    /// @param tags Contains the accumulated tags (like
    /// {@link UnionArrayOf#tags UnionArray::tags}).
    /// @param index Contains the accumulated index (like
    /// {@link UnionArrayOf#index UnionArray::index}).
    /// @param contents A Builder for each of the union's possibilities.
    UnionBuilder(const ArrayBuilderOptions& options,
                 const GrowableBuffer<int8_t>& tags,
                 const GrowableBuffer<int64_t>& index,
                 std::vector<BuilderPtr>& contents);

    /// @brief User-friendly name of this class: `"UnionBuilder"`.
    const std::string
      classname() const override;

    int64_t
      length() const override;

    void
      clear() override;

    const ContentPtr
      snapshot() const override;

    /// @copydoc Builder::active()
    ///
    /// A UnionBuilder is active if and only if one of its `contents` is
    /// active.
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

    const BuilderPtr
      append(const ContentPtr& array, int64_t at) override;

    // FIXME: re-factor
    const GrowableBuffer<int8_t>& tags() const {  return tags_; }

    const GrowableBuffer<int64_t>& index() const { return index_; }

    const std::vector<BuilderPtr>& contents() const { return contents_; }

  private:
    const ArrayBuilderOptions options_;
    GrowableBuffer<int8_t> tags_;
    GrowableBuffer<int64_t> index_;
    std::vector<BuilderPtr> contents_;
    int8_t current_;
  };
}

#endif // AWKWARD_UNIONBUILDER_H_
