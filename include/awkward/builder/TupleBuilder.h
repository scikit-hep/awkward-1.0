// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_TUPLEBUILDER_H_
#define AWKWARD_TUPLEBUILDER_H_

#include <vector>

#include "awkward/common.h"
#include "awkward/builder/GrowableBuffer.h"
#include "awkward/builder/Builder.h"

namespace awkward {
  class ArrayBuilderOptions;

  /// @class TupleBuilder
  ///
  /// @brief Builder node for accumulated tuples.
  class LIBAWKWARD_EXPORT_SYMBOL TupleBuilder: public Builder {
  public:
    /// @brief Create an empty TupleBuilder.
    /// @param options Configuration options for building an array;
    /// these are passed to every Builder's constructor.
    static const BuilderPtr
      fromempty(const ArrayBuilderOptions& options);

    /// @brief Create a TupleBuilder from a full set of parameters.
    ///
    /// @param options Configuration options for building an array;
    /// these are passed to every Builder's constructor.
    /// @param contents A Builder for each tuple field.
    /// @param length Length of accumulated array (same as #length).
    /// @param begun If `true`, the TupleBuilder is in an active state;
    /// `false` otherwise.
    /// @param nextindex The next field index to fill with data.
    TupleBuilder(const ArrayBuilderOptions& options,
                 const std::vector<BuilderPtr>& contents,
                 int64_t length,
                 bool begun,
                 size_t nextindex);

    /// @brief Current number of fields.
    int64_t
      numfields() const;

    /// @brief User-friendly name of this class: `"TupleBuilder"`.
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
    /// Calling #begintuple makes a TupleBuilder active; #endtuple makes it
    /// inactive.
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

    // FIXME: refactor
    const std::vector<BuilderPtr>& contents() const { return contents_; }

  private:
    const ArrayBuilderOptions options_;
    std::vector<BuilderPtr> contents_;
    int64_t length_;
    bool begun_;
    int64_t nextindex_;

    void
      maybeupdate(int64_t i, const BuilderPtr& tmp);
  };
}

#endif // AWKWARD_TUPLEBUILDER_H_
