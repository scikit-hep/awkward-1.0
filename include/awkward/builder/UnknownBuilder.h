// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_UNKNOWNBUILDER_H_
#define AWKWARD_UNKNOWNBUILDER_H_

#include <vector>

#include "awkward/common.h"
#include "awkward/builder/Builder.h"

namespace awkward {
  class ArrayBuilderOptions;

  /// @class UnknownBuilder
  ///
  /// @brief Builder node for accumulated data whose type is not yet known.
  class LIBAWKWARD_EXPORT_SYMBOL UnknownBuilder: public Builder {
  public:
    /// @brief Create an empty UnknownBuilder.
    /// @param options Configuration options for building an array;
    /// these are passed to every Builder's constructor.
    static const BuilderPtr
      fromempty(const ArrayBuilderOptions& options);

    /// @brief Create a ListBuilder from a full set of parameters.
    ///
    /// @param options Configuration options for building an array;
    /// these are passed to every Builder's constructor.
    /// @param nullcount The number of null values encountered so far.
    UnknownBuilder(const ArrayBuilderOptions& options, int64_t nullcount);

    /// @brief User-friendly name of this class: `"UnknownBuilder"`.
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
    /// An UnknownBuilder is never active.
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
    const int64_t nullcount() const { return nullcount_; }

  private:
    const ArrayBuilderOptions options_;
    int64_t nullcount_;
  };
}

#endif // AWKWARD_UNKNOWNBUILDER_H_
