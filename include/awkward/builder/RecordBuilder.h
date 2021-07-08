// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_RECORDBUILDER_H_
#define AWKWARD_RECORDBUILDER_H_

#include <vector>

#include "awkward/common.h"
#include "awkward/builder/GrowableBuffer.h"
#include "awkward/builder/Builder.h"

namespace awkward {
  class ArrayBuilderOptions;

  /// @class RecordBuilder
  ///
  /// @brief Builder node for accumulated records.
  class LIBAWKWARD_EXPORT_SYMBOL RecordBuilder: public Builder {
  public:
    /// @brief Create an empty RecordBuilder.
    /// @param options Configuration options for building an array;
    /// these are passed to every Builder's constructor.
    static const BuilderPtr
      fromempty(const ArrayBuilderOptions& options);

    /// @brief Create a RecordBuilder from a full set of parameters.
    ///
    /// @param options Configuration options for building an array;
    /// these are passed to every Builder's constructor.
    /// @param contents A Builder for each record field.
    /// @param keys Names for each record field.
    /// @param pointers String pointers for each record field name.
    /// @param name String name of the record.
    /// @param nameptr String pointer for the name of the record.
    /// @param length Length of accumulated array (same as #length).
    /// @param begun If `true`, the RecordBuilder is in an active state;
    /// `false` otherwise.
    /// @param nextindex The next field index to fill with data.
    /// @param nexttotry The next field index to check against a key string.
    RecordBuilder(const ArrayBuilderOptions& options,
                  const std::vector<BuilderPtr>& contents,
                  const std::vector<std::string>& keys,
                  const std::vector<const char*>& pointers,
                  const std::string& name,
                  const char* nameptr,
                  int64_t length,
                  bool begun,
                  int64_t nextindex,
                  int64_t nexttotry);

    /// @brief Name of the record (STL wrapped #nameptr).
    const std::string
      name() const;

    /// @brief String pointer for the name of the record.
    const char*
      nameptr() const;

    /// @brief User-friendly name of this class: `"RecordBuilder"`.
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
    /// Calling #beginrecord makes a RecordBuilder active; #endrecord makes it
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
    const std::vector<std::string>& keys() const { return keys_; }

    // FIXME: refactor
    const std::vector<BuilderPtr>& builders() const { return contents_; }

  private:
    const BuilderPtr
      field_fast(const char* key);

    const BuilderPtr
      field_check(const char* key);

    const ArrayBuilderOptions options_;
    std::vector<BuilderPtr> contents_;
    std::vector<std::string> keys_;
    std::vector<const char*> pointers_;
    std::string name_;
    const char* nameptr_;
    int64_t length_;
    bool begun_;
    int64_t nextindex_;
    int64_t nexttotry_;
    int64_t keys_size_;

    void
      maybeupdate(int64_t i, const BuilderPtr& tmp);
  };
}

#endif // AWKWARD_RECORDBUILDER_H_
