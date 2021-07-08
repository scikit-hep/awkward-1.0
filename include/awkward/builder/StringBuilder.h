// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_STRINGBUILDER_H_
#define AWKWARD_STRINGBUILDER_H_

#include "awkward/common.h"
#include "awkward/builder/GrowableBuffer.h"
#include "awkward/builder/Builder.h"

namespace awkward {
  class ArrayBuilderOptions;

  /// @class StringBuilder
  ///
  /// @brief Builder node that accumulates strings.
  class LIBAWKWARD_EXPORT_SYMBOL StringBuilder: public Builder {
  public:
    /// @brief Create an empty StringBuilder.
    /// @param options Configuration options for building an array;
    /// these are passed to every Builder's constructor.
    /// @param encoding If `nullptr`, the string is an unencoded bytestring;
    /// if `"utf-8"`, it is encoded with variable-width UTF-8.
    /// Currently, no other encodings have been defined.
    static const BuilderPtr
      fromempty(const ArrayBuilderOptions& options, const char* encoding);

    /// @brief Create a StringBuilder from a full set of parameters.
    ///
    /// @param options Configuration options for building an array;
    /// these are passed to every Builder's constructor.
    /// @param offsets Contains the accumulated offsets (like
    /// {@link ListOffsetArrayOf#offsets ListOffsetArray::offsets}).
    /// @param content Another GrowableBuffer, but for the characters in all
    /// the strings.
    /// @param encoding If `nullptr`, the string is an unencoded bytestring;
    /// if `"utf-8"`, it is encoded with variable-width UTF-8.
    /// Currently, no other encodings have been defined.
    StringBuilder(const ArrayBuilderOptions& options,
                  const GrowableBuffer<int64_t>& offsets,
                  const GrowableBuffer<uint8_t>& content,
                  const char* encoding);

    /// @brief If `nullptr`, the string is an unencoded bytestring;
    /// if `"utf-8"`, it is encoded with variable-width UTF-8.
    /// Currently, no other encodings have been defined.
    const char*
      encoding() const;

    /// @brief User-friendly name of this class: `"StringBuilder"`.
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
    /// A StringBuilder is never active.
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
    const GrowableBuffer<int64_t>& buffer() const { return offsets_; }

    // FIXME: refactor
    const GrowableBuffer<uint8_t>& content() const { return content_; }

  private:
    const ArrayBuilderOptions options_;
    GrowableBuffer<int64_t> offsets_;
    GrowableBuffer<uint8_t> content_;
    const char* encoding_;
  };

}

#endif // AWKWARD_STRINGBUILDER_H_
