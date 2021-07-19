// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_ARRAYGENERATOR_H_
#define AWKWARD_ARRAYGENERATOR_H_

#include "awkward/Slice.h"
#include "awkward/Content.h"
#include "awkward/virtual/ArrayCache.h"

namespace awkward {
  ////////// ArrayGenerator

  class ArrayGenerator;
  using ArrayGeneratorPtr = std::shared_ptr<ArrayGenerator>;

  /// @class ArrayGenerator
  ///
  /// @brief Abstract superclass to generate arrays for VirtualArray, defining
  /// the interface.
  ///
  /// The main implementation, PyArrayGenerator, is passed through pybind11 to
  /// Python to work with Python functions and lambdas, but in principle, pure
  /// C++ generators could be written.
  class LIBAWKWARD_EXPORT_SYMBOL ArrayGenerator {
  public:
    /// @brief Called by subclasses to set the #form of an ArrayGenerator.
    ///
    /// The #form can be `nullptr`, in which case the generated array can have
    /// any Form and any Type, but doing so would cause VirtualArray to
    /// materialize in more circumstances, undermining its usefulness.
    ///
    /// Similarly, the #length can be specified to avoid materializing a
    /// VirtualArray when its length must be known. Any negative value,
    /// such as -1, is interpreted as "unknown."
    ArrayGenerator(const FormPtr& form, int64_t length);

    /// @brief Virtual destructor acts as a first non-inline virtual function
    /// that determines a specific translation unit in which vtable shall be
    /// emitted.
    virtual ~ArrayGenerator();

    /// @brief The Form the generated array is expected to take; may be
    /// `nullptr`.
    const FormPtr
      form() const;

    /// @brief The length the generated array is expected to have; may
    /// be negative to indicate that the length is unknown.
    int64_t
      length() const;

    /// @brief Creates an array but does not check it against the #form.
    virtual const ContentPtr
      generate() const = 0;

    /// @brief Creates an array and checks it against the #form.
    /// If form was not available initially, no check is made and the form
    /// inferred from the result is saved in case it is useful later
    const ContentPtr
      generate_and_check();

    /// @brief Accumulates all the unique #ArrayCache objects from nested
    /// #VirtualArray nodes. (Uniqueness is determined by pointer value.)
    virtual void
      caches(std::vector<ArrayCachePtr>& out) const = 0;

    /// @brief Returns a string representation of this ArrayGenerator.
    virtual const std::string
      tostring_part(const std::string& indent,
                    const std::string& pre,
                    const std::string& post) const = 0;

    /// @brief Copies this ArrayGenerator, referencing any contents.
    virtual const std::shared_ptr<ArrayGenerator>
      shallow_copy() const = 0;

    /// @brief Return a copy of this ArrayGenerator with a different form
    /// (or a now-known form, whereas it might have been unknown before).
    virtual const std::shared_ptr<ArrayGenerator>
      with_form(const FormPtr& form) const = 0;

    /// @brief Return a copy of this ArrayGenerator with a different length
    /// (or a now-known length, whereas it might have been unknown before).
    virtual const std::shared_ptr<ArrayGenerator>
      with_length(int64_t length) const = 0;

    /// @brief Returns `true` if this generator has all the same buffers and
    /// parameters as `other`; `false` otherwise.
    ///
    /// @param other The generator to compare this with.
    virtual bool
      referentially_equal(const ArrayGeneratorPtr& other) const = 0;

  protected:
    const FormPtr form_;
    FormPtr inferred_form_{nullptr};
    int64_t length_;
  };

  ////////// SliceGenerator

  /// @class SliceGenerator
  ///
  /// @brief Generator for lazy slicing. Used to avoid materializing a
  /// VirtualArray before its content is needed (in case its content is
  /// never needed).
  class LIBAWKWARD_EXPORT_SYMBOL SliceGenerator: public ArrayGenerator {
  public:
    SliceGenerator(const FormPtr& form,
                   int64_t length,
                   const ContentPtr& content,
                   const Slice& slice);

    const ContentPtr
      content() const;

    const Slice
      slice() const;

    const ContentPtr
      generate() const override;

    void
      caches(std::vector<ArrayCachePtr>& out) const override;

    const std::string
      tostring_part(const std::string& indent,
                    const std::string& pre,
                    const std::string& post) const override;

    const std::shared_ptr<ArrayGenerator>
      shallow_copy() const override;

    const std::shared_ptr<ArrayGenerator>
      with_form(const FormPtr& form) const override;

    const std::shared_ptr<ArrayGenerator>
      with_length(int64_t length) const override;

    virtual bool
      referentially_equal(const ArrayGeneratorPtr& other) const override;

  protected:
    const ContentPtr content_;
    const Slice slice_;
  };
}

#endif // AWKWARD_ARRAYGENERATOR_H_
