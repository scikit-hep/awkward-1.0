// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_LISTOFFSETARRAYBUILDER_H_
#define AWKWARD_LISTOFFSETARRAYBUILDER_H_

#include "awkward/layoutbuilder/FormBuilder.h"

namespace awkward {

  class ListOffsetForm;
  using ListOffsetFormPtr = std::shared_ptr<ListOffsetForm>;
  using FormBuilderPtr = std::shared_ptr<FormBuilder>;

  /// @class ListOffsetArrayBuilder
  ///
  /// @brief
  class LIBAWKWARD_EXPORT_SYMBOL ListOffsetArrayBuilder : public FormBuilder {
  public:
    /// @brief Creates a ListOffsetArrayBuilder from a full set of parameters.
    ListOffsetArrayBuilder(const ListOffsetFormPtr& form,
                           const std::string attribute = "offsets",
                           const std::string partition = "0");

    /// @brief User-friendly name of this class.
    const std::string
      classname() const override;

    /// @brief Turns the accumulated data into a Content array.
    const ContentPtr
      snapshot(const ForthOutputBufferMap& outputs) const override;

    /// @brief The Form describing the array.
    const FormPtr
      form() const override;

    /// @brief AwkwardForth virtual machine instructions of the data outputs.
    const std::string
      vm_output() const override;

    /// @brief AwkwardForth virtual machine data output key.
    const std::string
      vm_output_data() const override;

    /// @brief AwkwardForth virtual machine instructions of the array builder function.
    const std::string
      vm_func() const override;

    /// @brief The array builder VM function name.
    const std::string
      vm_func_name() const override;

    /// @brief The array builder VM function type.
    const std::string
      vm_func_type() const override;

    /// @brief AwkwardForth virtual machine instructions to retrieve the data from
    /// the VM stack.
    const std::string
      vm_from_stack() const override;

    /// @brief Error messages in the AwkwardForth virtual machine instructions.
    const std::string
      vm_error() const override;

    /// @brief Adds a boolean value `x` to the accumulated data.
    void
      boolean(bool x, LayoutBuilder* builder) override;

    /// @brief Adds an integer value `x` to the accumulated data.
    void
      int64(int64_t x, LayoutBuilder* builder) override;

    /// @brief Adds a real value `x` to the accumulated data.
    void
      float64(double x, LayoutBuilder* builder) override;

    /// @brief Adds a complex value `x` to the accumulated data.
    void
      complex(std::complex<double> x, LayoutBuilder* builder) override;

    /// @brief Adds an unencoded bytestring `x` in STL format to the
    /// accumulated data.
    void
      bytestring(const std::string& x, LayoutBuilder* builder) override;

    /// @brief Adds a UTF-8 encoded bytestring `x` in STL format to the
    /// accumulated data.
    void
      string(const std::string& x, LayoutBuilder* builder) override;

    /// @brief Begins building a nested list.
    void
      begin_list(LayoutBuilder* builder) override;

    /// @brief Ends a nested list.
    void
      end_list(LayoutBuilder* builder) override;

    /// @brief true if the builder is accumulating data
    bool
      active() override;

  private:
    /// @brief This builder Form
    const ListOffsetFormPtr form_;

    /// @brief 'true' if this builder 'array' parameter is 'string',
    /// 'false' otherwise.
    bool is_string_builder_;

    /// @brief 'true' if this builder has received a 'begin_list' command.
    /// 'false' if the builder either has not received a 'begin_list' command
    /// or has received an 'end_list' command.
    bool begun_;

    /// @brief an output buffer name is
    /// "part{partition}-{form_key}-{attribute}"
    const FormKey form_key_;
    const std::string attribute_;
    const std::string partition_;

    /// @brief This Form content builder
    FormBuilderPtr content_;

    /// @brief Forth virtual machine instructions
    /// generated from the Form
    std::string vm_output_;
    std::string vm_output_data_;
    std::string vm_func_;
    std::string vm_func_name_;
    std::string vm_func_type_;
    std::string vm_data_from_stack_;
    std::string vm_error_;
  };

}

#endif // AWKWARD_LISTOFFSETARRAYBUILDER_H_
