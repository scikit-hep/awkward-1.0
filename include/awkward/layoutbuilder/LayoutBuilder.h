// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_LayoutBuilder_H_
#define AWKWARD_LayoutBuilder_H_

#include "awkward/common.h"
#include "awkward/util.h"
#include "awkward/forth/ForthMachine.h"
#include "awkward/layoutbuilder/FormBuilder.h"

#include <complex>

namespace awkward {
  class ArrayBuilderOptions;

  class Content;
  using ContentPtr = std::shared_ptr<Content>;
  class Slice;
  class Type;
  using TypePtr = std::shared_ptr<Type>;

  class Form;
  using FormPtr = std::shared_ptr<Form>;

  using ForthOtputBufferMap = std::map<std::string, std::shared_ptr<ForthOutputBuffer>>;

  class FormBuilder;
  using FormBuilderPtr = std::shared_ptr<FormBuilder>;

  const std::string
    index_form_to_name(Index::Form form);

  const std::string
    index_form_to_vm_format(Index::Form form);

  enum class state : std::int32_t {
    int64 = 0,
    float64 = 1,
    begin_list = 2,
    end_list = 3,
    boolean = 4,
    int8 = 5,
    int16 = 6,
    int32 = 7,
    uint8 = 8,
    uint16 = 9,
    uint32 = 10,
    uint64 = 11,
    float16 = 12,
    float32 = 13,
    float128 = 14,
    complex64 = 15,
    complex128 = 16,
    complex256 = 17,
    null = 18,
    index = 19,
    tag = 20
  };
  using utype = std::underlying_type<state>::type;

  const std::string
    dtype_to_state(util::dtype dt);

  const std::string
    dtype_to_vm_format(util::dtype dt);

  /// @class LayoutBuilder
  ///
  /// @brief User interface to the FormBuilder system: the LayoutBuilder is a
  /// fixed reference while the FormBuilder subclass instances change in
  /// response to accumulating data.
  class LIBAWKWARD_EXPORT_SYMBOL LayoutBuilder {
  public:
    /// @brief Creates an LayoutBuilder from a full set of parameters.
    ///
    /// @param form The Form that defines the Array to be build.
    /// @param options The Array builder options.
    /// @param vm_init If 'true' the Virtual Machine is instantiated on
    /// construction. If 'false' an external Virtial Machine must be connected
    /// to the builder. The flag is used for debugging.
    LayoutBuilder(const FormPtr& form,
                      const ArrayBuilderOptions& options,
                      bool vm_init = true);

    /// @brief Connects a Virtual Machine if it was not initialized before.
    void
      connect(const std::shared_ptr<ForthMachine32>& vm);

    /// @brief Prints debug information from the Virtual Machine stack.
    void
      debug_step() const;

    /// @brief Returns the Form used to build the Array.
    const FormPtr
      form() const;

    /// @brief Returns an AwkwardForth source code generated from the 'Form' and
    /// passed to the 'ForthMachine' virtual machine.
    const std::string
      vm_source() const;

    /// @brief Returns a string representation of this array (single-line XML
    /// indicating the length and type).
    const std::string
      tostring() const;

    /// @brief Current length of the accumulated array.
    int64_t
      length() const;

    /// @brief Current high level Type of the accumulated array.
    ///
    /// @param typestrs A mapping from `"__record__"` parameters to string
    /// representations of those types, to override the derived strings.
    const TypePtr
      type(const util::TypeStrs& typestrs) const;

    /// @brief Turns the accumulated data into a Content array.
    ///
    /// This operation only converts FormBuilder nodes into Content nodes; the
    /// buffers holding array data are shared between the FormBuilder and the
    /// Content. Hence, taking a snapshot is a constant-time operation.
    const ContentPtr
      snapshot() const;

    /// @brief Returns the element at a given position in the array, handling
    /// negative indexing and bounds-checking like Python.
    ///
    /// The first item in the array is at `0`, the second at `1`, the last at
    /// `-1`, the penultimate at `-2`, etc.
    const ContentPtr
      getitem_at(int64_t at) const;

    /// @brief Subinterval of this array, handling negative indexing
    /// and bounds-checking like Python.
    ///
    /// The first item in the array is at `0`, the second at `1`, the last at
    /// `-1`, the penultimate at `-2`, etc.
    ///
    /// Ranges beyond the array are not an error; they are trimmed to
    /// `start = 0` on the left and `stop = length() - 1` on the right.
    const ContentPtr
      getitem_range(int64_t start, int64_t stop) const;

    /// @brief This array with the first nested RecordArray replaced by
    /// the field at `key`.
    const ContentPtr
      getitem_field(const std::string& key) const;

    /// @brief This array with the first nested RecordArray replaced by
    /// a RecordArray of a given subset of `keys`.
    const ContentPtr
      getitem_fields(const std::vector<std::string>& keys) const;

    /// @brief Entry point for general slicing: Slice represents a tuple of
    /// SliceItem nodes applying to each level of nested lists.
    const ContentPtr
      getitem(const Slice& where) const;

    /// @brief Adds a `null` value to the accumulated data.
    void
      null();

    /// @brief Adds a boolean value `x` to the accumulated data.
    void
      boolean(bool x);

    /// @brief Adds an integer value `x` to the accumulated data.
    void
      int64(int64_t x);

    /// @brief Adds a real value `x` to the accumulated data.
    void
      float64(double x);

    /// @brief Adds a complex value `x` to the accumulated data.
    void
      complex(std::complex<double> x);

    /// @brief Adds an unencoded, null-terminated bytestring value `x` to the
    /// accumulated data.
    void
      bytestring(const char* x);

    /// @brief Adds an unencoded bytestring value `x` with a given `length`
    /// to the accumulated data.
    ///
    /// The string does not need to be null-terminated.
    void
      bytestring(const char* x, int64_t length);

    /// @brief Adds an unencoded bytestring `x` in STL format to the
    /// accumulated data.
    void
      bytestring(const std::string& x);

    /// @brief Adds a UTF-8 encoded, null-terminated bytestring value `x` to
    /// the accumulated data.
    void
      string(const char* x);

    /// @brief Adds a UTF-8 encoded bytestring value `x` with a given `length`
    /// to the accumulated data.
    ///
    /// The string does not need to be null-terminated.
    void
      string(const char* x, int64_t length);

    /// @brief Adds a UTF-8 encoded bytestring `x` in STL format to the
    /// accumulated data.
    void
      string(const std::string& x);

    /// @brief Begins building a nested list.
    ///
    /// The first 'beginlist' puts AwkwardForth VM into a state that expects
    /// another 'beginlist' or 'endlist'.
    /// The second puts the VM into a state that expects 'int64', etc.
    /// or 'endlist'.
    void
      begin_list();

    /// @brief Begins building a nested list.
    void
      add_begin_list();

    /// @brief Ends a nested list.
    void
      end_list();

    /// @brief Ends a nested list.
    void
      add_end_list();

    /// @brief Sets the pointer to a given tag `tag`; the next
    /// command will fill that slot.
    void
      tag(int8_t tag);

    /// @brief Issues an 'index' vm command. The value 'x' is pushed to
    /// the VM stack, it is not added to the accumulated data, e.g.
    /// the VM output buffer.
    ///
    /// This is used to build a 'categorical' array.
    void
      index(int64_t x);

    /// @brief Finds an index of a data in a VM output buffer.
    /// This is used to build a 'categorical' array.
    template <typename T>
    bool
      find_index_of(T x, const std::string& vm_output_data) {
        auto const& outputs = vm_.get()->outputs();
        auto search = outputs.find(vm_output_data);
        if (search != outputs.end()) {
          auto data = std::static_pointer_cast<T>(search->second.get()->ptr());
          auto size = search->second.get()->len();
          for (int64_t i = 0; i < size; i++) {
            if (data.get()[i] == x) {
              index(i);
              return true;
            }
          }
        }
        return false;
      }

    /// @brief Adds an integer value `x` to the accumulated data.
    template <typename T>
    void
      add(T x);

    /// @brief Generates an Array builder from a Form
    static FormBuilderPtr
      formBuilderFromA(const FormPtr& form);

    /// @brief Generates next unique ID
    static int64_t
      next_id();

    /// @brief Generates a user-defined error ID
    static int64_t
      next_error_id();

    /// @brief Resume Virtual machine run.
    void
      resume() const;

  protected:
    /// @brief A unique ID to use when Form nodes do not have Form key
    /// defined.
    static int64_t
      next_node_id;

    /// @brief An error ID to be used to generate a user 'halt' message.
    static int64_t
      error_id;

  private:
    /// @ brief Initialise Virtual machine.
    void
      initialise();

    /// @brief Place data of a type 'T' to the VM output buffer.
    template <typename T>
    void
      set_data(T x);

    /// See #initial.
    int64_t initial_;

    /// @brief length of an input buffer
    int64_t length_;

    /// @brief Root node of the FormBuilder tree.
    FormBuilderPtr builder_;

    /// @brief Virtual machine.
    std::shared_ptr<ForthMachine32> vm_;

    /// @brief Virtual machine input buffers.
    std::map<std::string, std::shared_ptr<ForthInputBuffer>> vm_inputs_map_;

    /// @brief Input data label.
    std::string vm_input_data_;

    /// @brief Virtual machine source code.
    std::string vm_source_;

    /// @brief Virtual machine errors to ignore.
    std::set<util::ForthError> ignore_;

  };

}

#endif // AWKWARD_LayoutBuilder_H_
