// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "awkward/array/RegularArray.h"

#include "awkward/array/VirtualArray.h"

namespace awkward {
  ////////// VirtualForm

  VirtualForm::VirtualForm(bool has_identities,
                           const util::Parameters& parameters,
                           const FormPtr& form,
                           bool has_length)
      : Form(has_identities, parameters)
      , form_(form)
      , has_length_(has_length) { }

  bool
  VirtualForm::has_form() const {
    return form_.get() != nullptr;
  }

  const FormPtr
  VirtualForm::form() const {
    return form_;
  }

  bool
  VirtualForm::has_length() const {
    return has_length_;
  }

  const TypePtr
  VirtualForm::type(const util::TypeStrs& typestrs) const {
    if (form_.get() == nullptr) {
      throw std::invalid_argument(
          "VirtualForm cannot determine its type without an expected Form");
    }
    else {
      return form_.get()->type(typestrs);
    }
  }

  void
  VirtualForm::tojson_part(ToJson& builder, bool verbose) const {
    builder.beginrecord();
    builder.field("class");
    builder.string("VirtualArray");
    builder.field("form");
    if (form_.get() == nullptr) {
      builder.null();
    }
    else {
      form_.get()->tojson_part(builder, verbose);
    }
    builder.field("has_length");
    builder.boolean(has_length_);
    identities_tojson(builder, verbose);
    parameters_tojson(builder, verbose);
    builder.endrecord();
  }

  const FormPtr
  VirtualForm::shallow_copy() const {
    return std::make_shared<VirtualForm>(has_identities_,
                                         parameters_,
                                         form_,
                                         has_length_);
  }

  const std::string
  VirtualForm::purelist_parameter(const std::string& key) const {
    std::string out = parameter(key);
    if (out == std::string("null")) {
      if (form_.get() == nullptr) {
        throw std::invalid_argument("VirtualForm cannot determine nested "
                                    "parameters without an expected Form");
      }
      else {
        return form_.get()->purelist_parameter(key);
      }
    }
    else {
      return out;
    }
  }

  bool
  VirtualForm::purelist_isregular() const {
    if (form_.get() == nullptr) {
      throw std::invalid_argument(
          "VirtualForm cannot determine its type without an expected Form");
    }
    else {
      return form_.get()->purelist_isregular();
    }
  }

  int64_t
  VirtualForm::purelist_depth() const {
    if (form_.get() == nullptr) {
      throw std::invalid_argument(
          "VirtualForm cannot determine its type without an expected Form");
    }
    else {
      return form_.get()->purelist_depth();
    }
  }

  const std::pair<int64_t, int64_t>
  VirtualForm::minmax_depth() const {
    if (form_.get() == nullptr) {
      throw std::invalid_argument(
          "VirtualForm cannot determine its type without an expected Form");
    }
    else {
      return form_.get()->minmax_depth();
    }
  }

  const std::pair<bool, int64_t>
  VirtualForm::branch_depth() const {
    if (form_.get() == nullptr) {
      throw std::invalid_argument(
          "VirtualForm cannot determine its type without an expected Form");
    }
    else {
      return form_.get()->branch_depth();
    }
  }

  int64_t
  VirtualForm::numfields() const {
    if (form_.get() == nullptr) {
      throw std::invalid_argument(
          "VirtualForm cannot determine its type without an expected Form");
    }
    else {
      return form_.get()->numfields();
    }
  }

  int64_t
  VirtualForm::fieldindex(const std::string& key) const {
    if (form_.get() == nullptr) {
      throw std::invalid_argument(
          "VirtualForm cannot determine its type without an expected Form");
    }
    else {
      return form_.get()->fieldindex(key);
    }
  }

  const std::string
  VirtualForm::key(int64_t fieldindex) const {
    if (form_.get() == nullptr) {
      throw std::invalid_argument(
          "VirtualForm cannot determine its type without an expected Form");
    }
    else {
      return form_.get()->key(fieldindex);
    }
  }

  bool
  VirtualForm::haskey(const std::string& key) const {
    if (form_.get() == nullptr) {
      throw std::invalid_argument(
          "VirtualForm cannot determine its type without an expected Form");
    }
    else {
      return form_.get()->haskey(key);
    }
  }

  const std::vector<std::string>
  VirtualForm::keys() const {
    if (form_.get() == nullptr) {
      throw std::invalid_argument(
          "VirtualForm cannot determine its type without an expected Form");
    }
    else {
      return form_.get()->keys();
    }
  }

  bool
  VirtualForm::equal(const FormPtr& other,
                      bool check_identities,
                      bool check_parameters) const {
    if (check_identities  &&
        has_identities_ != other.get()->has_identities()) {
      return false;
    }
    if (check_parameters  &&
        !util::parameters_equal(parameters_, other.get()->parameters())) {
      return false;
    }
    if (VirtualForm* t = dynamic_cast<VirtualForm*>(other.get())) {
      if (form_.get() == nullptr  &&  t->form().get() != nullptr) {
        return false;
      }
      else if (form_.get() != nullptr  &&  t->form().get() == nullptr) {
        return false;
      }
      else if (form_.get() != nullptr  &&  t->form().get() != nullptr) {
        if (!form_.get()->equal(t->form(),
                                check_identities,
                                check_parameters)) {
          return false;
        }
      }
      return has_length_ == t->has_length();
    }
    else {
      return false;
    }
  }

  ////////// VirtualArray

  VirtualArray::VirtualArray(const IdentitiesPtr& identities,
                             const util::Parameters& parameters,
                             const ArrayGeneratorPtr& generator,
                             const ArrayCachePtr& cache,
                             const std::string& cache_key)
      : Content(identities, parameters)
      , generator_(generator)
      , cache_(cache)
      , cache_key_(cache_key) { }

  VirtualArray::VirtualArray(const IdentitiesPtr& identities,
                             const util::Parameters& parameters,
                             const ArrayGeneratorPtr& generator,
                             const ArrayCachePtr& cache)
      : Content(identities, parameters)
      , generator_(generator)
      , cache_(cache)
      , cache_key_(ArrayCache::newkey()) { }

  const ArrayGeneratorPtr
  VirtualArray::generator() const {
    return generator_;
  }

  const ArrayCachePtr
  VirtualArray::cache() const {
    return cache_;
  }

  const ContentPtr
  VirtualArray::peek_array() const {
    if (cache_.get() != nullptr) {
      return cache_.get()->get(cache_key());
    }
    return ContentPtr(nullptr);
  }

  const ContentPtr
  VirtualArray::array() const {
    ContentPtr out(nullptr);
    if (cache_.get() != nullptr) {
      out = cache_.get()->get(cache_key());
    }
    if (out.get() == nullptr) {
      out = generator_.get()->generate_and_check();
    }
    if (cache_.get() != nullptr) {
      cache_.get()->set(cache_key(), out);
    }
    return out;
  }

  const std::string
  VirtualArray::cache_key() const {
    return cache_key_;
  }

  const std::string
  VirtualArray::classname() const {
    return "VirtualArray";
  }

  void
  VirtualArray::setidentities(const IdentitiesPtr& identities) {
    throw std::runtime_error("FIXME: VirtualArray::setidentities(identities)");
  }

  void
  VirtualArray::setidentities() {
    throw std::runtime_error("FIXME: VirtualArray::setidentities");
  }

  const TypePtr
  VirtualArray::type(const util::TypeStrs& typestrs) const {
    return form().get()->type(typestrs);
  }

  const FormPtr
  VirtualArray::form() const {
    FormPtr generator_form = generator_.get()->form();
    if (generator_form.get() == nullptr) {
      generator_form = array().get()->form();
    }
    int64_t generator_length = generator_.get()->length();
    return std::make_shared<VirtualForm>(identities_.get() != nullptr,
                                         parameters_,
                                         generator_form,
                                         generator_length >= 0);
  }

  const std::string
  VirtualArray::tostring_part(const std::string& indent,
                            const std::string& pre,
                            const std::string& post) const {
    std::stringstream out;
    out << indent << pre << "<" << classname()
        << " cache_key=\"" << cache_key_ << "\">\n";
    if (identities_.get() != nullptr) {
      out << identities_.get()->tostring_part(
               indent + std::string("    "), "", "\n");
    }
    if (!parameters_.empty()) {
      out << parameters_tostring(indent + std::string("    "), "", "\n");
    }
    out << generator_.get()->tostring_part(indent + std::string("    "),
                                           "", "\n");
    if (cache_.get() != nullptr) {
      out << cache_.get()->tostring_part(indent + std::string("    "),
                                         "", "\n");
    }
    ContentPtr peek = peek_array();
    if (peek.get() != nullptr) {
      out << peek.get()->tostring_part(
               indent + std::string("    "), "<array>", "</array>\n");
    }
    out << indent << "</" << classname() << ">" << post;
    return out.str();
  }

  void
  VirtualArray::tojson_part(ToJson& builder,
                          bool include_beginendlist) const {
    return array().get()->tojson_part(builder, include_beginendlist);
  }

  void
  VirtualArray::nbytes_part(std::map<size_t, int64_t>& largest) const { }

  int64_t
  VirtualArray::length() const {
    int64_t out = generator_.get()->length();
    if (out < 0) {
      out = array().get()->length();
    }
    return out;
  }

  const ContentPtr
  VirtualArray::shallow_copy() const {
    return std::make_shared<VirtualArray>(identities_,
                                          parameters_,
                                          generator_,
                                          cache_,
                                          cache_key_);
  }

  const ContentPtr
  VirtualArray::deep_copy(bool copyarrays,
                          bool copyindexes,
                          bool copyidentities) const {
    return array().get()->deep_copy(copyarrays, copyindexes, copyidentities);
  }

  void
  VirtualArray::check_for_iteration() const { }

  const ContentPtr
  VirtualArray::getitem_nothing() const {
    return array().get()->getitem_nothing();
  }

  const ContentPtr
  VirtualArray::getitem_at(int64_t at) const {
    int64_t regular_at = at;
    if (regular_at < 0) {
      regular_at += length();
    }
    if (!(0 <= regular_at  &&  regular_at < length())) {
      util::handle_error(failure("index out of range", kSliceNone, at),
                         classname(),
                         identities_.get());
    }
    return getitem_at_nowrap(regular_at);
  }

  const ContentPtr
  VirtualArray::getitem_at_nowrap(int64_t at) const {
    return array().get()->getitem_at_nowrap(at);
  }

  const ContentPtr
  VirtualArray::getitem_range(int64_t start, int64_t stop) const {
    if (generator_.get()->length() < 0) {
      return array().get()->getitem_range(start, stop);
    }
    else {
      ContentPtr peek = peek_array();
      if (peek.get() != nullptr) {
        return peek.get()->getitem_range(start, stop);
      }

      int64_t regular_start = start;
      int64_t regular_stop = stop;
      awkward_regularize_rangeslice(&regular_start, &regular_stop,
        true, start != Slice::none(), stop != Slice::none(),
        generator_.get()->length());
      return getitem_range_nowrap(regular_start, regular_stop);
    }
  }

  const ContentPtr
  VirtualArray::getitem_range_nowrap(int64_t start, int64_t stop) const {
    ContentPtr peek = peek_array();
    if (peek.get() != nullptr) {
      return peek.get()->getitem_range_nowrap(start, stop);
    }

    if (generator_.get()->length() >= 0  &&
        start == 0  &&
        stop == generator_.get()->length()) {
      return shallow_copy();
    }

    Slice slice;
    slice.append(SliceRange(start, stop, 1));
    slice.become_sealed();
    FormPtr form(nullptr);
    ArrayGeneratorPtr generator = std::make_shared<SliceGenerator>(
                 form, stop - start, generator_, slice);
    ArrayCachePtr cache(nullptr);
    return std::make_shared<VirtualArray>(Identities::none(),
                                          parameters_,
                                          generator,
                                          cache);
  }

  const ContentPtr
  VirtualArray::getitem_field(const std::string& key) const {
    ContentPtr peek = peek_array();
    if (peek.get() != nullptr) {
      return peek.get()->getitem_field(key);
    }

    Slice slice;
    slice.append(SliceField(key));
    slice.become_sealed();
    FormPtr form(nullptr);
    ArrayGeneratorPtr generator = std::make_shared<SliceGenerator>(
                 form, generator_.get()->length(), generator_, slice);
    ArrayCachePtr cache(nullptr);
    return std::make_shared<VirtualArray>(Identities::none(),
                                          util::Parameters(),
                                          generator,
                                          cache);
  }

  const ContentPtr
  VirtualArray::getitem_fields(const std::vector<std::string>& keys) const {
    ContentPtr peek = peek_array();
    if (peek.get() != nullptr) {
      return peek.get()->getitem_fields(keys);
    }

    Slice slice;
    slice.append(SliceFields(keys));
    slice.become_sealed();
    FormPtr form(nullptr);
    ArrayGeneratorPtr generator = std::make_shared<SliceGenerator>(
                 form, generator_.get()->length(), generator_, slice);
    ArrayCachePtr cache(nullptr);
    return std::make_shared<VirtualArray>(Identities::none(),
                                          util::Parameters(),
                                          generator,
                                          cache);
  }

  const ContentPtr
  VirtualArray::carry(const Index64& carry) const {
    ContentPtr peek = peek_array();
    if (peek.get() != nullptr) {
      return peek.get()->carry(carry);
    }

    Slice slice;
    std::vector<int64_t> shape({ carry.length() });
    std::vector<int64_t> strides({ 1 });
    slice.append(SliceArray64(carry, shape, strides, false));
    slice.become_sealed();
    FormPtr form(nullptr);
    ArrayGeneratorPtr generator = std::make_shared<SliceGenerator>(
                 form, carry.length(), generator_, slice);
    ArrayCachePtr cache(nullptr);
    return std::make_shared<VirtualArray>(Identities::none(),
                                          parameters_,
                                          generator,
                                          cache);
  }

  const std::string
  VirtualArray::validityerror(const std::string& path) const {
    return array().get()->validityerror(path + ".array");
  }

  const ContentPtr
  VirtualArray::shallow_simplify() const {
    return array().get()->shallow_simplify();
  }

  const ContentPtr
  VirtualArray::num(int64_t axis, int64_t depth) const {
    return array().get()->num(axis, depth);
  }

  const std::pair<Index64, ContentPtr>
  VirtualArray::offsets_and_flattened(int64_t axis, int64_t depth) const {
    return array().get()->offsets_and_flattened(axis, depth);
  }

  bool
  VirtualArray::mergeable(const ContentPtr& other, bool mergebool) const {
    return array().get()->mergeable(other, mergebool);
  }

  const ContentPtr
  VirtualArray::merge(const ContentPtr& other) const {
    return array().get()->merge(other);
  }

  const SliceItemPtr
  VirtualArray::asslice() const {
    return array().get()->asslice();
  }

  const ContentPtr
  VirtualArray::fillna(const ContentPtr& value) const {
    return array().get()->fillna(value);
  }

  const ContentPtr
  VirtualArray::rpad(int64_t target, int64_t axis, int64_t depth) const {
    return array().get()->rpad(target, axis, depth);
  }

  const ContentPtr
  VirtualArray::rpad_and_clip(int64_t target,
                              int64_t axis,
                              int64_t depth) const {
    return array().get()->rpad_and_clip(target, axis, depth);
  }

  const ContentPtr
  VirtualArray::reduce_next(const Reducer& reducer,
                          int64_t negaxis,
                          const Index64& starts,
                          const Index64& parents,
                          int64_t outlength,
                          bool mask,
                          bool keepdims) const {
    return array().get()->reduce_next(reducer,
                                      negaxis,
                                      starts,
                                      parents,
                                      outlength,
                                      mask,
                                      keepdims);
  }

  const ContentPtr
  VirtualArray::localindex(int64_t axis, int64_t depth) const {
    return array().get()->localindex(axis, depth);
  }

  const ContentPtr
  VirtualArray::combinations(int64_t n,
                             bool replacement,
                             const util::RecordLookupPtr& recordlookup,
                             const util::Parameters& parameters,
                             int64_t axis,
                             int64_t depth) const {
    return array().get()->combinations(n,
                                       replacement,
                                       recordlookup,
                                       parameters,
                                       axis,
                                       depth);
  }

  const ContentPtr
  VirtualArray::getitem(const Slice& where) const {
    ContentPtr peek = peek_array();
    if (peek.get() != nullptr) {
      return peek.get()->getitem(where);
    }

    if (where.length() == 1) {
      SliceItemPtr head = where.head();

      if (SliceRange* range =
          dynamic_cast<SliceRange*>(head.get())) {
        if (range->step() == 0) {
            throw std::invalid_argument("slice step cannot be zero");
        }
        else if (generator_.get()->length() >= 0) {
          int64_t regular_start = range->start();
          int64_t regular_stop = range->stop();
          awkward_regularize_rangeslice(&regular_start,
                                        &regular_stop,
                                        range->step() > 0,
                                        range->start() != Slice::none(),
                                        range->stop() != Slice::none(),
                                        generator_.get()->length());
          int64_t length;
          if ((range->step() > 0  &&  regular_stop - regular_start > 0)  ||
              (range->step() < 0  &&  regular_stop - regular_start < 0)) {
            int64_t numer = abs(regular_start - regular_stop);
            int64_t denom = abs(range->step());
            int64_t d = numer / denom;
            int64_t m = numer % denom;
            length = d + (m != 0 ? 1 : 0);
          }
          else {
            length = 0;
          }
          FormPtr form(nullptr);
          ArrayGeneratorPtr generator = std::make_shared<SliceGenerator>(
                     form, length, generator_, where);
          ArrayCachePtr cache(nullptr);
          return std::make_shared<VirtualArray>(Identities::none(),
                                                util::Parameters(),
                                                generator,
                                                cache);
        }
        else {
          return array().get()->getitem(where);
        }
      }

      else if (SliceEllipsis* ellipsis =
               dynamic_cast<SliceEllipsis*>(head.get())) {
        FormPtr form(nullptr);
        ArrayGeneratorPtr generator = std::make_shared<SliceGenerator>(
                     form, generator_.get()->length(), generator_, where);
        ArrayCachePtr cache(nullptr);
        return std::make_shared<VirtualArray>(Identities::none(),
                                              util::Parameters(),
                                              generator,
                                              cache);
      }

      else if (SliceNewAxis* newaxis =
               dynamic_cast<SliceNewAxis*>(head.get())) {
        FormPtr form(nullptr);
        ArrayGeneratorPtr generator = std::make_shared<SliceGenerator>(
                     form, 1, generator_, where);
        ArrayCachePtr cache(nullptr);
        return std::make_shared<VirtualArray>(Identities::none(),
                                              util::Parameters(),
                                              generator,
                                              cache);
      }

      else if (SliceArray64* slicearray =
               dynamic_cast<SliceArray64*>(head.get())) {
        FormPtr form(nullptr);
        ArrayGeneratorPtr generator = std::make_shared<SliceGenerator>(
                     form, slicearray->length(), generator_, where);
        ArrayCachePtr cache(nullptr);
        return std::make_shared<VirtualArray>(Identities::none(),
                                              util::Parameters(),
                                              generator,
                                              cache);
      }

      else if (SliceField* field =
               dynamic_cast<SliceField*>(head.get())) {
        return getitem_field(field->key());
      }

      else if (SliceFields* fields =
               dynamic_cast<SliceFields*>(head.get())) {
        return getitem_fields(fields->keys());
      }

      else {
        return array().get()->getitem(where);
      }
    }

    else {
      return array().get()->getitem(where);
    }
  }

  const ContentPtr
  VirtualArray::getitem_next(const SliceAt& at,
                           const Slice& tail,
                           const Index64& advanced) const {
    throw std::runtime_error(
            "undefined operation: VirtualArray::getitem_next(at)");
  }

  const ContentPtr
  VirtualArray::getitem_next(const SliceRange& range,
                           const Slice& tail,
                           const Index64& advanced) const {
    throw std::runtime_error(
            "undefined operation: VirtualArray::getitem_next(range)");
  }

  const ContentPtr
  VirtualArray::getitem_next(const SliceArray64& array,
                           const Slice& tail,
                           const Index64& advanced) const {
    throw std::runtime_error(
            "undefined operation: VirtualArray::getitem_next(array)");
  }

  const ContentPtr
  VirtualArray::getitem_next(const SliceField& field,
                           const Slice& tail,
                           const Index64& advanced) const {
    throw std::runtime_error(
            "undefined operation: VirtualArray::getitem_next(field)");
  }

  const ContentPtr
  VirtualArray::getitem_next(const SliceFields& fields,
                           const Slice& tail,
                           const Index64& advanced) const {
    throw std::runtime_error(
            "undefined operation: VirtualArray::getitem_next(fields)");
  }

  const ContentPtr
  VirtualArray::getitem_next(const SliceJagged64& jagged,
                           const Slice& tail,
                           const Index64& advanced) const {
    throw std::runtime_error(
            "undefined operation: VirtualArray::getitem_next(jagged)");
  }

  const ContentPtr
  VirtualArray::getitem_next_jagged(const Index64& slicestarts,
                                  const Index64& slicestops,
                                  const SliceArray64& slicecontent,
                                  const Slice& tail) const {
    throw std::runtime_error(
            "undefined operation: VirtualArray::getitem_next_jagged(array)");
  }

  const ContentPtr
  VirtualArray::getitem_next_jagged(const Index64& slicestarts,
                                  const Index64& slicestops,
                                  const SliceMissing64& slicecontent,
                                  const Slice& tail) const {
    throw std::runtime_error(
            "undefined operation: VirtualArray::getitem_next_jagged(missing)");
  }

  const ContentPtr
  VirtualArray::getitem_next_jagged(const Index64& slicestarts,
                                  const Index64& slicestops,
                                  const SliceJagged64& slicecontent,
                                  const Slice& tail) const {
    throw std::runtime_error(
            "undefined operation: VirtualArray::getitem_next_jagged(jagged)");
  }

}
