// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_REFACTORING_H_
#define AWKWARD_REFACTORING_H_

// FIXME: refactor
#include "awkward/builder/OptionBuilder.h"
#include "awkward/builder/BoolBuilder.h"
#include "awkward/builder/DatetimeBuilder.h"
#include "awkward/builder/Int64Builder.h"
#include "awkward/builder/Float64Builder.h"
#include "awkward/builder/StringBuilder.h"
#include "awkward/builder/ListBuilder.h"
#include "awkward/builder/TupleBuilder.h"
#include "awkward/builder/RecordBuilder.h"
#include "awkward/builder/IndexedBuilder.h"
#include "awkward/builder/Complex128Builder.h"
#include "awkward/builder/UnionBuilder.h"
#include "awkward/builder/UnknownBuilder.h"
#include "awkward/array/EmptyArray.h"
#include "awkward/array/IndexedArray.h"
#include "awkward/array/ByteMaskedArray.h"
#include "awkward/array/BitMaskedArray.h"
#include "awkward/array/UnmaskedArray.h"
#include "awkward/array/ListArray.h"
#include "awkward/array/ListOffsetArray.h"
#include "awkward/array/None.h"
#include "awkward/array/NumpyArray.h"
#include "awkward/array/Record.h"
#include "awkward/array/RecordArray.h"
#include "awkward/array/RegularArray.h"
#include "awkward/array/UnionArray.h"
#include "awkward/datetime_util.h"

namespace ak = awkward;
namespace {

  const ak::ContentPtr
  builder_snapshot(const ak::Builder& builder) {
    ak::ContentPtr out;
    if (builder.classname() == "BoolBuilder") {
      const ak::BoolBuilder& raw = dynamic_cast<const ak::BoolBuilder&>(builder);
      std::vector<ssize_t> shape = { (ssize_t)raw.buffer().length() };
      std::vector<ssize_t> strides = { (ssize_t)sizeof(bool) };
      out = std::make_shared<ak::NumpyArray>(ak::Identities::none(),
                                             ak::util::Parameters(),
                                             raw.buffer().ptr(),
                                             shape,
                                             strides,
                                             0,
                                             sizeof(bool),
                                             "?",
                                             ak::util::dtype::boolean,
                                             ak::kernel::lib::cpu);

    }
    else if (builder.classname() == "Complex128Builder") {
      const ak::Complex128Builder& raw = dynamic_cast<const ak::Complex128Builder&>(builder);
      std::vector<ssize_t> shape = { (ssize_t)raw.buffer().length() };
      std::vector<ssize_t> strides = { (ssize_t)sizeof(std::complex<double>) };
      out = std::make_shared<ak::NumpyArray>(ak::Identities::none(),
                                             ak::util::Parameters(),
                                             raw.buffer().ptr(),
                                             shape,
                                             strides,
                                             0,
                                             sizeof(std::complex<double>),
                                             "Zd",
                                             ak::util::dtype::complex128,
                                             ak::kernel::lib::cpu);

    }
    else if (builder.classname() == "DatetimeBuilder") {
      const ak::DatetimeBuilder& raw = dynamic_cast<const ak::DatetimeBuilder&>(builder);
      std::vector<ssize_t> shape = { (ssize_t)raw.buffer().length() };
      std::vector<ssize_t> strides = { (ssize_t)sizeof(int64_t) };

      auto dtype = ak::util::name_to_dtype(raw.unit());
      auto format = std::string(ak::util::dtype_to_format(dtype))
        .append(std::to_string(ak::util::dtype_to_itemsize(dtype)))
        .append(ak::util::format_to_units(raw.unit()));
      out = std::make_shared<ak::NumpyArray>(
               ak::Identities::none(),
               ak::util::Parameters(),
               raw.buffer().ptr(),
               shape,
               strides,
               0,
               sizeof(int64_t),
               format,
               dtype,
               ak::kernel::lib::cpu);

    }
    else if (builder.classname() == "Float64Builder") {
      const ak::Float64Builder& raw = dynamic_cast<const ak::Float64Builder&>(builder);
      std::vector<ssize_t> shape = { (ssize_t)raw.buffer().length() };
      std::vector<ssize_t> strides = { (ssize_t)sizeof(double) };
      out = std::make_shared<ak::NumpyArray>(ak::Identities::none(),
                                             ak::util::Parameters(),
                                             raw.buffer().ptr(),
                                             shape,
                                             strides,
                                             0,
                                             sizeof(double),
                                             "d",
                                             ak::util::dtype::float64,
                                             ak::kernel::lib::cpu);
    }
    else if (builder.classname() == "IndexedGenericBuilder") {
      const ak::IndexedGenericBuilder& raw = dynamic_cast<const ak::IndexedGenericBuilder&>(builder);
      ak::Index64 index(raw.buffer().ptr(), 0, raw.buffer().length(), ak::kernel::lib::cpu);
      if (raw.hasnull()) {
        return std::make_shared<ak::IndexedOptionArray64>(
          ak::Identities::none(),
          ak::util::Parameters(),
          index,
          raw.array());
      }
      else {
        return std::make_shared<ak::IndexedArray64>(
          ak::Identities::none(),
          ak::util::Parameters(),
          index,
          raw.array());
      }
    }
    else if (builder.classname() == "IndexedI32Builder") {
      const ak::IndexedI32Builder& raw = dynamic_cast<const ak::IndexedI32Builder&>(builder);
      ak::Index64 index(raw.buffer().ptr(), 0, raw.buffer().length(), ak::kernel::lib::cpu);
      if (raw.hasnull()) {
        return std::make_shared<ak::IndexedOptionArray64>(
          ak::Identities::none(),
          raw.array().get()->content().get()->parameters(),
          index,
          raw.array().get()->content());
      }
      else {
        return std::make_shared<ak::IndexedArray64>(
          ak::Identities::none(),
          raw.array().get()->content().get()->parameters(),
          index,
          raw.array().get()->content());
      }
    }
    else if (builder.classname() == "IndexedIU32Builder") {
      const ak::IndexedIU32Builder& raw = dynamic_cast<const ak::IndexedIU32Builder&>(builder);
      ak::Index64 index(raw.buffer().ptr(), 0, raw.buffer().length(), ak::kernel::lib::cpu);
      if (raw.hasnull()) {
        return std::make_shared<ak::IndexedOptionArray64>(
            ak::Identities::none(),
            raw.array().get()->content().get()->parameters(),
            index,
            raw.array().get()->content());
      }
      else {
        return std::make_shared<ak::IndexedArray64>(
          ak::Identities::none(),
            raw.array().get()->content().get()->parameters(),
            index,
            raw.array().get()->content());
      }
    }
    else if (builder.classname() == "IndexedI64Builder") {
      const ak::IndexedI64Builder& raw = dynamic_cast<const ak::IndexedI64Builder&>(builder);
        ak::Index64 index(raw.buffer().ptr(), 0, raw.buffer().length(), ak::kernel::lib::cpu);
        if (raw.hasnull()) {
          return std::make_shared<ak::IndexedOptionArray64>(
            ak::Identities::none(),
            raw.array().get()->content().get()->parameters(),
            index,
            raw.array().get()->content());
        }
        else {
          return std::make_shared<ak::IndexedArray64>(
            ak::Identities::none(),
            raw.array().get()->content().get()->parameters(),
            index,
            raw.array().get()->content());
        }
      }
      else if (builder.classname() == "IndexedIO32Builder") {
        const ak::IndexedI64Builder& raw = dynamic_cast<const ak::IndexedI64Builder&>(builder);
        ak::Index64 index(raw.buffer().ptr(), 0, raw.buffer().length(), ak::kernel::lib::cpu);
        return std::make_shared<ak::IndexedOptionArray64>(
          ak::Identities::none(),
          raw.array().get()->content().get()->parameters(),
          index,
          raw.array().get()->content());
      }
      else if (builder.classname() == "IndexedIO64Builder") {
        const ak::IndexedIO64Builder& raw = dynamic_cast<const ak::IndexedIO64Builder&>(builder);
        ak::Index64 index(raw.buffer().ptr(), 0, raw.buffer().length(), ak::kernel::lib::cpu);
        return std::make_shared<ak::IndexedOptionArray64>(
          ak::Identities::none(),
          raw.array().get()->content().get()->parameters(),
          index,
          raw.array().get()->content());
      }
      else if (builder.classname() == "Int64Builder") {
        const ak::Int64Builder& raw = dynamic_cast<const ak::Int64Builder&>(builder);
        std::vector<ssize_t> shape = { (ssize_t)raw.buffer().length() };
        std::vector<ssize_t> strides = { (ssize_t)sizeof(int64_t) };
        return std::make_shared<ak::NumpyArray>(
                 ak::Identities::none(),
                 ak::util::Parameters(),
                 raw.buffer().ptr(),
                 shape,
                 strides,
                 0,
                 sizeof(int64_t),
                 ak::util::dtype_to_format(ak::util::dtype::int64),
                 ak::util::dtype::int64,
                 ak::kernel::lib::cpu);
      }
      else if (builder.classname() == "ListBuilder") {
        const ak::ListBuilder& raw = dynamic_cast<const ak::ListBuilder&>(builder);
        ak::Index64 offsets(raw.buffer().ptr(), 0, raw.buffer().length(), ak::kernel::lib::cpu);
        return std::make_shared<ak::ListOffsetArray64>(ak::Identities::none(),
                                                       ak::util::Parameters(),
                                                       offsets,
                                                       builder_snapshot(raw.builder()));
      }
      else if (builder.classname() == "OptionBuilder") {
        const ak::OptionBuilder& raw = dynamic_cast<const ak::OptionBuilder&>(builder);
        ak::Index64 index(raw.buffer().ptr(), 0, raw.buffer().length(), ak::kernel::lib::cpu);
        return ak::IndexedOptionArray64(ak::Identities::none(),
                                        ak::util::Parameters(),
                                        index,
                                        builder_snapshot(raw.builder())).simplify_optiontype();
      }
      else if (builder.classname() == "RecordBuilder") {
        const ak::RecordBuilder& raw = dynamic_cast<const ak::RecordBuilder&>(builder);
        if (raw.length() == -1) {
          return std::make_shared<ak::EmptyArray>(ak::Identities::none(),
                                                  ak::util::Parameters());
        }
        ak::util::Parameters parameters;
        if (raw.nameptr() != nullptr) {
          parameters["__record__"] = ak::util::quote(raw.name());
        }
        ak::ContentPtrVec contents;
        ak::util::RecordLookupPtr recordlookup =
          std::make_shared<ak::util::RecordLookup>();
        for (size_t i = 0;  i < raw.builders().size();  i++) {
          contents.push_back(builder_snapshot(*raw.builders()[i]));
          recordlookup.get()->push_back(raw.keys()[i]);
        }
        std::vector<ak::ArrayCachePtr> caches;  // nothing is virtual here
        return std::make_shared<ak::RecordArray>(ak::Identities::none(),
                                             parameters,
                                             contents,
                                             recordlookup,
                                             raw.length(),
                                             caches);
      }
      else if (builder.classname() == "StringBuilder") {
        const ak::StringBuilder& raw = dynamic_cast<const ak::StringBuilder&>(builder);
        ak::util::Parameters char_parameters;
        ak::util::Parameters string_parameters;

        if (raw.encoding() == nullptr) {
          char_parameters["__array__"] = std::string("\"byte\"");
          string_parameters["__array__"] = std::string("\"bytestring\"");
        }
        else if (std::string(raw.encoding()) == std::string("utf-8")) {
          char_parameters["__array__"] = std::string("\"char\"");
          string_parameters["__array__"] = std::string("\"string\"");
        }
        else {
          throw std::invalid_argument(
            std::string("unsupported encoding: ") + ak::util::quote(raw.encoding()));
        }

        ak::Index64 offsets(raw.buffer().ptr(), 0, raw.buffer().length(), ak::kernel::lib::cpu);
        std::vector<ssize_t> shape = { (ssize_t)raw.content().length() };
        std::vector<ssize_t> strides = { (ssize_t)sizeof(uint8_t) };
        ak::ContentPtr content;
        content = std::make_shared<ak::NumpyArray>(ak::Identities::none(),
                                               char_parameters,
                                               raw.content().ptr(),
                                               shape,
                                               strides,
                                               0,
                                               sizeof(uint8_t),
                                               "B",
                                               ak::util::dtype::uint8,
                                               ak::kernel::lib::cpu);
        return std::make_shared<ak::ListOffsetArray64>(ak::Identities::none(),
                                                   string_parameters,
                                                   offsets,
                                                   content);
      }
      else if (builder.classname() == "TupleBuilder") {
        const ak::TupleBuilder& raw = dynamic_cast<const ak::TupleBuilder&>(builder);
        if (raw.length() == -1) {
          return std::make_shared<ak::EmptyArray>(ak::Identities::none(),
                                                  ak::util::Parameters());
        }
        ak::ContentPtrVec contents;
        for (size_t i = 0;  i < raw.contents().size();  i++) {
          contents.push_back(builder_snapshot(*raw.contents()[i]));
        }
        std::vector<ak::ArrayCachePtr> caches;  // nothing is virtual here
        return std::make_shared<ak::RecordArray>(ak::Identities::none(),
                                                 ak::util::Parameters(),
                                                 contents,
                                                 ak::util::RecordLookupPtr(nullptr),
                                                 raw.length(),
                                                 caches);
      }
      else if (builder.classname() == "UnionBuilder") {
        const ak::UnionBuilder& raw = dynamic_cast<const ak::UnionBuilder&>(builder);
        ak::Index8 tags(raw.tags().ptr(), 0, raw.tags().length(), ak::kernel::lib::cpu);
        ak::Index64 index(raw.index().ptr(), 0, raw.index().length(), ak::kernel::lib::cpu);
        ak::ContentPtrVec contents;
        for (auto content : raw.contents()) {
          contents.push_back(builder_snapshot(*content.get()));
        }
        return ak::UnionArray8_64(ak::Identities::none(),
                                  ak::util::Parameters(),
                                  tags,
                                  index,
                                  contents).simplify_uniontype(true, false);

      }
      else if (builder.classname() == "UnknownBuilder") {
        const ak::UnknownBuilder& raw = dynamic_cast<const ak::UnknownBuilder&>(builder);
        if (raw.nullcount() == 0) {
          return std::make_shared<ak::EmptyArray>(ak::Identities::none(),
                                                  ak::util::Parameters());
        }
        else {
          // This is the only snapshot that is O(N), rather than O(1),
          // but it is a corner case (array of only Nones).
          ak::Index64 index(raw.nullcount());
          int64_t* rawptr = index.ptr().get();
          for (int64_t i = 0;  i < raw.nullcount();  i++) {
            rawptr[i] = -1;
          }
          return std::make_shared<ak::IndexedOptionArray64>(
            ak::Identities::none(),
            ak::util::Parameters(),
            index,
            std::make_shared<ak::EmptyArray>(ak::Identities::none(), ak::util::Parameters()));
      }
    }
    return out;
  }
}

#endif // AWKWARD_REFACTORING_H_
