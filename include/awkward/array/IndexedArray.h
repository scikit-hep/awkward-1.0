// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_INDEXEDARRAY_H_
#define AWKWARD_INDEXEDARRAY_H_

#include <string>
#include <memory>
#include <vector>

#include "awkward/cpu-kernels/util.h"
#include "awkward/Slice.h"
#include "awkward/Index.h"
#include "awkward/Content.h"

namespace awkward {
  template <typename T, bool ISOPTION>
  class EXPORT_SYMBOL IndexedArrayOf: public Content {
  public:
    IndexedArrayOf<T, ISOPTION>(const std::shared_ptr<Identities>& identities, const util::Parameters& parameters, const IndexOf<T>& index, ContentPtr& content);
    const IndexOf<T> index() const;
    ContentPtr content() const;
    bool isoption() const;
    ContentPtr project() const;
    ContentPtr project(const Index8& mask) const;
    const Index8 bytemask() const;
    ContentPtr simplify_optiontype() const;
    T index_at_nowrap(int64_t at) const;

    const std::string classname() const override;
    void setidentities() override;
    void setidentities(const std::shared_ptr<Identities>& identities) override;
    const std::shared_ptr<Type> type(const std::map<std::string, std::string>& typestrs) const override;
    const std::string tostring_part(const std::string& indent, const std::string& pre, const std::string& post) const override;
    void tojson_part(ToJson& builder) const override;
    void nbytes_part(std::map<size_t, int64_t>& largest) const override;
    int64_t length() const override;
    ContentPtr shallow_copy() const override;
    ContentPtr deep_copy(bool copyarrays, bool copyindexes, bool copyidentities) const override;
    void check_for_iteration() const override;
    ContentPtr getitem_nothing() const override;
    ContentPtr getitem_at(int64_t at) const override;
    ContentPtr getitem_at_nowrap(int64_t at) const override;
    ContentPtr getitem_range(int64_t start, int64_t stop) const override;
    ContentPtr getitem_range_nowrap(int64_t start, int64_t stop) const override;
    ContentPtr getitem_field(const std::string& key) const override;
    ContentPtr getitem_fields(const std::vector<std::string>& keys) const override;
    ContentPtr getitem_next(const std::shared_ptr<SliceItem>& head, const Slice& tail, const Index64& advanced) const override;
    ContentPtr carry(const Index64& carry) const override;
    const std::string purelist_parameter(const std::string& key) const override;
    bool purelist_isregular() const override;
    int64_t purelist_depth() const override;
    const std::pair<int64_t, int64_t> minmax_depth() const override;
    const std::pair<bool, int64_t> branch_depth() const override;
    int64_t numfields() const override;
    int64_t fieldindex(const std::string& key) const override;
    const std::string key(int64_t fieldindex) const override;
    bool haskey(const std::string& key) const override;
    const std::vector<std::string> keys() const override;

    // operations
    const std::string validityerror(const std::string& path) const override;
    ContentPtr shallow_simplify() const override;
    ContentPtr num(int64_t axis, int64_t depth) const override;
    const std::pair<Index64, std::shared_ptr<Content>> offsets_and_flattened(int64_t axis, int64_t depth) const override;
    bool mergeable(ContentPtr& other, bool mergebool) const override;
    ContentPtr reverse_merge(ContentPtr& other) const;
    ContentPtr merge(ContentPtr& other) const override;
    const std::shared_ptr<SliceItem> asslice() const override;
    ContentPtr fillna(ContentPtr& value) const override;
    ContentPtr rpad(int64_t length, int64_t axis, int64_t depth) const override;
    ContentPtr rpad_and_clip(int64_t length, int64_t axis, int64_t depth) const override;
    ContentPtr reduce_next(const Reducer& reducer, int64_t negaxis, const Index64& starts, const Index64& parents, int64_t outlength, bool mask, bool keepdims) const override;
    ContentPtr localindex(int64_t axis, int64_t depth) const override;
    ContentPtr choose(int64_t n, bool diagonal, const std::shared_ptr<util::RecordLookup>& recordlookup, const util::Parameters& parameters, int64_t axis, int64_t depth) const override;

    ContentPtr getitem_next(const SliceAt& at, const Slice& tail, const Index64& advanced) const override;
    ContentPtr getitem_next(const SliceRange& range, const Slice& tail, const Index64& advanced) const override;
    ContentPtr getitem_next(const SliceArray64& array, const Slice& tail, const Index64& advanced) const override;
    ContentPtr getitem_next(const SliceJagged64& jagged, const Slice& tail, const Index64& advanced) const override;
    ContentPtr getitem_next_jagged(const Index64& slicestarts, const Index64& slicestops, const SliceArray64& slicecontent, const Slice& tail) const override;
    ContentPtr getitem_next_jagged(const Index64& slicestarts, const Index64& slicestops, const SliceMissing64& slicecontent, const Slice& tail) const override;
    ContentPtr getitem_next_jagged(const Index64& slicestarts, const Index64& slicestops, const SliceJagged64& slicecontent, const Slice& tail) const override;

  protected:
    template <typename S>
    ContentPtr getitem_next_jagged_generic(const Index64& slicestarts, const Index64& slicestops, const S& slicecontent, const Slice& tail) const;

    const std::pair<Index64, IndexOf<T>> nextcarry_outindex(int64_t& numnull) const;

  private:
    const IndexOf<T> index_;
    ContentPtr content_;
  };

  typedef IndexedArrayOf<int32_t, false>  IndexedArray32;
  typedef IndexedArrayOf<uint32_t, false> IndexedArrayU32;
  typedef IndexedArrayOf<int64_t, false>  IndexedArray64;
  typedef IndexedArrayOf<int32_t, true>   IndexedOptionArray32;
  typedef IndexedArrayOf<int64_t, true>   IndexedOptionArray64;
}

#endif // AWKWARD_INDEXEDARRAY_H_
