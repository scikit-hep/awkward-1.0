// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_REGULARARRAY_H_
#define AWKWARD_REGULARARRAY_H_

#include <cassert>
#include <string>
#include <memory>
#include <vector>

#include "awkward/cpu-kernels/util.h"
#include "awkward/Slice.h"
#include "awkward/Content.h"

namespace awkward {
  class RegularArray: public Content {
  public:
    RegularArray(const std::shared_ptr<Identity> id, const std::shared_ptr<Content> content, int64_t size)
        : id_(id)
        , content_(content)
        , size_(size) { }

    const std::shared_ptr<Content> content() const { return content_; }
    int64_t size() const { return size_; }

    virtual const std::string classname() const;
    virtual const std::shared_ptr<Identity> id() const { return id_; }
    virtual void setid();
    virtual void setid(const std::shared_ptr<Identity> id);
    virtual const std::string tostring_part(const std::string indent, const std::string pre, const std::string post) const;
    virtual void tojson_part(ToJson& builder) const;
    virtual const std::shared_ptr<Type> type_part() const;
    virtual int64_t length() const;
    virtual const std::shared_ptr<Content> shallow_copy() const;
    virtual void check_for_iteration() const;
    virtual const std::shared_ptr<Content> getitem_at(int64_t at) const;
    virtual const std::shared_ptr<Content> getitem_at_nowrap(int64_t at) const;
    virtual const std::shared_ptr<Content> getitem_range(int64_t start, int64_t stop) const;
    virtual const std::shared_ptr<Content> getitem_range_nowrap(int64_t start, int64_t stop) const;
    virtual const std::shared_ptr<Content> getitem(const Slice& where) const;
    virtual const std::shared_ptr<Content> getitem_next(const std::shared_ptr<SliceItem> head, const Slice& tail, const Index64& advanced) const;
    virtual const std::shared_ptr<Content> carry(const Index64& carry) const;
    virtual const std::pair<int64_t, int64_t> minmax_depth() const;

  private:
    std::shared_ptr<Identity> id_;
    const std::shared_ptr<Content> content_;
    int64_t size_;
  };
}

#endif // AWKWARD_REGULARARRAY_H_
