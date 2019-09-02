// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_LISTOFFSETARRAYCONTENT_H_
#define AWKWARD_LISTOFFSETARRAYCONTENT_H_

#include <sstream>
#include <memory>
#include <type_traits>

#include "awkward/cpu-kernels/util.h"
#include "awkward/util.h"
#include "awkward/Index.h"
#include "awkward/Content.h"

namespace awkward {
  template <typename T>
  class ListOffsetArrayOf: public Content {
  public:
    ListOffsetArrayOf<T>(const std::shared_ptr<Identity> id, const IndexOf<T> offsets, const std::shared_ptr<Content> content)
        : id_(id)
        , offsets_(offsets)
        , content_(content) { }

    const IndexOf<T> offsets() const { return offsets_; }
    const std::shared_ptr<Content> content() const { return content_.get()->shallow_copy(); }

    virtual const std::shared_ptr<Identity> id() const { return id_; }
    virtual void setid();
    virtual void setid(const std::shared_ptr<Identity> id);
    virtual const std::string repr(const std::string indent, const std::string pre, const std::string post) const;
    virtual int64_t length() const;
    virtual std::shared_ptr<Content> shallow_copy() const;
    virtual std::shared_ptr<Content> get(int64_t at) const;
    virtual std::shared_ptr<Content> slice(int64_t start, int64_t stop) const;

  private:
    std::shared_ptr<Identity> id_;
    const IndexOf<T> offsets_;
    const std::shared_ptr<Content> content_;
  };

  typedef ListOffsetArrayOf<int32_t> ListOffsetArray32;
  typedef ListOffsetArrayOf<int64_t> ListOffsetArray64;
}

#endif // AWKWARD_LISTOFFSETARRAYCONTENT_H_
