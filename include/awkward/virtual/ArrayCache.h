// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#ifndef AWKWARD_ARRAYCACHE_H_
#define AWKWARD_ARRAYCACHE_H_

#include "awkward/Content.h"

namespace awkward {
  /// @class ArrayCache
  ///
  /// @brief Abstract superclass of cache for VirtualArray, defining
  /// the interface.
  ///
  /// The main implementation, PyArrayCache, is passed through pybind11 to
  /// Python to work with cachetools and MutableMapping, but in principle, pure
  /// C++ caches could be written.
  class LIBAWKWARD_EXPORT_SYMBOL ArrayCache {
  public:
    /// @brief Returns a new key that is globally unique in the current
    /// process.
    ///
    /// If process-independent keys are needed, they can be bound to
    /// VirtualArrays by explicitly setting the
    /// {@link VirtualArray#cache_key VirtualArray::cache_key}.
    static const std::string
      newkey();

    /// @brief Attempts to get an array; may be `nullptr` if not available.
    virtual ContentPtr
      get(const std::string& key) const = 0;

    /// @brief Writes or overwrites an array at `key`.
    virtual void
      set(const std::string& key, const ContentPtr& value) = 0;

    /// @brief Returns true if the cache cannot be used for any reason.
    virtual bool
      is_broken() const = 0;

    virtual const std::string
      tostring_part(const std::string& indent,
                    const std::string& pre,
                    const std::string& post) const = 0;
  };

  using ArrayCachePtr = std::shared_ptr<ArrayCache>;

  // Note: if you're creating a pure C++ cache (and it's not ridiculously
  // large), define it in this file and implement it in
  // src/libawkward/virtual/ArrayCache.cpp.

}

#endif // AWKWARD_ARRAYCACHE_H_
