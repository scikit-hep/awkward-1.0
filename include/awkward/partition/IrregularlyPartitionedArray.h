// BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

#ifndef AWKWARD_IRREGULARLYPARTITIONEDARRAY_H_
#define AWKWARD_IRREGULARLYPARTITIONEDARRAY_H_

#include "awkward/partition/PartitionedArray.h"

namespace awkward {
  /// @class IrregularlyPartitionedArray
  ///
  /// @brief PartitionedArray of arbitrary length partitions.
  class EXPORT_SYMBOL IrregularlyPartitionedArray {
  public:
    IrregularlyPartitionedArray(const ContentPtrVec& partitions,
                                const std::vector<int64_t> stops);

    /// @brief Logical index where each partition ends.
    ///
    /// An IrregularlyPartitionedArray could be described in terms of
    /// {@link ListOffsetArrayOf#offsets ListOffsetArray::offsets},
    /// but the first entry would always be zero. Hence,
    /// {@link ListArrayOf#stops ListArray::stops} is sufficient.
    const std::vector<int64_t> stops() const;

    void
      partitionid_index_at(int64_t at,
                           int64_t& partitionid,
                           int64_t& index) const override;

    const std::string
      classname() const override;

    int64_t
      length() const override;

    const PartitionedArrayPtr
      shallow_copy() const override;

  private:
    const std::vector<int64_t> stops_;
  };
}

#endif // AWKWARD_IRREGULARLYPARTITIONEDARRAY_H_
