// include/ops/paged_attention.hpp
// Paged Attention Kernel Interface for Block-Mapped Memory.
#pragma once

#include <cstdint>
#include <vector>
#include "tensor/tensor.hpp"

namespace aic::ops {

// Executes Paged Attention for decoding tokens against virtual block-mapped KV caches:
// query:        [num_seqs, num_heads, head_dim]
// key_cache:    [num_blocks, num_kv_heads, block_size, head_dim]
// value_cache:  [num_blocks, num_kv_heads, block_size, head_dim]
// block_table:  [num_seqs, max_num_blocks_per_seq] (int32_t)
// context_lens: [num_seqs] (int32_t)
// scale:        Softmax temperature scale factor (default: 1.0f / sqrt(head_dim))
// Returns out:  [num_seqs, num_heads, head_dim]
Tensor paged_attention(
    const Tensor& query,
    const Tensor& key_cache,
    const Tensor& value_cache,
    const Tensor& block_table,
    const Tensor& context_lens,
    float scale = -1.0f
);

// Writes a single token's Key and Value into the physical cache block at slot context_len:
void write_paged_kv_cache(
    Tensor& key_cache,
    Tensor& value_cache,
    const Tensor& key,          // [num_seqs, num_kv_heads, head_dim]
    const Tensor& value,        // [num_seqs, num_kv_heads, head_dim]
    const Tensor& block_table,  // [num_seqs, max_num_blocks_per_seq] (int32_t)
    const Tensor& slot_indices  // [num_seqs] (int32_t slot index to write into)
);

}  // namespace aic::ops
