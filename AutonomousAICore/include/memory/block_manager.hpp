// include/memory/block_manager.hpp
// Paged Attention Block Manager & Virtual Memory Page Allocator.
#pragma once

#include <cstdint>
#include <deque>
#include <memory>
#include <unordered_map>
#include <vector>

#include "core/device.hpp"
#include "core/dtype.hpp"
#include "tensor/tensor.hpp"

namespace aic::memory {

constexpr int DEFAULT_BLOCK_SIZE = 16;  // 16 tokens per physical KV block

struct PhysicalBlock {
    int32_t block_id = -1;
    int32_t ref_count = 0;
};

using BlockTable = std::vector<int32_t>;

struct BlockManagerConfig {
    int32_t num_blocks = 256;      // Total pool of physical blocks
    int32_t block_size = DEFAULT_BLOCK_SIZE; // Tokens per block
    int32_t num_heads = 4;
    int32_t head_dim = 16;
    DType dtype = DType::Float32;
    Device device = Device::cpu();
};

class BlockManager {
public:
    explicit BlockManager(BlockManagerConfig config);

    // Allocates blocks for a new sequence of given token count
    BlockTable allocate(int num_tokens);

    // Appends slots for additional tokens, allocating new physical blocks as needed
    void append_slots(BlockTable& table, int current_tokens, int new_tokens = 1);

    // Forks a sequence block table (Copy-on-Write reference counting for MCTS branching)
    BlockTable fork(const BlockTable& parent_table);

    // Frees a sequence block table, decrementing ref counts and recycling unreferenced blocks
    void free(const BlockTable& table);

    // Total and free blocks
    int32_t num_total_blocks() const noexcept { return config_.num_blocks; }
    int32_t num_free_blocks() const noexcept { return static_cast<int32_t>(free_list_.size()); }
    int32_t num_allocated_blocks() const noexcept { return num_total_blocks() - num_free_blocks(); }

    const BlockManagerConfig& config() const noexcept { return config_; }

    // Direct access to physical Key and Value cache storage tensors
    // Shape: [num_blocks, num_heads, block_size, head_dim]
    Tensor& key_cache() noexcept { return key_cache_; }
    const Tensor& key_cache() const noexcept { return key_cache_; }

    Tensor& value_cache() noexcept { return value_cache_; }
    const Tensor& value_cache() const noexcept { return value_cache_; }

    // Gets physical block reference count
    int32_t get_ref_count(int32_t block_id) const;

private:
    BlockManagerConfig config_;
    std::vector<PhysicalBlock> blocks_;
    std::deque<int32_t> free_list_;

    Tensor key_cache_;
    Tensor value_cache_;
};

}  // namespace aic::memory
