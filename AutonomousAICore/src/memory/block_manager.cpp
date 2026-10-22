// src/memory/block_manager.cpp
// Paged Attention Block Manager & Virtual Memory Page Allocator implementation.
#include "memory/block_manager.hpp"
#include "core/error.hpp"

namespace aic::memory {

BlockManager::BlockManager(BlockManagerConfig config)
    : config_(config) {
    AIC_CHECK(config_.num_blocks > 0, InvalidArgument, "num_blocks must be positive");
    AIC_CHECK(config_.block_size > 0, InvalidArgument, "block_size must be positive");
    AIC_CHECK(config_.num_heads > 0, InvalidArgument, "num_heads must be positive");
    AIC_CHECK(config_.head_dim > 0, InvalidArgument, "head_dim must be positive");

    blocks_.resize(static_cast<size_t>(config_.num_blocks));
    for (int32_t i = 0; i < config_.num_blocks; ++i) {
        blocks_[static_cast<size_t>(i)].block_id = i;
        blocks_[static_cast<size_t>(i)].ref_count = 0;
        free_list_.push_back(i);
    }

    // Allocate physical Key and Value cache tensors
    Shape cache_shape({
        config_.num_blocks,
        config_.num_heads,
        config_.block_size,
        config_.head_dim
    });

    key_cache_ = Tensor::zeros(cache_shape, config_.dtype, config_.device);
    value_cache_ = Tensor::zeros(cache_shape, config_.dtype, config_.device);
}

BlockTable BlockManager::allocate(int num_tokens) {
    int needed = (num_tokens > 0)
        ? ((num_tokens + config_.block_size - 1) / config_.block_size)
        : 1;

    AIC_CHECK(static_cast<int>(free_list_.size()) >= needed, OutOfMemory,
              "BlockManager: Out of physical KV cache blocks. Requested: ", needed,
              ", Free: ", free_list_.size());

    BlockTable table;
    table.reserve(static_cast<size_t>(needed));

    for (int i = 0; i < needed; ++i) {
        int32_t bid = free_list_.front();
        free_list_.pop_front();
        blocks_[static_cast<size_t>(bid)].ref_count = 1;
        table.push_back(bid);
    }

    return table;
}

void BlockManager::append_slots(BlockTable& table, int current_tokens, int new_tokens) {
    int current_capacity = static_cast<int>(table.size()) * config_.block_size;
    int target_tokens = current_tokens + new_tokens;

    if (target_tokens <= current_capacity) {
        return;
    }

    int additional_needed = (target_tokens - current_capacity + config_.block_size - 1) / config_.block_size;
    AIC_CHECK(static_cast<int>(free_list_.size()) >= additional_needed, OutOfMemory,
              "BlockManager: Cannot append slots. Free blocks: ", free_list_.size(),
              ", Needed: ", additional_needed);

    for (int i = 0; i < additional_needed; ++i) {
        int32_t bid = free_list_.front();
        free_list_.pop_front();
        blocks_[static_cast<size_t>(bid)].ref_count = 1;
        table.push_back(bid);
    }
}

BlockTable BlockManager::fork(const BlockTable& parent_table) {
    BlockTable child_table = parent_table;
    for (int32_t bid : child_table) {
        AIC_CHECK(bid >= 0 && bid < config_.num_blocks, OutOfRange, "Invalid block id: ", bid);
        blocks_[static_cast<size_t>(bid)].ref_count++;
    }
    return child_table;
}

void BlockManager::free(const BlockTable& table) {
    for (int32_t bid : table) {
        if (bid < 0 || bid >= config_.num_blocks) continue;

        auto& blk = blocks_[static_cast<size_t>(bid)];
        if (blk.ref_count > 0) {
            blk.ref_count--;
            if (blk.ref_count == 0) {
                free_list_.push_back(bid);
            }
        }
    }
}

int32_t BlockManager::get_ref_count(int32_t block_id) const {
    AIC_CHECK(block_id >= 0 && block_id < config_.num_blocks, OutOfRange, "Invalid block id: ", block_id);
    return blocks_[static_cast<size_t>(block_id)].ref_count;
}

}  // namespace aic::memory
