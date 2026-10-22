// include/data/dataset.hpp
// Memory-mapped binary token dataset with zero-copy sliding window.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "tensor/tensor.hpp"

namespace aic::data {

// RAII platform memory-mapped file abstraction
class MemoryMappedFile {
public:
    MemoryMappedFile() = default;
    explicit MemoryMappedFile(const std::string& path);
    ~MemoryMappedFile();

    MemoryMappedFile(const MemoryMappedFile&) = delete;
    MemoryMappedFile& operator=(const MemoryMappedFile&) = delete;
    MemoryMappedFile(MemoryMappedFile&& other) noexcept;
    MemoryMappedFile& operator=(MemoryMappedFile&& other) noexcept;

    const void* data() const noexcept { return data_; }
    size_t size_bytes() const noexcept { return size_bytes_; }
    bool is_open() const noexcept { return data_ != nullptr; }

private:
    void close();

    void* file_handle_ = nullptr;
    void* mapping_handle_ = nullptr;
    const void* data_ = nullptr;
    size_t size_bytes_ = 0;
};

// Binary Token Dataset with autoregressive next-token alignment
class BinaryTokenDataset {
public:
    // Construct from an on-disk binary file of int64_t tokens
    BinaryTokenDataset(const std::string& filepath, int64_t seq_len, int64_t stride = 0);

    // Construct from an in-memory vector of tokens (e.g. for synthetic data / tests)
    BinaryTokenDataset(std::vector<int64_t> in_memory_tokens, int64_t seq_len, int64_t stride = 0);

    int64_t size() const noexcept { return num_samples_; }
    int64_t seq_len() const noexcept { return seq_len_; }
    int64_t total_tokens() const noexcept { return total_tokens_; }

    // Returns a pair of (input_seq, target_seq) where target_seq is input_seq shifted by 1 token
    std::pair<Tensor, Tensor> get_item(int64_t idx) const;

    // Direct token query
    int64_t token_at(int64_t index) const;

    // Helper: writes an in-memory token sequence to disk in native binary format
    static void write_binary_file(const std::string& filepath, const std::vector<int64_t>& tokens);

private:
    void compute_samples();

    std::shared_ptr<MemoryMappedFile> mmap_file_;
    std::vector<int64_t> in_memory_tokens_;
    const int64_t* tokens_ptr_ = nullptr;
    int64_t total_tokens_ = 0;
    int64_t seq_len_ = 0;
    int64_t stride_ = 0;
    int64_t num_samples_ = 0;
};

}  // namespace aic::data
