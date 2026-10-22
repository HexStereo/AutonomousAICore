// src/data/dataset.cpp
// Memory-mapped binary token dataset implementation.
#include "data/dataset.hpp"
#include "core/error.hpp"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <fstream>
#include <algorithm>
#include <cstring>

namespace aic::data {

// ============================================================================
// MemoryMappedFile Implementation
// ============================================================================
MemoryMappedFile::MemoryMappedFile(const std::string& path) {
#ifdef _WIN32
    HANDLE hFile = CreateFileA(
        path.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        NULL
    );
    AIC_CHECK(hFile != INVALID_HANDLE_VALUE, IOError, "Failed to open file for mmap: ", path);
    file_handle_ = hFile;

    LARGE_INTEGER liSize;
    if (!GetFileSizeEx(hFile, &liSize)) {
        CloseHandle(hFile);
        file_handle_ = nullptr;
        AIC_THROW(IOError, "Failed to query file size: ", path);
    }
    size_bytes_ = static_cast<size_t>(liSize.QuadPart);

    if (size_bytes_ == 0) {
        // Zero-byte file is valid, but nothing to map
        return;
    }

    HANDLE hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!hMap) {
        CloseHandle(hFile);
        file_handle_ = nullptr;
        AIC_THROW(IOError, "Failed to create file mapping object: ", path);
    }
    mapping_handle_ = hMap;

    void* ptr = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
    if (!ptr) {
        CloseHandle(hMap);
        CloseHandle(hFile);
        mapping_handle_ = nullptr;
        file_handle_ = nullptr;
        AIC_THROW(IOError, "Failed to map view of file: ", path);
    }
    data_ = ptr;
#else
    int fd = open(path.c_str(), O_RDONLY);
    AIC_CHECK(fd != -1, IOError, "Failed to open file for mmap: ", path);

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        close(fd);
        AIC_THROW(IOError, "Failed to stat file: ", path);
    }
    size_bytes_ = static_cast<size_t>(sb.st_size);

    if (size_bytes_ > 0) {
        void* ptr = mmap(nullptr, size_bytes_, PROT_READ, MAP_SHARED, fd, 0);
        if (ptr == MAP_FAILED) {
            close(fd);
            AIC_THROW(IOError, "mmap failed for: ", path);
        }
        data_ = ptr;
    }
    file_handle_ = reinterpret_cast<void*>(static_cast<intptr_t>(fd));
#endif
}

MemoryMappedFile::~MemoryMappedFile() {
    close();
}

void MemoryMappedFile::close() {
#ifdef _WIN32
    if (data_) {
        UnmapViewOfFile(data_);
        data_ = nullptr;
    }
    if (mapping_handle_) {
        CloseHandle(static_cast<HANDLE>(mapping_handle_));
        mapping_handle_ = nullptr;
    }
    if (file_handle_ && file_handle_ != INVALID_HANDLE_VALUE) {
        CloseHandle(static_cast<HANDLE>(file_handle_));
        file_handle_ = nullptr;
    }
#else
    if (data_) {
        munmap(const_cast<void*>(data_), size_bytes_);
        data_ = nullptr;
    }
    if (file_handle_) {
        int fd = static_cast<int>(reinterpret_cast<intptr_t>(file_handle_));
        ::close(fd);
        file_handle_ = nullptr;
    }
#endif
    size_bytes_ = 0;
}

MemoryMappedFile::MemoryMappedFile(MemoryMappedFile&& other) noexcept
    : file_handle_(other.file_handle_),
      mapping_handle_(other.mapping_handle_),
      data_(other.data_),
      size_bytes_(other.size_bytes_) {
    other.file_handle_ = nullptr;
    other.mapping_handle_ = nullptr;
    other.data_ = nullptr;
    other.size_bytes_ = 0;
}

MemoryMappedFile& MemoryMappedFile::operator=(MemoryMappedFile&& other) noexcept {
    if (this != &other) {
        close();
        file_handle_ = other.file_handle_;
        mapping_handle_ = other.mapping_handle_;
        data_ = other.data_;
        size_bytes_ = other.size_bytes_;
        other.file_handle_ = nullptr;
        other.mapping_handle_ = nullptr;
        other.data_ = nullptr;
        other.size_bytes_ = 0;
    }
    return *this;
}

// ============================================================================
// BinaryTokenDataset Implementation
// ============================================================================
BinaryTokenDataset::BinaryTokenDataset(const std::string& filepath, int64_t seq_len, int64_t stride)
    : seq_len_(seq_len), stride_(stride) {
    AIC_CHECK(seq_len_ > 0, InvalidArgument, "seq_len must be positive, got ", seq_len_);
    mmap_file_ = std::make_shared<MemoryMappedFile>(filepath);

    AIC_CHECK(mmap_file_->size_bytes() % sizeof(int64_t) == 0, InvalidArgument,
              "File size in bytes (", mmap_file_->size_bytes(),
              ") is not a multiple of sizeof(int64_t)");

    tokens_ptr_ = reinterpret_cast<const int64_t*>(mmap_file_->data());
    total_tokens_ = static_cast<int64_t>(mmap_file_->size_bytes() / sizeof(int64_t));
    compute_samples();
}

BinaryTokenDataset::BinaryTokenDataset(std::vector<int64_t> in_memory_tokens, int64_t seq_len, int64_t stride)
    : in_memory_tokens_(std::move(in_memory_tokens)), seq_len_(seq_len), stride_(stride) {
    AIC_CHECK(seq_len_ > 0, InvalidArgument, "seq_len must be positive, got ", seq_len_);
    tokens_ptr_ = in_memory_tokens_.data();
    total_tokens_ = static_cast<int64_t>(in_memory_tokens_.size());
    compute_samples();
}

void BinaryTokenDataset::compute_samples() {
    if (stride_ <= 0) {
        stride_ = seq_len_;
    }
    // Need (seq_len + 1) tokens per sample to form (input, target) next-token pair
    if (total_tokens_ >= seq_len_ + 1) {
        num_samples_ = (total_tokens_ - 1 - seq_len_) / stride_ + 1;
    } else {
        num_samples_ = 0;
    }
}

std::pair<Tensor, Tensor> BinaryTokenDataset::get_item(int64_t idx) const {
    AIC_CHECK(idx >= 0 && idx < num_samples_, OutOfRange,
              "Dataset index ", idx, " out of range [0, ", num_samples_, ")");

    int64_t start = idx * stride_;

    Tensor input = Tensor::empty(Shape({seq_len_}), DType::Int64, Device::cpu());
    Tensor target = Tensor::empty(Shape({seq_len_}), DType::Int64, Device::cpu());

    std::memcpy(input.data_ptr<int64_t>(), tokens_ptr_ + start, static_cast<size_t>(seq_len_) * sizeof(int64_t));
    std::memcpy(target.data_ptr<int64_t>(), tokens_ptr_ + start + 1, static_cast<size_t>(seq_len_) * sizeof(int64_t));

    return {input, target};
}

int64_t BinaryTokenDataset::token_at(int64_t index) const {
    AIC_CHECK(index >= 0 && index < total_tokens_, OutOfRange,
              "Token index ", index, " out of bounds for total tokens ", total_tokens_);
    return tokens_ptr_[index];
}

void BinaryTokenDataset::write_binary_file(const std::string& filepath, const std::vector<int64_t>& tokens) {
    std::ofstream out(filepath, std::ios::binary);
    AIC_CHECK(out.is_open(), IOError, "Failed to open output binary token file: ", filepath);
    if (!tokens.empty()) {
        out.write(reinterpret_cast<const char*>(tokens.data()),
                  static_cast<std::streamsize>(tokens.size() * sizeof(int64_t)));
    }
}

}  // namespace aic::data
