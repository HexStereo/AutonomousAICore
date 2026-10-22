// include/data/tokenizer.hpp
// Byte-level Byte-Pair Encoding (BPE) Tokenizer with zero out-of-vocabulary guarantee.
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace aic::data {

class Tokenizer {
public:
    Tokenizer();

    // Factory: returns a pre-configured byte-level tokenizer with base 256 byte tokens,
    // special tokens, and common merges.
    static Tokenizer create_default();

    // Vocabulary sizes and special token identifiers
    int64_t vocab_size() const noexcept { return static_cast<int64_t>(id_to_token_.size()); }
    int64_t bos_token_id() const noexcept { return bos_id_; }
    int64_t eos_token_id() const noexcept { return eos_id_; }
    int64_t pad_token_id() const noexcept { return pad_id_; }
    int64_t unk_token_id() const noexcept { return unk_id_; }

    // Text encoding: converts string to sequence of token IDs
    std::vector<int64_t> encode(std::string_view text, bool add_bos = false, bool add_eos = false) const;

    // Token decoding: converts token sequence back to UTF-8 string
    std::string decode(std::span<const int64_t> tokens) const;
    std::string decode(const std::vector<int64_t>& tokens) const {
        return decode(std::span<const int64_t>(tokens.data(), tokens.size()));
    }

    // Bidirectional token <-> string piece queries
    std::string token_to_piece(int64_t id) const;
    int64_t piece_to_token(std::string_view piece) const;

    // Vocabulary management
    int64_t add_token(std::string token);
    void add_merge(std::string piece1, std::string piece2);

    // Serialization and deserialization
    void save(const std::string& path) const;
    static Tokenizer load(const std::string& path);

private:
    std::unordered_map<std::string, int64_t> token_to_id_;
    std::vector<std::string> id_to_token_;

    // Map: hash(pair<int64_t, int64_t>) -> merged token id
    std::unordered_map<uint64_t, int64_t> merges_;

    int64_t bos_id_ = -1;
    int64_t eos_id_ = -1;
    int64_t pad_id_ = -1;
    int64_t unk_id_ = -1;

    static uint64_t pair_hash(int64_t a, int64_t b) noexcept {
        return (static_cast<uint64_t>(a) << 32) | (static_cast<uint64_t>(b) & 0xFFFFFFFFULL);
    }
};

}  // namespace aic::data
