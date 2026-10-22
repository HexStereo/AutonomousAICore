// src/data/tokenizer.cpp
// Byte-level Byte-Pair Encoding (BPE) Tokenizer implementation.
#include "data/tokenizer.hpp"
#include "core/error.hpp"

#include <fstream>
#include <sstream>

namespace aic::data {

Tokenizer::Tokenizer() {
    pad_id_ = add_token("<|pad|>");
    bos_id_ = add_token("<|bos|>");
    eos_id_ = add_token("<|eos|>");
    unk_id_ = add_token("<|unk|>");

    // Base 256 single-byte vocabulary
    for (int b = 0; b < 256; ++b) {
        std::string byte_piece(1, static_cast<char>(static_cast<unsigned char>(b)));
        add_token(byte_piece);
    }
}

int64_t Tokenizer::add_token(std::string token) {
    auto it = token_to_id_.find(token);
    if (it != token_to_id_.end()) {
        return it->second;
    }
    int64_t new_id = static_cast<int64_t>(id_to_token_.size());
    token_to_id_[token] = new_id;
    id_to_token_.push_back(std::move(token));
    return new_id;
}

void Tokenizer::add_merge(std::string piece1, std::string piece2) {
    int64_t id1 = piece_to_token(piece1);
    if (id1 < 0) {
        id1 = add_token(piece1);
    }
    int64_t id2 = piece_to_token(piece2);
    if (id2 < 0) {
        id2 = add_token(piece2);
    }

    std::string merged_piece = piece1 + piece2;
    int64_t merged_id = add_token(merged_piece);

    uint64_t key = pair_hash(id1, id2);
    merges_[key] = merged_id;
}

Tokenizer Tokenizer::create_default() {
    Tokenizer tok;

    // Register common subword merges for natural language and code
    const std::vector<std::pair<std::string, std::string>> common_merges = {
        {"t", "h"}, {"h", "e"}, {"i", "n"}, {"e", "r"}, {"a", "n"}, {"r", "e"},
        {"o", "n"}, {"a", "t"}, {"e", "n"}, {"n", "d"}, {"t", "i"}, {"e", "s"},
        {"o", "r"}, {"t", "e"}, {"a", "r"}, {"a", "l"}, {"i", "t"}, {"a", "s"},
        {"i", "s"}, {"e", "d"}, {"th", "e"}, {"i", "ng"}, {"a", "nd"}, {"o", "f"},
        {"t", "o"}, {"i", "n"}, {"i", "s"}, {"f", "o"}, {"fo", "r"}, {"o", "n"},
        {" ", "t"}, {" ", "a"}, {" ", "o"}, {" ", "i"}, {" ", "w"}, {" ", "b"},
        {" ", "f"}, {" ", "s"}, {" ", "c"}, {" ", "m"}, {" ", "d"}, {" ", "p"},
        {" ", "th"}, {" ", "the"}, {" ", "in"}, {" ", "to"}, {" ", "of"}, {" ", "and"},
        {" ", "for"}, {" ", "is"}, {" ", "on"}, {" ", "that"}, {" ", "with"},
        {" ", "as"}, {" ", "at"}, {" ", "by"}, {" ", "from"}, {" ", "this"},
        {"\n", "\n"}, {" ", " "}, {"  ", "  "}
    };

    for (const auto& [p1, p2] : common_merges) {
        tok.add_merge(p1, p2);
    }

    return tok;
}

std::vector<int64_t> Tokenizer::encode(std::string_view text, bool add_bos, bool add_eos) const {
    std::vector<int64_t> tokens;
    if (add_bos && bos_id_ >= 0) {
        tokens.push_back(bos_id_);
    }

    if (text.empty()) {
        if (add_eos && eos_id_ >= 0) {
            tokens.push_back(eos_id_);
        }
        return tokens;
    }

    // 1. Initial byte-level token sequence
    std::vector<int64_t> current;
    current.reserve(text.size());
    for (char c : text) {
        unsigned char byte_val = static_cast<unsigned char>(c);
        int64_t byte_id = 4 + static_cast<int64_t>(byte_val);
        current.push_back(byte_id);
    }

    // 2. Iterative BPE merges
    if (!merges_.empty()) {
        while (current.size() >= 2) {
            int64_t best_pair_idx = -1;
            int64_t best_merged_id = -1;

            for (size_t i = 0; i < current.size() - 1; ++i) {
                uint64_t key = pair_hash(current[i], current[i + 1]);
                auto it = merges_.find(key);
                if (it != merges_.end()) {
                    if (best_pair_idx < 0 || it->second < best_merged_id) {
                        best_pair_idx = static_cast<int64_t>(i);
                        best_merged_id = it->second;
                    }
                }
            }

            if (best_pair_idx < 0) {
                break; // No more eligible merges
            }

            // Apply merge
            std::vector<int64_t> next_tokens;
            next_tokens.reserve(current.size());
            for (size_t i = 0; i < current.size(); ++i) {
                if (static_cast<int64_t>(i) == best_pair_idx) {
                    next_tokens.push_back(best_merged_id);
                    ++i; // skip second token in merged pair
                } else {
                    next_tokens.push_back(current[i]);
                }
            }
            current = std::move(next_tokens);
        }
    }

    for (int64_t id : current) {
        tokens.push_back(id);
    }

    if (add_eos && eos_id_ >= 0) {
        tokens.push_back(eos_id_);
    }

    return tokens;
}

std::string Tokenizer::decode(std::span<const int64_t> tokens) const {
    std::string out;
    for (int64_t id : tokens) {
        if (id == pad_id_ || id == bos_id_ || id == eos_id_ || id == unk_id_) {
            continue; // Skip special tokens in text rendering
        }
        if (id >= 0 && id < static_cast<int64_t>(id_to_token_.size())) {
            out += id_to_token_[static_cast<size_t>(id)];
        }
    }
    return out;
}

std::string Tokenizer::token_to_piece(int64_t id) const {
    if (id >= 0 && id < static_cast<int64_t>(id_to_token_.size())) {
        return id_to_token_[static_cast<size_t>(id)];
    }
    return "<|unk|>";
}

int64_t Tokenizer::piece_to_token(std::string_view piece) const {
    auto it = token_to_id_.find(std::string(piece));
    if (it != token_to_id_.end()) {
        return it->second;
    }
    return -1;
}

void Tokenizer::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    AIC_CHECK(out.is_open(), IOError, "Failed to open file for tokenizer save: ", path);

    // Write header and vocabulary size
    uint64_t num_tokens = id_to_token_.size();
    out.write(reinterpret_cast<const char*>(&num_tokens), sizeof(num_tokens));

    for (const auto& tok : id_to_token_) {
        uint32_t len = static_cast<uint32_t>(tok.size());
        out.write(reinterpret_cast<const char*>(&len), sizeof(len));
        out.write(tok.data(), len);
    }

    // Write merges
    uint64_t num_merges = merges_.size();
    out.write(reinterpret_cast<const char*>(&num_merges), sizeof(num_merges));
    for (const auto& [key, merged_id] : merges_) {
        out.write(reinterpret_cast<const char*>(&key), sizeof(key));
        out.write(reinterpret_cast<const char*>(&merged_id), sizeof(merged_id));
    }
}

Tokenizer Tokenizer::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    AIC_CHECK(in.is_open(), IOError, "Failed to open file for tokenizer load: ", path);

    Tokenizer tok;
    tok.token_to_id_.clear();
    tok.id_to_token_.clear();
    tok.merges_.clear();

    uint64_t num_tokens = 0;
    in.read(reinterpret_cast<char*>(&num_tokens), sizeof(num_tokens));
    for (uint64_t i = 0; i < num_tokens; ++i) {
        uint32_t len = 0;
        in.read(reinterpret_cast<char*>(&len), sizeof(len));
        std::string s(len, '\0');
        in.read(s.data(), len);
        tok.add_token(std::move(s));
    }

    uint64_t num_merges = 0;
    in.read(reinterpret_cast<char*>(&num_merges), sizeof(num_merges));
    for (uint64_t i = 0; i < num_merges; ++i) {
        uint64_t key = 0;
        int64_t merged_id = 0;
        in.read(reinterpret_cast<char*>(&key), sizeof(key));
        in.read(reinterpret_cast<char*>(&merged_id), sizeof(merged_id));
        tok.merges_[key] = merged_id;
    }

    tok.pad_id_ = tok.piece_to_token("<|pad|>");
    tok.bos_id_ = tok.piece_to_token("<|bos|>");
    tok.eos_id_ = tok.piece_to_token("<|eos|>");
    tok.unk_id_ = tok.piece_to_token("<|unk|>");

    return tok;
}

}  // namespace aic::data
