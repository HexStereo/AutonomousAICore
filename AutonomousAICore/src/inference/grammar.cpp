// src/inference/grammar.cpp
// Implementation of Structured Grammar-Constrained Decoding Engine (Aşama 22)
#include "inference/grammar.hpp"
#include "core/logging.hpp"

#if AIC_WITH_CUDA
#include "cuda/cuda_grammar.cuh"
#include <cuda_runtime.h>
#endif

#include <cctype>
#include <cstring>
#include <queue>

namespace aic::inference {

// ----------------------------------------------------------------------------
// DFA Implementation
// ----------------------------------------------------------------------------
DFA::DFA(int num_states, int start_state)
    : num_states_(num_states), start_state_(start_state),
      transitions_(static_cast<size_t>(num_states * kMaxDfaAlphabet), kDeadState),
      accepting_(static_cast<size_t>(num_states), false) {}

void DFA::set_transition(int from_state, uint8_t byte_val, int to_state) {
    if (from_state >= 0 && from_state < num_states_) {
        transitions_[static_cast<size_t>(from_state * kMaxDfaAlphabet + byte_val)] = to_state;
    }
}

void DFA::set_range_transition(int from_state, uint8_t byte_min, uint8_t byte_max, int to_state) {
    if (from_state >= 0 && from_state < num_states_) {
        for (int b = byte_min; b <= byte_max; ++b) {
            transitions_[static_cast<size_t>(from_state * kMaxDfaAlphabet + b)] = to_state;
        }
    }
}

void DFA::set_accepting(int state, bool is_accepting) {
    if (state >= 0 && state < num_states_) {
        accepting_[static_cast<size_t>(state)] = is_accepting;
    }
}

int DFA::transition(int state, uint8_t byte_val) const noexcept {
    if (state < 0 || state >= num_states_) return kDeadState;
    return transitions_[static_cast<size_t>(state * kMaxDfaAlphabet + byte_val)];
}

bool DFA::is_accepting(int state) const noexcept {
    if (state < 0 || state >= num_states_) return false;
    return accepting_[static_cast<size_t>(state)];
}

DFA DFA::create_integer_dfa() {
    // States: 0 = start, 1 = sign seen, 2 = digits (accepting)
    DFA dfa(3, 0);
    dfa.set_transition(0, '-', 1);
    dfa.set_range_transition(0, '0', '9', 2);
    dfa.set_range_transition(1, '0', '9', 2);
    dfa.set_range_transition(2, '0', '9', 2);
    dfa.set_accepting(2, true);
    return dfa;
}

DFA DFA::create_number_dfa() {
    // States: 0 = start, 1 = sign, 2 = int digits (acc), 3 = dot, 4 = frac digits (acc)
    DFA dfa(5, 0);
    dfa.set_transition(0, '-', 1);
    dfa.set_range_transition(0, '0', '9', 2);
    dfa.set_range_transition(1, '0', '9', 2);
    dfa.set_range_transition(2, '0', '9', 2);
    dfa.set_transition(2, '.', 3);
    dfa.set_range_transition(3, '0', '9', 4);
    dfa.set_range_transition(4, '0', '9', 4);
    dfa.set_accepting(2, true);
    dfa.set_accepting(4, true);
    return dfa;
}

DFA DFA::create_string_dfa() {
    // States: 0 = start (expects '"'), 1 = inside string, 2 = escape seen, 3 = end quote (acc)
    DFA dfa(4, 0);
    dfa.set_transition(0, '"', 1);

    // In state 1, accept any byte except '"' and '\\'
    for (int b = 32; b <= 126; ++b) {
        if (b != '"' && b != '\\') {
            dfa.set_transition(1, static_cast<uint8_t>(b), 1);
        }
    }
    dfa.set_transition(1, '\\', 2);
    dfa.set_transition(1, '"', 3);

    // In state 2 (escape), accept common escape characters and return to state 1
    dfa.set_transition(2, '"', 1);
    dfa.set_transition(2, '\\', 1);
    dfa.set_transition(2, '/', 1);
    dfa.set_transition(2, 'b', 1);
    dfa.set_transition(2, 'f', 1);
    dfa.set_transition(2, 'n', 1);
    dfa.set_transition(2, 'r', 1);
    dfa.set_transition(2, 't', 1);

    dfa.set_accepting(3, true);
    return dfa;
}

DFA DFA::create_boolean_dfa() {
    // Matches "true" or "false"
    // States: 0 = start
    // 1 = 't', 2 = 'tr', 3 = 'tru', 4 = "true" (acc)
    // 5 = 'f', 6 = 'fa', 7 = 'fal', 8 = 'fals', 9 = "false" (acc)
    DFA dfa(10, 0);
    dfa.set_transition(0, 't', 1);
    dfa.set_transition(1, 'r', 2);
    dfa.set_transition(2, 'u', 3);
    dfa.set_transition(3, 'e', 4);
    dfa.set_accepting(4, true);

    dfa.set_transition(0, 'f', 5);
    dfa.set_transition(5, 'a', 6);
    dfa.set_transition(6, 'l', 7);
    dfa.set_transition(7, 's', 8);
    dfa.set_transition(8, 'e', 9);
    dfa.set_accepting(9, true);
    return dfa;
}

DFA DFA::create_keyword_dfa(std::string_view keyword) {
    if (keyword.empty()) {
        DFA dfa(1, 0);
        dfa.set_accepting(0, true);
        return dfa;
    }
    int n = static_cast<int>(keyword.size());
    DFA dfa(n + 1, 0);
    for (int i = 0; i < n; ++i) {
        dfa.set_transition(i, static_cast<uint8_t>(keyword[i]), i + 1);
    }
    dfa.set_accepting(n, true);
    return dfa;
}

// ----------------------------------------------------------------------------
// VocabularyTrie Implementation
// ----------------------------------------------------------------------------
void VocabularyTrie::insert(std::string_view piece, int64_t token_id) {
    int current = 0;
    for (char c : piece) {
        uint8_t byte_val = static_cast<uint8_t>(c);
        auto it = nodes_[current].children.find(byte_val);
        if (it == nodes_[current].children.end()) {
            int new_idx = static_cast<int>(nodes_.size());
            nodes_.emplace_back();
            nodes_[current].children[byte_val] = new_idx;
            current = new_idx;
        } else {
            current = it->second;
        }
    }
    nodes_[current].token_ids.push_back(token_id);
}

VocabularyTrie VocabularyTrie::build(const data::Tokenizer& tokenizer) {
    VocabularyTrie trie;
    int64_t V = tokenizer.vocab_size();
    for (int64_t id = 0; id < V; ++id) {
        std::string piece = tokenizer.token_to_piece(id);
        if (!piece.empty()) {
            trie.insert(piece, id);
        }
    }
    return trie;
}

// ----------------------------------------------------------------------------
// GrammarMatcher Base Implementation
// ----------------------------------------------------------------------------
bool GrammarMatcher::consume_token(int64_t token_id, const data::Tokenizer& tokenizer) {
    std::string piece = tokenizer.token_to_piece(token_id);
    for (char c : piece) {
        uint8_t b = static_cast<uint8_t>(c);
        if (!can_accept_byte(b)) {
            return false;
        }
        consume_byte(b);
    }
    return true;
}

// ----------------------------------------------------------------------------
// DFAMatcher Implementation
// ----------------------------------------------------------------------------
DFAMatcher::DFAMatcher(DFA dfa)
    : dfa_(std::move(dfa)), current_state_(dfa_.start_state()) {}

bool DFAMatcher::can_accept_byte(uint8_t byte_val) const {
    return dfa_.transition(current_state_, byte_val) != kDeadState;
}

void DFAMatcher::consume_byte(uint8_t byte_val) {
    current_state_ = dfa_.transition(current_state_, byte_val);
}

void DFAMatcher::traverse_trie(
    int trie_node_idx,
    int dfa_state,
    const VocabularyTrie& trie,
    std::vector<bool>& valid_mask
) const {
    const auto& node = trie.node(trie_node_idx);

    // If this node represents one or more complete tokens, mark them valid
    for (int64_t tid : node.token_ids) {
        if (tid >= 0 && tid < static_cast<int64_t>(valid_mask.size())) {
            valid_mask[static_cast<size_t>(tid)] = true;
        }
    }

    // Traverse children along valid DFA transitions
    for (const auto& [byte_val, child_idx] : node.children) {
        int next_dfa = dfa_.transition(dfa_state, byte_val);
        if (next_dfa != kDeadState) {
            traverse_trie(child_idx, next_dfa, trie, valid_mask);
        }
    }
}

void DFAMatcher::get_valid_tokens(
    const VocabularyTrie& trie,
    const data::Tokenizer& tokenizer,
    std::vector<bool>& valid_mask
) const {
    valid_mask.assign(static_cast<size_t>(tokenizer.vocab_size()), false);
    if (current_state_ == kDeadState) return;

    // From root of trie, explore all transitions starting at current_state_
    const auto& root = trie.node(trie.root_index());
    for (const auto& [byte_val, child_idx] : root.children) {
        int next_dfa = dfa_.transition(current_state_, byte_val);
        if (next_dfa != kDeadState) {
            traverse_trie(child_idx, next_dfa, trie, valid_mask);
        }
    }
}

// ----------------------------------------------------------------------------
// JSONSchema & JSONSchemaMatcher Implementation
// ----------------------------------------------------------------------------
JSONSchema JSONSchema::create_key_value_object(const std::vector<JSONProperty>& props) {
    JSONSchema schema;
    for (const auto& p : props) {
        schema.add_property(p.name, p.type, p.required);
    }
    return schema;
}

JSONSchemaMatcher::JSONSchemaMatcher(JSONSchema schema)
    : schema_(std::move(schema)) {
    setup_current_val_matcher();
}

void JSONSchemaMatcher::setup_current_val_matcher() {
    if (current_prop_idx_ >= schema_.properties().size()) {
        current_val_matcher_ = nullptr;
        return;
    }
    const auto& prop = schema_.properties()[current_prop_idx_];
    switch (prop.type) {
        case JSONPropertyType::Integer:
            current_val_matcher_ = std::make_shared<DFAMatcher>(DFA::create_integer_dfa());
            break;
        case JSONPropertyType::Number:
            current_val_matcher_ = std::make_shared<DFAMatcher>(DFA::create_number_dfa());
            break;
        case JSONPropertyType::Boolean:
            current_val_matcher_ = std::make_shared<DFAMatcher>(DFA::create_boolean_dfa());
            break;
        case JSONPropertyType::String:
        default:
            current_val_matcher_ = std::make_shared<DFAMatcher>(DFA::create_string_dfa());
            break;
    }
}

bool JSONSchemaMatcher::can_accept_byte(uint8_t byte_val) const {
    if (state_ == State::Finished || state_ == State::Error) {
        return false;
    }

    if (state_ == State::ExpectStartBrace) {
        return byte_val == '{';
    }

    if (state_ == State::ExpectKey) {
        if (current_prop_idx_ < schema_.properties().size()) {
            std::string expected_key = "\"" + schema_.properties()[current_prop_idx_].name + "\"";
            std::string target = buffer_ + static_cast<char>(byte_val);
            size_t last_delim = target.find_last_of("{,");
            std::string after = (last_delim != std::string::npos) ? target.substr(last_delim + 1) : target;
            size_t first = after.find_first_not_of(" \t\n\r");
            if (first == std::string::npos) {
                return after.size() <= 1 && byte_val == ' ';
            }
            std::string stripped = after.substr(first);
            if (expected_key.rfind(stripped, 0) == 0) return true;
        }
        return false;
    }

    if (state_ == State::ExpectColon) {
        return byte_val == ':';
    }

    if (state_ == State::ExpectValue) {
        if (current_val_matcher_) {
            if (current_val_matcher_->current_state() == 0 && byte_val == ' ') {
                return !buffer_.empty() && buffer_.back() == ':';
            }
            if (current_val_matcher_->can_accept_byte(byte_val)) {
                return true;
            }
            // If value matcher is already accepting, we might transition to comma or brace
            if (current_val_matcher_->is_terminated()) {
                if (current_prop_idx_ + 1 < schema_.properties().size()) {
                    return byte_val == ',';
                } else {
                    return byte_val == '}';
                }
            }
        }
        return false;
    }

    if (state_ == State::ExpectCommaOrEnd) {
        if (current_prop_idx_ + 1 < schema_.properties().size()) {
            return byte_val == ',';
        } else {
            return byte_val == '}';
        }
    }

    return false;
}

void JSONSchemaMatcher::consume_byte(uint8_t byte_val) {
    buffer_ += static_cast<char>(byte_val);

    if (state_ == State::ExpectStartBrace) {
        if (byte_val == '{') {
            state_ = State::ExpectKey;
        }
        return;
    }

    if (state_ == State::ExpectKey) {
        if (current_prop_idx_ < schema_.properties().size()) {
            std::string expected = "\"" + schema_.properties()[current_prop_idx_].name + "\"";
            size_t last_delim = buffer_.find_last_of("{,");
            std::string after = (last_delim != std::string::npos) ? buffer_.substr(last_delim + 1) : buffer_;
            if (after.find(expected) != std::string::npos) {
                state_ = State::ExpectColon;
            }
        }
        return;
    }

    if (state_ == State::ExpectColon) {
        if (byte_val == ':') {
            state_ = State::ExpectValue;
            setup_current_val_matcher();
        }
        return;
    }

    if (state_ == State::ExpectValue) {
        if (current_val_matcher_) {
            if (current_val_matcher_->current_state() == 0 && byte_val == ' ') {
                return;
            }
            if (current_val_matcher_->can_accept_byte(byte_val)) {
                current_val_matcher_->consume_byte(byte_val);
                return;
            }
            // Transition to next property or finish if value matcher is terminated
            if (current_val_matcher_->is_terminated()) {
                if (byte_val == ',') {
                    current_prop_idx_++;
                    state_ = State::ExpectKey;
                    setup_current_val_matcher();
                    return;
                } else if (byte_val == '}') {
                    state_ = State::Finished;
                    return;
                }
            }
        }
        return;
    }

    if (state_ == State::ExpectCommaOrEnd) {
        if (byte_val == ',') {
            current_prop_idx_++;
            state_ = State::ExpectKey;
            setup_current_val_matcher();
        } else if (byte_val == '}') {
            state_ = State::Finished;
        }
    }
}

void JSONSchemaMatcher::get_valid_tokens(
    const VocabularyTrie& trie,
    const data::Tokenizer& tokenizer,
    std::vector<bool>& valid_mask
) const {
    valid_mask.assign(static_cast<size_t>(tokenizer.vocab_size()), false);

    // For any candidate token, test if it can be consumed without error
    int64_t V = tokenizer.vocab_size();
    for (int64_t tid = 0; tid < V; ++tid) {
        std::string piece = tokenizer.token_to_piece(tid);
        if (piece.empty()) continue;

        auto test_matcher = fork();
        bool ok = true;
        for (char c : piece) {
            uint8_t b = static_cast<uint8_t>(c);
            if (!test_matcher->can_accept_byte(b)) {
                ok = false;
                break;
            }
            test_matcher->consume_byte(b);
        }
        if (ok) {
            valid_mask[static_cast<size_t>(tid)] = true;
        }
    }
}

bool JSONSchemaMatcher::is_terminated() const {
    return state_ == State::Finished;
}

// ----------------------------------------------------------------------------
// GrammarLogitProcessor Implementation
// ----------------------------------------------------------------------------
void GrammarLogitProcessor::apply_mask(
    Tensor& logits,
    const std::vector<bool>& valid_mask,
    float mask_value
) {
    int64_t V = logits.dim(-1);
    if (V != static_cast<int64_t>(valid_mask.size())) {
        AIC_THROW(ShapeMismatch, "GrammarLogitProcessor::apply_mask vocab size mismatch");
    }

    int64_t total_elements = logits.numel();
    int64_t B = total_elements / V;

#if AIC_WITH_CUDA
    if (logits.is_cuda()) {
        std::vector<uint8_t> u8_mask(valid_mask.size());
        for (size_t i = 0; i < valid_mask.size(); ++i) {
            u8_mask[i] = valid_mask[i] ? 1 : 0;
        }
        Tensor dev_mask = Tensor::from_blob(u8_mask.data(), Shape({V}), DType::UInt8).to(logits.device());
        cuda::launch_apply_logit_mask(
            logits.data_ptr<float>(),
            dev_mask.data_ptr<uint8_t>(),
            B,
            V,
            mask_value
        );
        cudaDeviceSynchronize();
        return;
    }
#endif

    // CPU execution path
    float* p_logits = logits.data_ptr<float>();
    for (int64_t b = 0; b < B; ++b) {
        float* row = p_logits + b * V;
        for (int64_t v = 0; v < V; ++v) {
            if (!valid_mask[static_cast<size_t>(v)]) {
                row[v] = mask_value;
            }
        }
    }
}

} // namespace aic::inference
