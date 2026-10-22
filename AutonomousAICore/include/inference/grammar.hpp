// include/inference/grammar.hpp
// Structured Grammar-Constrained Decoding Engine (Aşama 22)
// Enforces formal Context-Free Grammars (CFG), JSON Schemas, and Regex Constraints
// via pushdown/finite automata and vocabulary prefix trie logit masking.
#pragma once

#include "core/device.hpp"
#include "core/dtype.hpp"
#include "core/error.hpp"
#include "core/shape.hpp"
#include "data/tokenizer.hpp"
#include "tensor/tensor.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace aic::inference {

// ----------------------------------------------------------------------------
// Deterministic Finite Automaton (DFA) State & Transitions
// ----------------------------------------------------------------------------
constexpr int kDeadState = -1;
constexpr int kMaxDfaAlphabet = 256;

class DFA {
public:
    DFA() = default;
    explicit DFA(int num_states, int start_state = 0);

    void set_transition(int from_state, uint8_t byte_val, int to_state);
    void set_range_transition(int from_state, uint8_t byte_min, uint8_t byte_max, int to_state);
    void set_accepting(int state, bool is_accepting = true);

    int transition(int state, uint8_t byte_val) const noexcept;
    bool is_accepting(int state) const noexcept;

    int num_states() const noexcept { return num_states_; }
    int start_state() const noexcept { return start_state_; }

    // Helpers to construct standard DFAs
    static DFA create_integer_dfa();
    static DFA create_number_dfa(); // Float with optional minus and decimal
    static DFA create_string_dfa(); // Quoted string with escapes
    static DFA create_boolean_dfa(); // "true" or "false"
    static DFA create_keyword_dfa(std::string_view keyword);

private:
    int num_states_ = 0;
    int start_state_ = 0;
    std::vector<int> transitions_; // Flat table: [num_states * 256]
    std::vector<bool> accepting_;
};

// ----------------------------------------------------------------------------
// Vocabulary Prefix Trie
// ----------------------------------------------------------------------------
struct TrieNode {
    std::unordered_map<uint8_t, int> children;
    std::vector<int64_t> token_ids; // Tokens that end exactly at this node
};

class VocabularyTrie {
public:
    VocabularyTrie() {
        nodes_.emplace_back(); // Root node at index 0
    }

    // Build the prefix trie from the given tokenizer vocabulary
    static VocabularyTrie build(const data::Tokenizer& tokenizer);

    void insert(std::string_view piece, int64_t token_id);

    const TrieNode& node(int index) const noexcept { return nodes_[index]; }
    int root_index() const noexcept { return 0; }
    size_t num_nodes() const noexcept { return nodes_.size(); }

private:
    std::vector<TrieNode> nodes_;
};

// ----------------------------------------------------------------------------
// JSON Schema Specification & Parser Automaton
// ----------------------------------------------------------------------------
enum class JSONPropertyType {
    String,
    Integer,
    Number,
    Boolean,
    Array,
    Object
};

struct JSONProperty {
    std::string name;
    JSONPropertyType type = JSONPropertyType::String;
    bool required = true;
};

class JSONSchema {
public:
    JSONSchema() = default;

    // Builder pattern
    JSONSchema& add_property(std::string name, JSONPropertyType type, bool required = true) {
        properties_.push_back({std::move(name), type, required});
        return *this;
    }

    const std::vector<JSONProperty>& properties() const noexcept { return properties_; }

    // Standard schemas
    static JSONSchema create_key_value_object(const std::vector<JSONProperty>& props);

private:
    std::vector<JSONProperty> properties_;
};

// ----------------------------------------------------------------------------
// Grammar Matcher (Tracks Autoregressive State and Filters Tokens)
// ----------------------------------------------------------------------------
class GrammarMatcher {
public:
    virtual ~GrammarMatcher() = default;

    // Clone parser state for MCTS / Tree-of-Thoughts branch exploration
    virtual std::shared_ptr<GrammarMatcher> fork() const = 0;

    // Byte-level checks
    virtual bool can_accept_byte(uint8_t byte_val) const = 0;
    virtual void consume_byte(uint8_t byte_val) = 0;

    // Token-level consume
    virtual bool consume_token(int64_t token_id, const data::Tokenizer& tokenizer);

    // Filter valid tokens in the vocabulary using the Vocabulary Trie
    virtual void get_valid_tokens(
        const VocabularyTrie& trie,
        const data::Tokenizer& tokenizer,
        std::vector<bool>& valid_mask
    ) const = 0;

    virtual bool is_terminated() const = 0;
};

// DFA-based matcher (for Regex or flat syntax)
class DFAMatcher : public GrammarMatcher {
public:
    explicit DFAMatcher(DFA dfa);

    virtual std::shared_ptr<GrammarMatcher> fork() const override {
        return std::make_shared<DFAMatcher>(*this);
    }

    virtual bool can_accept_byte(uint8_t byte_val) const override;
    virtual void consume_byte(uint8_t byte_val) override;

    virtual void get_valid_tokens(
        const VocabularyTrie& trie,
        const data::Tokenizer& tokenizer,
        std::vector<bool>& valid_mask
    ) const override;

    virtual bool is_terminated() const override {
        return dfa_.is_accepting(current_state_);
    }

    int current_state() const noexcept { return current_state_; }

private:
    DFA dfa_;
    int current_state_ = 0;

    void traverse_trie(
        int trie_node_idx,
        int dfa_state,
        const VocabularyTrie& trie,
        std::vector<bool>& valid_mask
    ) const;
};

// JSON Schema Matcher (enforces rigid JSON structure)
class JSONSchemaMatcher : public GrammarMatcher {
public:
    explicit JSONSchemaMatcher(JSONSchema schema);

    JSONSchemaMatcher(const JSONSchemaMatcher& other)
        : schema_(other.schema_),
          buffer_(other.buffer_),
          current_prop_idx_(other.current_prop_idx_),
          state_(other.state_) {
        if (other.current_val_matcher_) {
            current_val_matcher_ = std::make_shared<DFAMatcher>(*other.current_val_matcher_);
        }
    }

    virtual std::shared_ptr<GrammarMatcher> fork() const override {
        return std::make_shared<JSONSchemaMatcher>(*this);
    }

    virtual bool can_accept_byte(uint8_t byte_val) const override;
    virtual void consume_byte(uint8_t byte_val) override;

    virtual void get_valid_tokens(
        const VocabularyTrie& trie,
        const data::Tokenizer& tokenizer,
        std::vector<bool>& valid_mask
    ) const override;

    virtual bool is_terminated() const override;

    const std::string& buffer() const noexcept { return buffer_; }

private:
    JSONSchema schema_;
    std::string buffer_;
    size_t current_prop_idx_ = 0;

    enum class State {
        ExpectStartBrace,
        ExpectKey,
        ExpectColon,
        ExpectValue,
        ExpectCommaOrEnd,
        Finished,
        Error
    };
    State state_ = State::ExpectStartBrace;
    std::shared_ptr<DFAMatcher> current_val_matcher_;

    void setup_current_val_matcher();
};

// ----------------------------------------------------------------------------
// Logit Masking Processor
// ----------------------------------------------------------------------------
class GrammarLogitProcessor {
public:
    // Applies -inf (mask_value) to invalid tokens according to valid_mask
    static void apply_mask(Tensor& logits, const std::vector<bool>& valid_mask, float mask_value = -1e9f);
};

} // namespace aic::inference
