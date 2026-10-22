// tests/unit/test_grammar.cpp
// Unit tests for Structured Grammar-Constrained Decoding Engine (Aşama 22)
#include "tests/test_framework.hpp"
#include "inference/grammar.hpp"
#include "inference/continuous_batch_engine.hpp"
#include "data/tokenizer.hpp"
#include "models/transformer_lm.hpp"
#include "core/device.hpp"
#include "runtime/cuda_info.hpp"

#include <cmath>
#include <memory>
#include <string>
#include <vector>

using namespace aic;
using namespace aic::inference;
using namespace aic::data;
using namespace aic::models;

AIC_TEST(GrammarTest, DFAStateTransitionsAndRegex) {
    // 1. Integer DFA
    DFA int_dfa = DFA::create_integer_dfa();
    auto run_dfa = [](const DFA& dfa, std::string_view str) -> bool {
        int state = dfa.start_state();
        for (char c : str) {
            state = dfa.transition(state, static_cast<uint8_t>(c));
            if (state == kDeadState) return false;
        }
        return dfa.is_accepting(state);
    };

    ASSERT_TRUE(run_dfa(int_dfa, "42"));
    ASSERT_TRUE(run_dfa(int_dfa, "-1024"));
    ASSERT_TRUE(run_dfa(int_dfa, "0"));
    ASSERT_FALSE(run_dfa(int_dfa, ""));
    ASSERT_FALSE(run_dfa(int_dfa, "-"));
    ASSERT_FALSE(run_dfa(int_dfa, "abc"));
    ASSERT_FALSE(run_dfa(int_dfa, "12a"));

    // 2. Number DFA (Float/Double)
    DFA num_dfa = DFA::create_number_dfa();
    ASSERT_TRUE(run_dfa(num_dfa, "3.1415"));
    ASSERT_TRUE(run_dfa(num_dfa, "-0.005"));
    ASSERT_TRUE(run_dfa(num_dfa, "100"));
    ASSERT_FALSE(run_dfa(num_dfa, "1.2.3"));
    ASSERT_FALSE(run_dfa(num_dfa, "abc"));
    ASSERT_FALSE(run_dfa(num_dfa, "-"));

    // 3. Quoted String DFA
    DFA str_dfa = DFA::create_string_dfa();
    ASSERT_TRUE(run_dfa(str_dfa, "\"hello world\""));
    ASSERT_TRUE(run_dfa(str_dfa, "\"\""));
    ASSERT_TRUE(run_dfa(str_dfa, "\"with \\\"escaped\\\" quote\""));
    ASSERT_FALSE(run_dfa(str_dfa, "unquoted"));
    ASSERT_FALSE(run_dfa(str_dfa, "\"unterminated"));

    // 4. Boolean DFA
    DFA bool_dfa = DFA::create_boolean_dfa();
    ASSERT_TRUE(run_dfa(bool_dfa, "true"));
    ASSERT_TRUE(run_dfa(bool_dfa, "false"));
    ASSERT_FALSE(run_dfa(bool_dfa, "True"));
    ASSERT_FALSE(run_dfa(bool_dfa, "FALSE"));
    ASSERT_FALSE(run_dfa(bool_dfa, "tru"));
    ASSERT_FALSE(run_dfa(bool_dfa, "falsee"));
}

AIC_TEST(GrammarTest, VocabularyTrieTokenMatching) {
    VocabularyTrie trie;
    trie.insert("apple", 101);
    trie.insert("app", 102);
    trie.insert("banana", 103);
    trie.insert("band", 104);

    ASSERT_TRUE(trie.num_nodes() > 10);
    const auto& root = trie.node(trie.root_index());
    ASSERT_TRUE(root.children.find('a') != root.children.end());
    ASSERT_TRUE(root.children.find('b') != root.children.end());
    ASSERT_TRUE(root.children.find('z') == root.children.end());

    // Build trie from standard Byte-Level Tokenizer
    Tokenizer tok = Tokenizer::create_default();
    VocabularyTrie vocab_trie = VocabularyTrie::build(tok);
    ASSERT_TRUE(vocab_trie.num_nodes() >= 256);

    // Verify token retrieval via trie traversal with DFAMatcher
    DFA bool_dfa = DFA::create_boolean_dfa();
    DFAMatcher matcher(bool_dfa);
    std::vector<bool> valid_mask;
    matcher.get_valid_tokens(vocab_trie, tok, valid_mask);

    // Tokens starting with 't' or 'f' should be candidate tokens
    bool found_valid = false;
    for (size_t i = 0; i < valid_mask.size(); ++i) {
        if (valid_mask[i]) {
            found_valid = true;
            std::string piece = tok.token_to_piece(static_cast<int64_t>(i));
            ASSERT_TRUE(piece[0] == 't' || piece[0] == 'f');
        }
    }
    ASSERT_TRUE(found_valid);
}

AIC_TEST(GrammarTest, JSONSchemaAutomatonValidation) {
    JSONSchema schema;
    schema.add_property("name", JSONPropertyType::String);
    schema.add_property("age", JSONPropertyType::Integer);
    schema.add_property("active", JSONPropertyType::Boolean);

    JSONSchemaMatcher matcher(schema);
    std::string valid_json = "{\"name\": \"Bob\", \"age\": 35, \"active\": true}";

    for (char c : valid_json) {
        uint8_t b = static_cast<uint8_t>(c);
        ASSERT_TRUE(matcher.can_accept_byte(b));
        matcher.consume_byte(b);
    }

    ASSERT_TRUE(matcher.is_terminated());

    // Invalid JSON transition verification
    JSONSchemaMatcher invalid_matcher(schema);
    std::string invalid_prefix = "{\"name\": ";
    for (char c : invalid_prefix) {
        invalid_matcher.consume_byte(static_cast<uint8_t>(c));
    }
    // "name" property must be a string (starts with '"'). A digit should be rejected!
    ASSERT_FALSE(invalid_matcher.can_accept_byte('1'));
    ASSERT_FALSE(invalid_matcher.can_accept_byte('{'));
    ASSERT_TRUE(invalid_matcher.can_accept_byte('"'));
}

AIC_TEST(GrammarTest, GrammarLogitProcessorMasking) {
    int64_t B = 2;
    int64_t V = 8;
    Tensor logits = Tensor::full(Shape({B, V}), 2.0f, DType::Float32, Device::cpu());

    std::vector<bool> valid_mask = {true, false, true, false, false, true, false, true};
    GrammarLogitProcessor::apply_mask(logits, valid_mask, -1e9f);

    const float* p = logits.data_ptr<float>();
    for (int64_t b = 0; b < B; ++b) {
        const float* row = p + b * V;
        ASSERT_NEAR(row[0], 2.0f, 1e-4f);
        ASSERT_NEAR(row[1], -1e9f, 1e-2f);
        ASSERT_NEAR(row[2], 2.0f, 1e-4f);
        ASSERT_NEAR(row[3], -1e9f, 1e-2f);
        ASSERT_NEAR(row[4], -1e9f, 1e-2f);
        ASSERT_NEAR(row[5], 2.0f, 1e-4f);
        ASSERT_NEAR(row[6], -1e9f, 1e-2f);
        ASSERT_NEAR(row[7], 2.0f, 1e-4f);
    }
}

AIC_TEST(GrammarTest, ConstrainedAutoregressiveGeneration) {
    auto tok = std::make_shared<Tokenizer>(Tokenizer::create_default());

    TransformerLMConfig cfg;
    cfg.vocab_size = tok->vocab_size();
    cfg.hidden_dim = 64;
    cfg.num_layers = 1;
    cfg.num_heads = 2;
    cfg.max_seq_len = 128;

    auto model = std::make_shared<TransformerLM>(cfg);

    ContinuousBatchConfig b_cfg;
    b_cfg.max_num_seqs = 4;
    b_cfg.max_num_batched_tokens = 64;
    b_cfg.tokenizer = tok;

    ContinuousBatchEngine engine(model, b_cfg);

    JSONSchema schema;
    schema.add_property("id", JSONPropertyType::Integer);

    auto matcher = std::make_shared<JSONSchemaMatcher>(schema);
    std::vector<int64_t> prompt = {static_cast<int64_t>('{')}; // Start with '{'
    matcher->consume_byte('{');

    uint64_t req_id = engine.add_request(prompt, 32, 0.7f, 10, 0.9f, -1, matcher);

    int max_steps = 100;
    while (engine.has_active_requests() && max_steps-- > 0) {
        engine.step();
    }

    auto req = engine.get_request(req_id);
    ASSERT_TRUE(req != nullptr);
    ASSERT_TRUE(req->status == RequestStatus::FINISHED);

    // Verify all generated tokens follow the schema
    std::string full_output = "{" + tok->decode(req->output_tokens);
    ASSERT_TRUE(full_output.find("\"id\":") != std::string::npos);
    ASSERT_TRUE(req->grammar_matcher->is_terminated());
}

AIC_TEST(GrammarTest, MCTSBranchStateForking) {
    JSONSchema schema;
    schema.add_property("status", JSONPropertyType::String);
    schema.add_property("code", JSONPropertyType::Integer);

    JSONSchemaMatcher root_matcher(schema);
    std::string prefix = "{\"status\": \"OK\", ";
    for (char c : prefix) {
        root_matcher.consume_byte(static_cast<uint8_t>(c));
    }

    // Fork parser state for 2 independent reasoning paths
    auto branch1 = root_matcher.fork();
    auto branch2 = root_matcher.fork();

    // Branch 1 consumes: "\"code\": 200}"
    std::string b1_suffix = "\"code\": 200}";
    for (char c : b1_suffix) {
        ASSERT_TRUE(branch1->can_accept_byte(static_cast<uint8_t>(c)));
        branch1->consume_byte(static_cast<uint8_t>(c));
    }
    ASSERT_TRUE(branch1->is_terminated());

    // Branch 2 consumes: "\"code\": 404}"
    std::string b2_suffix = "\"code\": 404}";
    for (char c : b2_suffix) {
        ASSERT_TRUE(branch2->can_accept_byte(static_cast<uint8_t>(c)));
        branch2->consume_byte(static_cast<uint8_t>(c));
    }
    ASSERT_TRUE(branch2->is_terminated());

    // Verify root matcher is still at the fork point
    ASSERT_FALSE(root_matcher.is_terminated());
}

AIC_TEST(GrammarTest, CudaLogitMaskingAndParity) {
    if (!is_cuda_available()) {
        std::cout << "[GrammarTest] CUDA not available, skipping GPU logit mask test\n";
        return;
    }

    int64_t B = 4;
    int64_t V = 512;
    Tensor cpu_logits = Tensor::uniform(Shape({B, V}), -5.0f, 5.0f, DType::Float32, Device::cpu());
    Tensor cuda_logits = cpu_logits.to(Device::cuda(0));

    std::vector<bool> valid_mask(static_cast<size_t>(V));
    for (int64_t v = 0; v < V; ++v) {
        valid_mask[static_cast<size_t>(v)] = (v % 3 != 0); // Disallow 1/3 of the vocabulary
    }

    GrammarLogitProcessor::apply_mask(cpu_logits, valid_mask, -1e9f);
    GrammarLogitProcessor::apply_mask(cuda_logits, valid_mask, -1e9f);

    Tensor cuda_result = cuda_logits.cpu();
    const float* p_cpu = cpu_logits.data_ptr<float>();
    const float* p_cuda = cuda_result.data_ptr<float>();

    for (int64_t i = 0; i < cpu_logits.numel(); ++i) {
        ASSERT_NEAR(p_cpu[i], p_cuda[i], 1e-4f);
    }
}
