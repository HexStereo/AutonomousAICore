// tests/unit/test_reasoning.cpp
// Unit tests for Autonomous Reasoning Engine (Tree-of-Thoughts / MCTS & Self-Verification).
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "data/tokenizer.hpp"
#include "models/transformer_lm.hpp"
#include "reasoning/reasoning.hpp"
#include "tests/test_framework.hpp"

using namespace aic;
using namespace aic::reasoning;
using namespace aic::models;

AIC_TEST(ThoughtNodeTest, PuctAndBackpropagation) {
    ThoughtStep root_step;
    root_step.tokens = {1, 2, 3};
    root_step.score = 1.0f;
    root_step.is_valid = true;

    auto root = std::make_shared<ThoughtNode>(std::weak_ptr<ThoughtNode>{}, root_step, /*depth=*/0);

    // Add 2 children
    ThoughtStep s1;
    s1.tokens = {4, 5};
    s1.score = 0.6f;
    s1.is_valid = true;
    auto child1 = std::make_shared<ThoughtNode>(root, s1, 1, /*prior=*/0.6f);
    root->add_child(child1);

    ThoughtStep s2;
    s2.tokens = {6, 7};
    s2.score = 0.8f;
    s2.is_valid = true;
    auto child2 = std::make_shared<ThoughtNode>(root, s2, 1, /*prior=*/0.8f);
    root->add_child(child2);

    ASSERT_EQ(root->children().size(), 2);
    ASSERT_EQ(child1->visits(), 0);
    ASSERT_EQ(child2->visits(), 0);

    // Initial best child selection (prior + exploration)
    auto best = root->select_best_child(/*c_puct=*/1.414f);
    ASSERT_TRUE(best == child2); // child2 has higher initial score/prior (0.8 > 0.6)

    // Backpropagate a reward of 0.9 through child2
    child2->backpropagate(0.9f);
    ASSERT_EQ(child2->visits(), 1);
    ASSERT_NEAR(child2->total_value(), 0.9f, 1e-5f);
    ASSERT_NEAR(child2->q_value(), 0.9f, 1e-5f);
    ASSERT_EQ(root->visits(), 1);

    // Now backpropagate through child1 with reward 0.95
    child1->backpropagate(0.95f);
    ASSERT_EQ(child1->visits(), 1);
    ASSERT_NEAR(child1->q_value(), 0.95f, 1e-5f);
    ASSERT_EQ(root->visits(), 2);
}

AIC_TEST(ThoughtNodeTest, AccumulatedTokensAndTrace) {
    auto root = std::make_shared<ThoughtNode>(std::weak_ptr<ThoughtNode>{}, ThoughtStep{{1, 2}, "Premise"}, 0);
    auto step1 = std::make_shared<ThoughtNode>(root, ThoughtStep{{3, 4}, "Step 1: deduction"}, 1);
    root->add_child(step1);
    auto step2 = std::make_shared<ThoughtNode>(step1, ThoughtStep{{5, 6}, "Conclusion"}, 2);
    step1->add_child(step2);

    auto tokens = step2->accumulated_tokens();
    std::vector<int64_t> expected = {1, 2, 3, 4, 5, 6};
    ASSERT_EQ(tokens.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        ASSERT_EQ(tokens[i], expected[i]);
    }

    std::string trace = step2->full_reasoning_trace();
    ASSERT_TRUE(trace.find("Step 1") != std::string::npos);
    ASSERT_TRUE(trace.find("Conclusion") != std::string::npos);
}

AIC_TEST(SelfVerifierTest, ContradictionDetection) {
    SelfVerifier verifier;

    ThoughtStep good_step;
    good_step.tokens = {10, 11, 12};
    good_step.text = "First, we observe that x is positive.";
    auto res_good = verifier.verify_step({}, {}, good_step);
    ASSERT_TRUE(res_good.is_valid);
    ASSERT_FALSE(res_good.is_contradiction);
    ASSERT_TRUE(res_good.score > 0.4f);

    ThoughtStep bad_step;
    bad_step.tokens = {13, 14};
    bad_step.text = "This leads to a contradiction since 1 = 0.";
    auto res_bad = verifier.verify_step({}, {}, bad_step);
    ASSERT_FALSE(res_bad.is_valid);
    ASSERT_TRUE(res_bad.is_contradiction);
    ASSERT_NEAR(res_bad.score, 0.05f, 1e-4f);
}

AIC_TEST(SelfVerifierTest, RepetitionPenalty) {
    SelfVerifier verifier;

    ThoughtStep repetitive_step;
    repetitive_step.tokens = {7, 7, 7, 7, 7}; // 5 consecutive identical tokens
    repetitive_step.text = "word word word word word";
    repetitive_step.score = 0.6f;

    auto res = verifier.verify_step({}, {}, repetitive_step);
    // Repetition penalty should reduce the score
    ASSERT_TRUE(res.score < 0.6f);
}

AIC_TEST(SelfVerifierTest, DeductionBonusAndConclusion) {
    SelfVerifier verifier;

    ThoughtStep step;
    step.tokens = {20, 21, 22};
    step.text = "Therefore the answer is 42.";
    step.score = 0.5f;

    auto res = verifier.verify_step({}, {}, step);
    ASSERT_TRUE(res.is_valid);
    ASSERT_TRUE(res.is_conclusive);
    // Deduction bonus should increase score above base 0.5
    ASSERT_TRUE(res.score > 0.5f);
}

AIC_TEST(ReasoningEngineTest, TreeOfThoughtsSearch) {
    data::Tokenizer tokenizer = data::Tokenizer::create_default();

    TransformerLMConfig cfg;
    cfg.vocab_size = static_cast<int64_t>(tokenizer.vocab_size());
    cfg.hidden_dim = 32;
    cfg.num_layers = 1;
    cfg.num_heads = 2;
    cfg.intermediate_dim = 64;
    cfg.max_seq_len = 64;

    auto model = std::make_shared<TransformerLM>(cfg, Device::cpu());

    ReasoningConfig r_cfg;
    r_cfg.max_search_iterations = 6;
    r_cfg.beam_width = 2;
    r_cfg.max_depth = 3;
    r_cfg.max_tokens_per_step = 6;

    ReasoningSearchEngine engine(model, tokenizer, r_cfg);

    auto trace = engine.reason("Let x = 5.");

    ASSERT_TRUE(trace.nodes_explored > 1);
    ASSERT_TRUE(trace.search_time_ms >= 0.0);
    ASSERT_FALSE(trace.prompt.empty());
    ASSERT_TRUE(trace.cumulative_confidence >= 0.0f);

    std::string report = trace.to_string();
    ASSERT_FALSE(report.empty());
    ASSERT_TRUE(report.find("AUTONOMOUS REASONING TRACE") != std::string::npos);
}

AIC_TEST(ReasoningEngineTest, PruningAndBacktracking) {
    ThoughtStep root_step{{1, 2}, "Root", 0.0f, 1.0f, false, true};
    auto root = std::make_shared<ThoughtNode>(std::weak_ptr<ThoughtNode>{}, root_step, 0);

    // Child 1: Valid thought
    ThoughtStep s1{{3, 4}, "Step 1", 0.0f, 0.7f, false, true};
    auto child1 = std::make_shared<ThoughtNode>(root, s1, 1, 0.7f);
    root->add_child(child1);

    // Child 2: Invalid / pruned thought (contradiction)
    ThoughtStep s2{{5, 6}, "Contradiction 1 = 0", 0.0f, 0.05f, false, /*is_valid=*/false};
    auto child2 = std::make_shared<ThoughtNode>(root, s2, 1, 0.05f);
    root->add_child(child2);

    // select_best_child should NEVER select the pruned child2
    auto selected = root->select_best_child();
    ASSERT_TRUE(selected == child1);
    ASSERT_FALSE(selected == child2);
}
