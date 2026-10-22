// include/autonomous/react_engine.hpp
// ReAct (Reasoning + Acting) Autonomous Loop Engine (Aşama 24)
// Coordinates multi-step Thought -> Action -> Observation -> Solution cycles.
#pragma once

#include "autonomous/sandbox.hpp"
#include "autonomous/tool.hpp"
#include "data/tokenizer.hpp"
#include "models/transformer_lm.hpp"

#include <memory>
#include <string>
#include <vector>

namespace aic::autonomous {

enum class StepType {
    Thought,
    Action,
    Observation,
    FinalAnswer,
    Error
};

const char* step_type_name(StepType type) noexcept;

inline std::ostream& operator<<(std::ostream& os, StepType type) {
    return os << step_type_name(type);
}

struct AgentStep {
    StepType type = StepType::Thought;
    int step_number = 0;
    std::string content;
    ToolCall tool_call;
    ToolResult tool_result;
    double elapsed_ms = 0.0;
};

struct AgentTrajectory {
    std::string goal;
    std::vector<AgentStep> steps;
    std::string final_answer;
    bool success = false;
    int total_tool_calls = 0;
    double total_time_ms = 0.0;

    std::string format_transcript() const;
};

struct ReActConfig {
    int max_steps = 8;
    bool enable_grammar_constraints = true;
    bool stop_on_first_error = false;
    float temperature = 0.2f;
    int top_k = 40;
    float top_p = 0.9f;
    int max_tokens_per_step = 256;
};

class ReActEngine {
public:
    ReActEngine(
        std::shared_ptr<ToolRegistry> registry,
        std::shared_ptr<ExecutionSandbox> sandbox = nullptr,
        std::shared_ptr<models::TransformerLM> model = nullptr,
        std::shared_ptr<data::Tokenizer> tokenizer = nullptr
    );

    const ToolRegistry& registry() const noexcept { return *registry_; }
    ToolRegistry& registry() noexcept { return *registry_; }

    std::shared_ptr<ExecutionSandbox> sandbox() const noexcept { return sandbox_; }

    // ReAct Autonomous Problem-Solving Loop
    AgentTrajectory run(const std::string& goal, const ReActConfig& config = {});

    // Parse model generated action string into ToolCall
    static ToolCall parse_tool_call(const std::string& text);

    // Detect if text contains a final answer
    static bool is_final_answer(const std::string& text, std::string& out_answer);

    // Prompt construction for ReAct sequence
    std::string build_prompt(const std::string& goal, const std::vector<AgentStep>& previous_steps) const;

    // Single step tool execution
    AgentStep execute_tool_step(int step_num, const ToolCall& call);

private:
    std::shared_ptr<ToolRegistry> registry_;
    std::shared_ptr<ExecutionSandbox> sandbox_;
    std::shared_ptr<models::TransformerLM> model_;
    std::shared_ptr<data::Tokenizer> tokenizer_;
};

} // namespace aic::autonomous
