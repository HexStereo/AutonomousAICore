// src/autonomous/react_engine.cpp
#include "autonomous/react_engine.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <sstream>

namespace aic::autonomous {

const char* step_type_name(StepType type) noexcept {
    switch (type) {
        case StepType::Thought:     return "Thought";
        case StepType::Action:      return "Action";
        case StepType::Observation: return "Observation";
        case StepType::FinalAnswer: return "FinalAnswer";
        case StepType::Error:       return "Error";
    }
    return "Unknown";
}

std::string AgentTrajectory::format_transcript() const {
    std::ostringstream oss;
    oss << "=== AGENT TRAJECTORY ===\n"
        << "Goal: " << goal << "\n"
        << "Status: " << (success ? "SUCCESS" : "FAILED / INCOMPLETE") << "\n"
        << "Total Tool Calls: " << total_tool_calls << "\n"
        << "Total Elapsed Time: " << total_time_ms << " ms\n"
        << "------------------------\n";

    for (const auto& step : steps) {
        oss << "[" << step_type_name(step.type) << " #" << step.step_number << "] "
            << "(" << step.elapsed_ms << " ms)\n"
            << step.content << "\n";
        if (step.type == StepType::Action && !step.tool_call.tool_name.empty()) {
            oss << "  -> Invoking: " << step.tool_call.tool_name << "\n";
            for (const auto& [k, v] : step.tool_call.arguments) {
                oss << "     " << k << " = " << v << "\n";
            }
        } else if (step.type == StepType::Observation) {
            oss << "  -> Result (" << (step.tool_result.success ? "OK" : "FAILED") << "): "
                << (step.tool_result.success ? step.tool_result.output : step.tool_result.error_message) << "\n";
        }
        oss << "\n";
    }

    if (!final_answer.empty()) {
        oss << "Final Answer:\n" << final_answer << "\n";
    }
    oss << "========================\n";
    return oss.str();
}

ReActEngine::ReActEngine(
    std::shared_ptr<ToolRegistry> registry,
    std::shared_ptr<ExecutionSandbox> sandbox,
    std::shared_ptr<models::TransformerLM> model,
    std::shared_ptr<data::Tokenizer> tokenizer
) : registry_(std::move(registry)),
    sandbox_(std::move(sandbox)),
    model_(std::move(model)),
    tokenizer_(std::move(tokenizer)) {
    AIC_CHECK(registry_ != nullptr, InvalidArgument, "ToolRegistry cannot be null in ReActEngine");
}

namespace {

// Helper to trim whitespace from both ends
std::string trim_str(const std::string& s) {
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) start++;
    size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) end--;
    return s.substr(start, end - start);
}

// Simple JSON key-value extractor for flat argument objects {"k1": "v1", "k2": "v2"}
std::unordered_map<std::string, std::string> parse_flat_json_args(const std::string& json_str) {
    std::unordered_map<std::string, std::string> args;
    size_t pos = json_str.find('{');
    if (pos == std::string::npos) return args;

    while (pos < json_str.size()) {
        size_t key_quote_open = json_str.find('"', pos);
        if (key_quote_open == std::string::npos) break;
        size_t key_quote_close = json_str.find('"', key_quote_open + 1);
        if (key_quote_close == std::string::npos) break;

        std::string key = json_str.substr(key_quote_open + 1, key_quote_close - key_quote_open - 1);

        size_t colon = json_str.find(':', key_quote_close + 1);
        if (colon == std::string::npos) break;

        // Skip whitespace after colon
        size_t val_start = colon + 1;
        while (val_start < json_str.size() && std::isspace(static_cast<unsigned char>(json_str[val_start]))) {
            val_start++;
        }
        if (val_start >= json_str.size()) break;

        std::string val;
        size_t next_pos = val_start;
        if (json_str[val_start] == '"') {
            // Quoted string value
            size_t val_quote_close = json_str.find('"', val_start + 1);
            if (val_quote_close != std::string::npos) {
                val = json_str.substr(val_start + 1, val_quote_close - val_start - 1);
                next_pos = val_quote_close + 1;
            }
        } else {
            // Numeric, boolean, or bare value until comma or closing brace
            size_t val_end = json_str.find_first_of(",}", val_start);
            if (val_end != std::string::npos) {
                val = trim_str(json_str.substr(val_start, val_end - val_start));
                next_pos = val_end;
            } else {
                val = trim_str(json_str.substr(val_start));
                next_pos = json_str.size();
            }
        }

        args[key] = val;
        pos = next_pos;
        if (pos < json_str.size() && json_str[pos] == '}') break;
    }

    return args;
}

} // anonymous namespace

ToolCall ReActEngine::parse_tool_call(const std::string& text) {
    ToolCall call;

    // 1. Try standard ReAct lines:
    // Action: <tool_name>
    // Action Input: <json_args>
    size_t action_idx = text.find("Action:");
    if (action_idx != std::string::npos) {
        size_t line_end = text.find('\n', action_idx);
        std::string act_line = (line_end == std::string::npos)
            ? text.substr(action_idx + 7)
            : text.substr(action_idx + 7, line_end - (action_idx + 7));
        call.tool_name = trim_str(act_line);

        size_t input_idx = text.find("Action Input:", action_idx);
        if (input_idx != std::string::npos) {
            size_t input_end = text.find('\n', input_idx);
            std::string input_line = (input_end == std::string::npos)
                ? text.substr(input_idx + 13)
                : text.substr(input_idx + 13, input_end - (input_idx + 13));
            call.raw_arguments_json = trim_str(input_line);
            call.arguments = parse_flat_json_args(call.raw_arguments_json);
        }
        return call;
    }

    // 2. Try JSON function calling format: {"tool": "name", "arguments": {...}}
    size_t tool_key = text.find("\"tool\"");
    if (tool_key != std::string::npos) {
        auto args = parse_flat_json_args(text);
        if (args.find("tool") != args.end()) {
            call.tool_name = args.at("tool");
            // If arguments were sub-parsed or remaining:
            size_t args_pos = text.find("\"arguments\"");
            if (args_pos != std::string::npos) {
                size_t brace_open = text.find('{', args_pos);
                if (brace_open != std::string::npos) {
                    size_t brace_close = text.find('}', brace_open);
                    if (brace_close != std::string::npos) {
                        call.raw_arguments_json = text.substr(brace_open, brace_close - brace_open + 1);
                        call.arguments = parse_flat_json_args(call.raw_arguments_json);
                    }
                }
            }
        }
        return call;
    }

    return call;
}

bool ReActEngine::is_final_answer(const std::string& text, std::string& out_answer) {
    size_t idx = text.find("Final Answer:");
    if (idx != std::string::npos) {
        out_answer = trim_str(text.substr(idx + 13));
        return true;
    }

    idx = text.find("Final answer:");
    if (idx != std::string::npos) {
        out_answer = trim_str(text.substr(idx + 13));
        return true;
    }

    idx = text.find("\"final_answer\"");
    if (idx != std::string::npos) {
        auto args = parse_flat_json_args(text);
        if (args.find("final_answer") != args.end()) {
            out_answer = args.at("final_answer");
            return true;
        }
    }

    return false;
}

std::string ReActEngine::build_prompt(const std::string& goal, const std::vector<AgentStep>& previous_steps) const {
    std::ostringstream oss;
    oss << registry_->format_system_prompt() << "\n\n"
        << "User Objective: " << goal << "\n\n";

    for (const auto& step : previous_steps) {
        if (step.type == StepType::Thought) {
            oss << "Thought: " << step.content << "\n";
        } else if (step.type == StepType::Action) {
            oss << "Action: " << step.tool_call.tool_name << "\n"
                << "Action Input: " << step.tool_call.raw_arguments_json << "\n";
        } else if (step.type == StepType::Observation) {
            oss << "Observation: "
                << (step.tool_result.success ? step.tool_result.output : step.tool_result.error_message) << "\n";
        }
    }

    return oss.str();
}

AgentStep ReActEngine::execute_tool_step(int step_num, const ToolCall& call) {
    AgentStep step;
    step.step_number = step_num;
    step.tool_call = call;

    if (call.tool_name.empty()) {
        step.type = StepType::Error;
        step.content = "No action specified in tool call";
        step.tool_result = ToolResult::fail(step.content);
        return step;
    }

    auto tool = registry_->get_tool(call.tool_name);
    if (!tool) {
        step.type = StepType::Observation;
        step.tool_result = ToolResult::fail("Error: Tool '" + call.tool_name + "' is not registered.");
        step.content = step.tool_result.error_message;
        step.elapsed_ms = 0.1;
        return step;
    }

    step.type = StepType::Observation;
    step.tool_result = tool->execute(call.arguments);
    step.elapsed_ms = step.tool_result.execution_time_ms;
    step.content = step.tool_result.success ? step.tool_result.output : step.tool_result.error_message;
    return step;
}

AgentTrajectory ReActEngine::run(const std::string& goal, const ReActConfig& config) {
    auto t_start = std::chrono::high_resolution_clock::now();

    AgentTrajectory traj;
    traj.goal = goal;

    for (int step_idx = 1; step_idx <= config.max_steps; ++step_idx) {
        auto t_step0 = std::chrono::high_resolution_clock::now();

        std::string current_prompt = build_prompt(goal, traj.steps);

        std::string generated_text;
        if (model_ && tokenizer_) {
            // Encode prompt into tokens
            std::vector<int64_t> prompt_toks = tokenizer_->encode(current_prompt);

            std::vector<int64_t> gen_toks = model_->generate(
                prompt_toks,
                config.max_tokens_per_step,
                config.temperature,
                config.top_k,
                config.top_p,
                tokenizer_->eos_token_id()
            );

            generated_text = tokenizer_->decode(gen_toks);
        } else {
            // When running without model weights (e.g. testing orchestration loop)
            generated_text = "Final Answer: Goal reached successfully.";
        }

        auto t_step1 = std::chrono::high_resolution_clock::now();
        double gen_ms = std::chrono::duration<double, std::milli>(t_step1 - t_step0).count();

        // 1. Check if model produced final answer
        std::string final_ans;
        if (is_final_answer(generated_text, final_ans)) {
            AgentStep thought_step;
            thought_step.type = StepType::FinalAnswer;
            thought_step.step_number = step_idx;
            thought_step.content = final_ans;
            thought_step.elapsed_ms = gen_ms;
            traj.steps.push_back(thought_step);

            traj.final_answer = final_ans;
            traj.success = true;
            break;
        }

        // 2. Parse tool call action
        ToolCall call = parse_tool_call(generated_text);
        if (!call.tool_name.empty()) {
            // Record Action Step
            AgentStep act_step;
            act_step.type = StepType::Action;
            act_step.step_number = step_idx;
            act_step.content = generated_text;
            act_step.tool_call = call;
            act_step.elapsed_ms = gen_ms;
            traj.steps.push_back(act_step);

            // Execute tool in sandbox / registry
            AgentStep obs_step = execute_tool_step(step_idx, call);
            traj.steps.push_back(obs_step);
            traj.total_tool_calls++;

            if (!obs_step.tool_result.success && config.stop_on_first_error) {
                traj.success = false;
                break;
            }
        } else {
            // Plain thought step without tool call
            AgentStep thought_step;
            thought_step.type = StepType::Thought;
            thought_step.step_number = step_idx;
            thought_step.content = generated_text;
            thought_step.elapsed_ms = gen_ms;
            traj.steps.push_back(thought_step);
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    traj.total_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    return traj;
}

} // namespace aic::autonomous
