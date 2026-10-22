// include/autonomous/tool.hpp
// Autonomous Function & Tool Calling Engine (Aşama 24)
// Provides structured schema registration, argument validation, and invocation.
#pragma once

#include "core/error.hpp"
#include "inference/grammar.hpp"

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace aic::autonomous {

enum class ToolParamType {
    String,
    Integer,
    Number,
    Boolean
};

struct ToolParam {
    std::string name;
    ToolParamType type = ToolParamType::String;
    std::string description;
    bool required = true;
    std::string default_value = "";
};

struct ToolCall {
    std::string tool_name;
    std::string call_id;
    std::unordered_map<std::string, std::string> arguments;
    std::string raw_arguments_json;
};

struct ToolResult {
    bool success = true;
    std::string output;
    std::string error_message;
    double execution_time_ms = 0.0;

    static ToolResult ok(std::string out, double ms = 0.0) {
        return ToolResult{true, std::move(out), "", ms};
    }

    static ToolResult fail(std::string err, double ms = 0.0) {
        return ToolResult{false, "", std::move(err), ms};
    }
};

class Tool {
public:
    using Handler = std::function<ToolResult(const std::unordered_map<std::string, std::string>&)>;

    Tool(
        std::string name,
        std::string description,
        std::vector<ToolParam> params,
        Handler handler
    );

    const std::string& name() const noexcept { return name_; }
    const std::string& description() const noexcept { return description_; }
    const std::vector<ToolParam>& params() const noexcept { return params_; }

    inference::JSONSchema to_json_schema() const;
    std::string format_doc() const;

    ToolResult execute(const std::unordered_map<std::string, std::string>& args) const;

private:
    std::string name_;
    std::string description_;
    std::vector<ToolParam> params_;
    Handler handler_;
};

class ToolRegistry {
public:
    ToolRegistry() = default;

    void register_tool(std::shared_ptr<Tool> tool);
    std::shared_ptr<Tool> get_tool(const std::string& name) const;
    bool has_tool(const std::string& name) const noexcept;
    std::vector<std::string> list_tools() const;
    size_t size() const noexcept { return tools_.size(); }

    std::string format_system_prompt() const;
    inference::JSONSchema generate_combined_schema() const;

    // Built-in standard tool factories
    static std::shared_ptr<Tool> create_calculator_tool();
    static std::shared_ptr<Tool> create_system_info_tool();
    static std::shared_ptr<Tool> create_string_transform_tool();

private:
    std::unordered_map<std::string, std::shared_ptr<Tool>> tools_;
    std::vector<std::string> tool_order_;
};

} // namespace aic::autonomous
