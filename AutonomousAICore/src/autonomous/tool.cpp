// src/autonomous/tool.cpp
#include "autonomous/tool.hpp"
#include "runtime/cpu_info.hpp"
#include "runtime/cuda_info.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>
#include <stack>
#include <stdexcept>

namespace aic::autonomous {

Tool::Tool(
    std::string name,
    std::string description,
    std::vector<ToolParam> params,
    Handler handler
) : name_(std::move(name)),
    description_(std::move(description)),
    params_(std::move(params)),
    handler_(std::move(handler)) {}

inference::JSONSchema Tool::to_json_schema() const {
    inference::JSONSchema schema;
    for (const auto& p : params_) {
        inference::JSONPropertyType type = inference::JSONPropertyType::String;
        switch (p.type) {
            case ToolParamType::String:  type = inference::JSONPropertyType::String; break;
            case ToolParamType::Integer: type = inference::JSONPropertyType::Integer; break;
            case ToolParamType::Number:  type = inference::JSONPropertyType::Number; break;
            case ToolParamType::Boolean: type = inference::JSONPropertyType::Boolean; break;
        }
        schema.add_property(p.name, type, p.required);
    }
    return schema;
}

std::string Tool::format_doc() const {
    std::ostringstream oss;
    oss << "Tool: " << name_ << "\n"
        << "Description: " << description_ << "\n"
        << "Parameters:\n";
    if (params_.empty()) {
        oss << "  (none)\n";
    } else {
        for (const auto& p : params_) {
            oss << "  - " << p.name << " (";
            switch (p.type) {
                case ToolParamType::String:  oss << "string"; break;
                case ToolParamType::Integer: oss << "integer"; break;
                case ToolParamType::Number:  oss << "number"; break;
                case ToolParamType::Boolean: oss << "boolean"; break;
            }
            oss << ", " << (p.required ? "required" : "optional") << "): "
                << p.description << "\n";
        }
    }
    return oss.str();
}

ToolResult Tool::execute(const std::unordered_map<std::string, std::string>& args) const {
    auto t0 = std::chrono::high_resolution_clock::now();
    try {
        // Validate required parameters
        for (const auto& p : params_) {
            if (p.required && args.find(p.name) == args.end()) {
                auto t1 = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                return ToolResult::fail("Missing required parameter: " + p.name, ms);
            }
        }
        ToolResult res = handler_(args);
        auto t1 = std::chrono::high_resolution_clock::now();
        res.execution_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return res;
    } catch (const std::exception& e) {
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return ToolResult::fail(std::string("Exception: ") + e.what(), ms);
    } catch (...) {
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return ToolResult::fail("Unknown exception occurred during tool execution", ms);
    }
}

void ToolRegistry::register_tool(std::shared_ptr<Tool> tool) {
    AIC_CHECK(tool != nullptr, InvalidArgument, "Cannot register null tool");
    std::string n = tool->name();
    if (tools_.find(n) == tools_.end()) {
        tool_order_.push_back(n);
    }
    tools_[n] = std::move(tool);
}

std::shared_ptr<Tool> ToolRegistry::get_tool(const std::string& name) const {
    auto it = tools_.find(name);
    if (it != tools_.end()) return it->second;
    return nullptr;
}

bool ToolRegistry::has_tool(const std::string& name) const noexcept {
    return tools_.find(name) != tools_.end();
}

std::vector<std::string> ToolRegistry::list_tools() const {
    return tool_order_;
}

std::string ToolRegistry::format_system_prompt() const {
    std::ostringstream oss;
    oss << "You are an autonomous AI problem-solving agent with access to the following tools:\n\n";
    for (const auto& name : tool_order_) {
        const auto& t = tools_.at(name);
        oss << t->format_doc() << "\n";
    }
    oss << "Use the following ReAct format strictly:\n"
        << "Thought: <reasoning about what to do next>\n"
        << "Action: <tool_name>\n"
        << "Action Input: {\"param\": \"value\"}\n"
        << "Observation: <tool output will be provided here>\n"
        << "... (Thought/Action/Action Input/Observation can repeat as needed)\n"
        << "Thought: I now have the final answer\n"
        << "Final Answer: <concise answer to user objective>\n";
    return oss.str();
}

inference::JSONSchema ToolRegistry::generate_combined_schema() const {
    inference::JSONSchema schema;
    schema.add_property("tool", inference::JSONPropertyType::String, true);
    schema.add_property("arguments", inference::JSONPropertyType::String, true);
    return schema;
}

// ----------------------------------------------------------------------------
// Built-in Standard Tool: Safe Calculator
// ----------------------------------------------------------------------------
namespace {

// Shunting-yard and evaluation engine for arithmetic expressions
struct ExprParser {
    std::string s;
    size_t pos = 0;

    void skip_whitespace() {
        while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) pos++;
    }

    double parse_number() {
        skip_whitespace();
        size_t start = pos;
        if (pos < s.size() && (s[pos] == '+' || s[pos] == '-')) pos++;
        while (pos < s.size() && (std::isdigit(static_cast<unsigned char>(s[pos])) || s[pos] == '.')) pos++;
        if (pos == start) throw std::runtime_error("Expected number");
        return std::stod(s.substr(start, pos - start));
    }

    double parse_primary() {
        skip_whitespace();
        if (pos >= s.size()) throw std::runtime_error("Unexpected end of expression");

        if (s[pos] == '(') {
            pos++;
            double val = parse_expr();
            skip_whitespace();
            if (pos >= s.size() || s[pos] != ')') throw std::runtime_error("Mismatched parentheses");
            pos++;
            return val;
        }

        // Functions like sqrt, abs, round, floor, ceil
        if (std::isalpha(static_cast<unsigned char>(s[pos]))) {
            size_t start = pos;
            while (pos < s.size() && std::isalpha(static_cast<unsigned char>(s[pos]))) pos++;
            std::string func = s.substr(start, pos - start);
            skip_whitespace();
            if (pos >= s.size() || s[pos] != '(') throw std::runtime_error("Expected '(' after function " + func);
            pos++;
            double arg = parse_expr();
            skip_whitespace();
            if (pos >= s.size() || s[pos] != ')') throw std::runtime_error("Mismatched ')' after function " + func);
            pos++;

            if (func == "sqrt") {
                if (arg < 0.0) throw std::runtime_error("sqrt of negative number");
                return std::sqrt(arg);
            } else if (func == "abs") {
                return std::abs(arg);
            } else if (func == "round") {
                return std::round(arg);
            } else if (func == "floor") {
                return std::floor(arg);
            } else if (func == "ceil") {
                return std::ceil(arg);
            } else {
                throw std::runtime_error("Unknown function: " + func);
            }
        }

        return parse_number();
    }

    double parse_power() {
        double base = parse_primary();
        skip_whitespace();
        if (pos < s.size() && s[pos] == '^') {
            pos++;
            double exp = parse_power(); // right associative
            return std::pow(base, exp);
        }
        return base;
    }

    double parse_term() {
        double val = parse_power();
        skip_whitespace();
        while (pos < s.size()) {
            char op = s[pos];
            if (op == '*' || op == '/' || op == '%') {
                pos++;
                double next_val = parse_power();
                if (op == '*') val *= next_val;
                else if (op == '/') {
                    if (std::abs(next_val) < 1e-12) throw std::runtime_error("Division by zero");
                    val /= next_val;
                } else if (op == '%') {
                    if (std::abs(next_val) < 1e-12) throw std::runtime_error("Modulo by zero");
                    val = std::fmod(val, next_val);
                }
                skip_whitespace();
            } else {
                break;
            }
        }
        return val;
    }

    double parse_expr() {
        skip_whitespace();
        double val = 0.0;
        char lead_op = '+';
        if (pos < s.size() && (s[pos] == '+' || s[pos] == '-')) {
            lead_op = s[pos++];
        }
        val = parse_term();
        if (lead_op == '-') val = -val;

        skip_whitespace();
        while (pos < s.size()) {
            char op = s[pos];
            if (op == '+' || op == '-') {
                pos++;
                double next_val = parse_term();
                if (op == '+') val += next_val;
                else val -= next_val;
                skip_whitespace();
            } else {
                break;
            }
        }
        return val;
    }
};

} // anonymous namespace

std::shared_ptr<Tool> ToolRegistry::create_calculator_tool() {
    std::vector<ToolParam> params = {
        {"expression", ToolParamType::String, "The mathematical expression to evaluate (e.g., '14 * 15 + sqrt(144)')", true}
    };

    auto handler = [](const std::unordered_map<std::string, std::string>& args) -> ToolResult {
        auto it = args.find("expression");
        if (it == args.end() || it->second.empty()) {
            return ToolResult::fail("Parameter 'expression' is required");
        }

        try {
            ExprParser parser{it->second, 0};
            double res = parser.parse_expr();
            parser.skip_whitespace();
            if (parser.pos < parser.s.size()) {
                return ToolResult::fail("Syntax error near: " + parser.s.substr(parser.pos));
            }

            // Clean format (omit trailing zeros if whole number)
            std::ostringstream oss;
            if (std::abs(res - std::round(res)) < 1e-9) {
                oss << static_cast<int64_t>(std::round(res));
            } else {
                oss << res;
            }
            return ToolResult::ok(oss.str());
        } catch (const std::exception& e) {
            return ToolResult::fail(std::string("Calculation error: ") + e.what());
        }
    };

    return std::make_shared<Tool>(
        "calculator",
        "Evaluates mathematical expressions including +, -, *, /, %, ^, sqrt, abs, round, floor, ceil",
        std::move(params),
        std::move(handler)
    );
}

// ----------------------------------------------------------------------------
// Built-in Standard Tool: System Info
// ----------------------------------------------------------------------------
std::shared_ptr<Tool> ToolRegistry::create_system_info_tool() {
    std::vector<ToolParam> params = {};

    auto handler = [](const std::unordered_map<std::string, std::string>&) -> ToolResult {
        std::ostringstream oss;
        const auto& cpu = get_cpu_features();
        oss << "Platform: Windows x64\n"
            << "CPU Cores: " << cpu.logical_cores << " threads\n"
            << "AVX2/FMA Supported: " << (cpu.avx2 ? "Yes" : "No") << "\n";

#if AIC_WITH_CUDA
        if (is_cuda_available()) {
            auto gpu = get_gpu_properties(0);
            oss << "CUDA sm_120 Acceleration: Enabled\n"
                << "GPU Device: " << gpu.name << "\n"
                << "Total VRAM: " << (gpu.total_vram_bytes / (1024 * 1024)) << " MB\n";
        } else {
            oss << "CUDA sm_120 Acceleration: No CUDA GPU available\n";
        }
#else
        oss << "CUDA sm_120 Acceleration: Disabled (CPU Native Mode)\n";
#endif
        return ToolResult::ok(oss.str());
    };

    return std::make_shared<Tool>(
        "system_info",
        "Returns operating system, CPU hardware capabilities, and GPU VRAM statistics",
        std::move(params),
        std::move(handler)
    );
}

// ----------------------------------------------------------------------------
// Built-in Standard Tool: String Transform
// ----------------------------------------------------------------------------
std::shared_ptr<Tool> ToolRegistry::create_string_transform_tool() {
    std::vector<ToolParam> params = {
        {"text", ToolParamType::String, "The input string to process", true},
        {"operation", ToolParamType::String, "Transform operation: upper, lower, reverse, length, words", true}
    };

    auto handler = [](const std::unordered_map<std::string, std::string>& args) -> ToolResult {
        auto it_text = args.find("text");
        auto it_op = args.find("operation");
        if (it_text == args.end()) return ToolResult::fail("Missing 'text' parameter");
        if (it_op == args.end()) return ToolResult::fail("Missing 'operation' parameter");

        std::string text = it_text->second;
        std::string op = it_op->second;
        std::transform(op.begin(), op.end(), op.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

        if (op == "upper") {
            std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
            return ToolResult::ok(text);
        } else if (op == "lower") {
            std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return ToolResult::ok(text);
        } else if (op == "reverse") {
            std::reverse(text.begin(), text.end());
            return ToolResult::ok(text);
        } else if (op == "length") {
            return ToolResult::ok(std::to_string(text.size()));
        } else if (op == "words") {
            std::istringstream iss(text);
            std::string word;
            size_t count = 0;
            while (iss >> word) count++;
            return ToolResult::ok(std::to_string(count));
        } else {
            return ToolResult::fail("Unknown operation: " + op + " (valid: upper, lower, reverse, length, words)");
        }
    };

    return std::make_shared<Tool>(
        "string_transform",
        "Performs string manipulations: upper, lower, reverse, length, word count",
        std::move(params),
        std::move(handler)
    );
}

} // namespace aic::autonomous
