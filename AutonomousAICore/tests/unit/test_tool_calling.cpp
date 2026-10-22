// tests/unit/test_tool_calling.cpp
// Unit tests for Autonomous Tool Calling & ReAct Sandbox Engine (Aşama 24)
#include "tests/test_framework.hpp"
#include "autonomous/tool.hpp"
#include "autonomous/sandbox.hpp"
#include "autonomous/react_engine.hpp"
#include "cuda/cuda_tool_routing.cuh"
#include "runtime/cuda_info.hpp"
#include "core/device.hpp"

#include <cmath>
#include <memory>
#include <string>
#include <vector>

using namespace aic;
using namespace aic::autonomous;

AIC_TEST(ToolCallingTest, RegistrationParameterValidationAndSchema) {
    ToolRegistry registry;
    ASSERT_EQ(registry.size(), 0);

    // Register a custom mock tool
    std::vector<ToolParam> params = {
        {"city", ToolParamType::String, "Target city name", true},
        {"units", ToolParamType::String, "Units: celsius or fahrenheit", false, "celsius"}
    };

    auto mock_weather = std::make_shared<Tool>(
        "get_weather",
        "Fetches current weather for a specified city",
        params,
        [](const std::unordered_map<std::string, std::string>& args) -> ToolResult {
            std::string city = args.at("city");
            return ToolResult::ok("22 degrees in " + city);
        }
    );

    registry.register_tool(mock_weather);
    ASSERT_EQ(registry.size(), 1);
    ASSERT_TRUE(registry.has_tool("get_weather"));
    ASSERT_FALSE(registry.has_tool("non_existent"));

    // Verify documentation format
    std::string doc = mock_weather->format_doc();
    ASSERT_TRUE(doc.find("get_weather") != std::string::npos);
    ASSERT_TRUE(doc.find("Target city name") != std::string::npos);

    // Verify JSON Schema generation
    auto schema = mock_weather->to_json_schema();
    ASSERT_EQ(schema.properties().size(), 2);
    ASSERT_EQ(schema.properties()[0].name, "city");
    ASSERT_TRUE(schema.properties()[0].required);

    // Test successful execution with required parameter
    ToolResult r_ok = mock_weather->execute({{"city", "Istanbul"}});
    ASSERT_TRUE(r_ok.success);
    ASSERT_EQ(r_ok.output, "22 degrees in Istanbul");
    ASSERT_TRUE(r_ok.execution_time_ms >= 0.0);

    // Test failure when required parameter is missing
    ToolResult r_fail = mock_weather->execute({});
    ASSERT_FALSE(r_fail.success);
    ASSERT_TRUE(r_fail.error_message.find("Missing required parameter: city") != std::string::npos);
}

AIC_TEST(ToolCallingTest, CalculatorEvaluationMathAndFunctions) {
    auto calc = ToolRegistry::create_calculator_tool();
    ASSERT_TRUE(calc != nullptr);

    // Simple operations
    ToolResult r1 = calc->execute({{"expression", "14 * 15 + 10"}});
    ASSERT_TRUE(r1.success);
    ASSERT_EQ(r1.output, "220");

    // Parentheses and operator precedence
    ToolResult r2 = calc->execute({{"expression", "(20 - 5) * 4 / 2"}});
    ASSERT_TRUE(r2.success);
    ASSERT_EQ(r2.output, "30");

    // Power operator
    ToolResult r3 = calc->execute({{"expression", "2 ^ 10"}});
    ASSERT_TRUE(r3.success);
    ASSERT_EQ(r3.output, "1024");

    // Built-in functions: sqrt and round
    ToolResult r4 = calc->execute({{"expression", "sqrt(144) + round(3.7)"}});
    ASSERT_TRUE(r4.success);
    ASSERT_EQ(r4.output, "16");

    // Modulo operator
    ToolResult r5 = calc->execute({{"expression", "29 % 7"}});
    ASSERT_TRUE(r5.success);
    ASSERT_EQ(r5.output, "1");

    // Error handling: Division by zero
    ToolResult r_err1 = calc->execute({{"expression", "100 / 0"}});
    ASSERT_FALSE(r_err1.success);
    ASSERT_TRUE(r_err1.error_message.find("Division by zero") != std::string::npos);

    // Error handling: Invalid syntax
    ToolResult r_err2 = calc->execute({{"expression", "14 * + (("}});
    ASSERT_FALSE(r_err2.success);
}

AIC_TEST(ToolCallingTest, SystemInfoAndStringTransform) {
    // 1. System info tool
    auto sys_info = ToolRegistry::create_system_info_tool();
    ToolResult sys_res = sys_info->execute({});
    ASSERT_TRUE(sys_res.success);
    ASSERT_TRUE(sys_res.output.find("Platform: Windows") != std::string::npos);
    ASSERT_TRUE(sys_res.output.find("CPU Cores:") != std::string::npos);

    // 2. String transform tool
    auto str_tool = ToolRegistry::create_string_transform_tool();

    // Uppercase
    ToolResult r_up = str_tool->execute({{"text", "hello autonomous core"}, {"operation", "upper"}});
    ASSERT_TRUE(r_up.success);
    ASSERT_EQ(r_up.output, "HELLO AUTONOMOUS CORE");

    // Reverse
    ToolResult r_rev = str_tool->execute({{"text", "Blackwell"}, {"operation", "reverse"}});
    ASSERT_TRUE(r_rev.success);
    ASSERT_EQ(r_rev.output, "llewkcalB");

    // Word count
    ToolResult r_words = str_tool->execute({{"text", "one two three four five"}, {"operation", "words"}});
    ASSERT_TRUE(r_words.success);
    ASSERT_EQ(r_words.output, "5");

    // Length
    ToolResult r_len = str_tool->execute({{"text", "123456789"}, {"operation", "length"}});
    ASSERT_TRUE(r_len.success);
    ASSERT_EQ(r_len.output, "9");
}

AIC_TEST(ToolCallingTest, SandboxPathTraversalSecurityAndCanonicalization) {
    SandboxConfig cfg;
    cfg.root_directory = "test_sandbox_workspace";
    auto sandbox = std::make_shared<ExecutionSandbox>(cfg);
    sandbox->reset_workspace();

    // Legitimate write inside sandbox
    ToolResult r_write = sandbox->write_file("sub/file.txt", "safe content");
    ASSERT_TRUE(r_write.success);

    // Security Test 1: Relative traversal using .. to escape sandbox
    ToolResult r_trav1 = sandbox->write_file("../forbidden.txt", "malicious payload");
    ASSERT_FALSE(r_trav1.success);
    ASSERT_TRUE(r_trav1.error_message.find("Directory traversal forbidden") != std::string::npos);

    // Security Test 2: Deep traversal
    ToolResult r_trav2 = sandbox->read_file("sub/../../escape.txt");
    ASSERT_FALSE(r_trav2.success);
    ASSERT_TRUE(r_trav2.error_message.find("Directory traversal forbidden") != std::string::npos);

    // Security Test 3: Absolute path
    ToolResult r_abs = sandbox->write_file("C:/Windows/System32/evil.dll", "bad");
    ASSERT_FALSE(r_abs.success);
    ASSERT_TRUE(r_abs.error_message.find("Absolute paths are forbidden") != std::string::npos);

    // Cleanup
    sandbox->reset_workspace();
}

AIC_TEST(ToolCallingTest, SandboxIsolatedFileOperationsAndReset) {
    SandboxConfig cfg;
    cfg.root_directory = "test_sandbox_io";
    auto sandbox = std::make_shared<ExecutionSandbox>(cfg);
    sandbox->reset_workspace();

    // 1. Write file
    ToolResult w_res = sandbox->write_file("notes/todo.txt", "Task 1: Train LLM\n");
    ASSERT_TRUE(w_res.success);

    // 2. Append file
    ToolResult app_res = sandbox->append_file("notes/todo.txt", "Task 2: Evaluate Tools\n");
    ASSERT_TRUE(app_res.success);

    // 3. Read file
    ToolResult r_res = sandbox->read_file("notes/todo.txt");
    ASSERT_TRUE(r_res.success);
    ASSERT_EQ(r_res.output, "Task 1: Train LLM\nTask 2: Evaluate Tools\n");

    // 4. File exists check
    ToolResult ex_yes = sandbox->file_exists("notes/todo.txt");
    ASSERT_TRUE(ex_yes.success);
    ASSERT_EQ(ex_yes.output, "true");

    ToolResult ex_no = sandbox->file_exists("notes/ghost.txt");
    ASSERT_TRUE(ex_no.success);
    ASSERT_EQ(ex_no.output, "false");

    // 5. List files
    ToolResult l_res = sandbox->list_files("notes");
    ASSERT_TRUE(l_res.success);
    ASSERT_TRUE(l_res.output.find("todo.txt") != std::string::npos);

    // 6. Delete file
    ToolResult d_res = sandbox->delete_file("notes/todo.txt");
    ASSERT_TRUE(d_res.success);
    ASSERT_EQ(sandbox->file_exists("notes/todo.txt").output, "false");

    // Reset workspace
    sandbox->reset_workspace();
}

AIC_TEST(ToolCallingTest, ReActParsingTrajectoryAndMultiStepExecution) {
    auto registry = std::make_shared<ToolRegistry>();
    registry->register_tool(ToolRegistry::create_calculator_tool());
    registry->register_tool(ToolRegistry::create_string_transform_tool());

    SandboxConfig s_cfg;
    s_cfg.root_directory = "test_sandbox_react";
    auto sandbox = std::make_shared<ExecutionSandbox>(s_cfg);
    sandbox->reset_workspace();
    registry->register_tool(sandbox->create_file_writer_tool());
    registry->register_tool(sandbox->create_file_reader_tool());

    ReActEngine engine(registry, sandbox);

    // 1. Test ReAct line format parsing
    std::string text1 = "Thought: I need to calculate the area of circle\nAction: calculator\nAction Input: {\"expression\": \"3.14159 * 10 * 10\"}\n";
    ToolCall call1 = ReActEngine::parse_tool_call(text1);
    ASSERT_EQ(call1.tool_name, "calculator");
    ASSERT_EQ(call1.arguments["expression"], "3.14159 * 10 * 10");

    // 2. Test JSON format parsing
    std::string text2 = "{\"tool\": \"write_file\", \"arguments\": {\"path\": \"area.txt\", \"content\": \"314.159\"}}";
    ToolCall call2 = ReActEngine::parse_tool_call(text2);
    ASSERT_EQ(call2.tool_name, "write_file");
    ASSERT_EQ(call2.arguments["path"], "area.txt");
    ASSERT_EQ(call2.arguments["content"], "314.159");

    // 3. Test final answer detection
    std::string text3 = "Thought: The calculation is complete.\nFinal Answer: The area is 314.159 square units.";
    std::string ans;
    ASSERT_TRUE(ReActEngine::is_final_answer(text3, ans));
    ASSERT_EQ(ans, "The area is 314.159 square units.");

    // 4. Test multi-step trajectory execution
    AgentStep step1 = engine.execute_tool_step(1, call1);
    ASSERT_EQ(step1.type, StepType::Observation);
    ASSERT_TRUE(step1.tool_result.success);
    ASSERT_TRUE(step1.tool_result.output.find("314.159") != std::string::npos);

    AgentStep step2 = engine.execute_tool_step(2, call2);
    ASSERT_EQ(step2.type, StepType::Observation);
    ASSERT_TRUE(step2.tool_result.success);

    // Verify file was written
    ToolResult read_back = sandbox->read_file("area.txt");
    ASSERT_TRUE(read_back.success);
    ASSERT_EQ(read_back.output, "314.159");

    // 5. Test unknown tool observation
    ToolCall call_bad{"quantum_teleport", "1", {}};
    AgentStep step_bad = engine.execute_tool_step(3, call_bad);
    ASSERT_FALSE(step_bad.tool_result.success);
    ASSERT_TRUE(step_bad.tool_result.error_message.find("not registered") != std::string::npos);

    sandbox->reset_workspace();
}

AIC_TEST(ToolCallingTest, CudaAndCpuToolRoutingCosineSimilarity) {
    int num_tools = 4;
    int embed_dim = 16;

    // Synthetic query vector: e.g. pointing towards dimension 0 & 1
    std::vector<float> query_embed(embed_dim, 0.0f);
    query_embed[0] = 1.0f;
    query_embed[1] = 2.0f;

    // 4 tool vectors:
    // Tool 0: strongly aligned with query [1.0, 2.0, ...]
    // Tool 1: orthogonal [0.0, 0.0, 1.0, ...]
    // Tool 2: opposite [-1.0, -2.0, ...]
    // Tool 3: partial alignment [0.5, 0.5, ...]
    std::vector<float> tool_embeds(num_tools * embed_dim, 0.0f);

    tool_embeds[0 * embed_dim + 0] = 1.0f;
    tool_embeds[0 * embed_dim + 1] = 2.0f;

    tool_embeds[1 * embed_dim + 2] = 3.0f;

    tool_embeds[2 * embed_dim + 0] = -1.0f;
    tool_embeds[2 * embed_dim + 1] = -2.0f;

    tool_embeds[3 * embed_dim + 0] = 0.5f;
    tool_embeds[3 * embed_dim + 1] = 0.5f;

    // CPU Evaluation
    std::vector<float> cpu_scores(num_tools, 0.0f);
    cuda::cpu_tool_embedding_similarity(
        query_embed.data(),
        tool_embeds.data(),
        cpu_scores.data(),
        num_tools,
        embed_dim
    );

    ASSERT_NEAR(cpu_scores[0], 1.0f, 1e-4f);   // Perfect alignment
    ASSERT_NEAR(cpu_scores[1], 0.0f, 1e-4f);   // Orthogonal
    ASSERT_NEAR(cpu_scores[2], -1.0f, 1e-4f);  // Exact opposite
    ASSERT_TRUE(cpu_scores[3] > 0.0f);          // Positive partial
    ASSERT_TRUE(cpu_scores[3] < 1.0f);

    // Highest ranked tool must be Tool 0
    int best_tool = static_cast<int>(std::max_element(cpu_scores.begin(), cpu_scores.end()) - cpu_scores.begin());
    ASSERT_EQ(best_tool, 0);

#if AIC_WITH_CUDA
    if (aic::is_cuda_available()) {
        float* d_query = nullptr;
        float* d_tools = nullptr;
        float* d_scores = nullptr;

        cudaMalloc(&d_query, embed_dim * sizeof(float));
        cudaMalloc(&d_tools, num_tools * embed_dim * sizeof(float));
        cudaMalloc(&d_scores, num_tools * sizeof(float));

        cudaMemcpy(d_query, query_embed.data(), embed_dim * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_tools, tool_embeds.data(), num_tools * embed_dim * sizeof(float), cudaMemcpyHostToDevice);

        cuda::launch_tool_embedding_similarity(
            d_query,
            d_tools,
            d_scores,
            num_tools,
            embed_dim
        );
        cudaDeviceSynchronize();

        std::vector<float> gpu_scores(num_tools, 0.0f);
        cudaMemcpy(gpu_scores.data(), d_scores, num_tools * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_query);
        cudaFree(d_tools);
        cudaFree(d_scores);

        for (int i = 0; i < num_tools; ++i) {
            ASSERT_NEAR(gpu_scores[i], cpu_scores[i], 1e-4f);
        }
    }
#endif
}
