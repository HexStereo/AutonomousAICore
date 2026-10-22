// include/autonomous/sandbox.hpp
// Sandboxed Execution & File I/O Environment (Aşama 24)
// Provides isolated, canonicalized filesystem operations and command execution.
#pragma once

#include "autonomous/tool.hpp"
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace aic::autonomous {

struct SandboxConfig {
    std::string root_directory = "sandbox_workspace";
    size_t max_file_size_bytes = 10 * 1024 * 1024; // 10 MB
    size_t max_output_chars = 4096;
    int execution_timeout_sec = 5;
    bool read_only = false;
    bool allow_subprocesses = true;
};

class ExecutionSandbox : public std::enable_shared_from_this<ExecutionSandbox> {
public:
    explicit ExecutionSandbox(SandboxConfig config = {});
    ~ExecutionSandbox() = default;

    const SandboxConfig& config() const noexcept { return config_; }
    const std::filesystem::path& root_path() const noexcept { return root_path_; }

    // Filesystem Sandboxed Primitives
    ToolResult write_file(const std::string& relative_path, const std::string& content);
    ToolResult read_file(const std::string& relative_path);
    ToolResult append_file(const std::string& relative_path, const std::string& content);
    ToolResult list_files(const std::string& sub_dir = "");
    ToolResult file_exists(const std::string& relative_path);
    ToolResult delete_file(const std::string& relative_path);

    // Command Execution Sandbox
    ToolResult execute_command(const std::string& command);

    // Tool creation helpers wrapping this sandbox
    std::shared_ptr<Tool> create_file_writer_tool();
    std::shared_ptr<Tool> create_file_reader_tool();
    std::shared_ptr<Tool> create_file_lister_tool();
    std::shared_ptr<Tool> create_shell_tool();

    // Reset / cleanup sandbox workspace
    void reset_workspace();

private:
    std::filesystem::path resolve_safe_path(const std::string& relative_path) const;

    SandboxConfig config_;
    std::filesystem::path root_path_;
};

} // namespace aic::autonomous
