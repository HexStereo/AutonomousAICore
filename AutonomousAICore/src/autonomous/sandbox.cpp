// src/autonomous/sandbox.cpp
#include "autonomous/sandbox.hpp"

#include <cstdio>
#include <fstream>
#include <sstream>

#if defined(_WIN32)
#define popen _popen
#define pclose _pclose
#endif

namespace aic::autonomous {

ExecutionSandbox::ExecutionSandbox(SandboxConfig config)
    : config_(std::move(config)) {
    root_path_ = std::filesystem::absolute(config_.root_directory);
    std::filesystem::create_directories(root_path_);
}

std::filesystem::path ExecutionSandbox::resolve_safe_path(const std::string& relative_path) const {
    if (relative_path.empty()) {
        throw std::runtime_error("Empty file path is invalid");
    }

    std::filesystem::path p(relative_path);
    // Disallow absolute drive/root paths
    if (p.is_absolute() || p.has_root_name() || p.has_root_directory()) {
        throw std::runtime_error("Absolute paths are forbidden inside sandbox: " + relative_path);
    }

    // Combine and check weakly canonical path
    std::filesystem::path combined = root_path_ / p;
    std::filesystem::path canonical = std::filesystem::weakly_canonical(combined);

    // Verify canonical path begins with root_path_
    std::string root_str = root_path_.lexically_normal().string();
    std::string can_str = canonical.lexically_normal().string();

    if (can_str.rfind(root_str, 0) != 0) {
        throw std::runtime_error("Directory traversal forbidden: attempt to escape sandbox: " + relative_path);
    }

    return canonical;
}

ToolResult ExecutionSandbox::write_file(const std::string& relative_path, const std::string& content) {
    if (config_.read_only) {
        return ToolResult::fail("Sandbox is in read-only mode");
    }
    if (content.size() > config_.max_file_size_bytes) {
        return ToolResult::fail("File content exceeds max limit of " + std::to_string(config_.max_file_size_bytes) + " bytes");
    }

    try {
        std::filesystem::path target = resolve_safe_path(relative_path);
        if (target.has_parent_path()) {
            std::filesystem::create_directories(target.parent_path());
        }

        std::ofstream ofs(target, std::ios::binary | std::ios::trunc);
        if (!ofs.is_open()) {
            return ToolResult::fail("Failed to open file for writing: " + relative_path);
        }
        ofs.write(content.data(), static_cast<std::streamsize>(content.size()));
        ofs.close();

        return ToolResult::ok("Successfully wrote " + std::to_string(content.size()) + " bytes to " + relative_path);
    } catch (const std::exception& e) {
        return ToolResult::fail(std::string("write_file error: ") + e.what());
    }
}

ToolResult ExecutionSandbox::read_file(const std::string& relative_path) {
    try {
        std::filesystem::path target = resolve_safe_path(relative_path);
        if (!std::filesystem::exists(target)) {
            return ToolResult::fail("File does not exist: " + relative_path);
        }
        if (std::filesystem::is_directory(target)) {
            return ToolResult::fail("Path is a directory, not a file: " + relative_path);
        }

        auto sz = std::filesystem::file_size(target);
        if (sz > config_.max_file_size_bytes) {
            return ToolResult::fail("File size (" + std::to_string(sz) + " bytes) exceeds max limit");
        }

        std::ifstream ifs(target, std::ios::binary);
        if (!ifs.is_open()) {
            return ToolResult::fail("Failed to open file for reading: " + relative_path);
        }

        std::string content(static_cast<size_t>(sz), '\0');
        ifs.read(content.data(), static_cast<std::streamsize>(sz));
        ifs.close();

        if (content.size() > config_.max_output_chars) {
            std::string truncated = content.substr(0, config_.max_output_chars);
            truncated += "\n[... truncated " + std::to_string(content.size() - config_.max_output_chars) + " chars ...]";
            return ToolResult::ok(truncated);
        }

        return ToolResult::ok(content);
    } catch (const std::exception& e) {
        return ToolResult::fail(std::string("read_file error: ") + e.what());
    }
}

ToolResult ExecutionSandbox::append_file(const std::string& relative_path, const std::string& content) {
    if (config_.read_only) {
        return ToolResult::fail("Sandbox is in read-only mode");
    }

    try {
        std::filesystem::path target = resolve_safe_path(relative_path);
        if (target.has_parent_path()) {
            std::filesystem::create_directories(target.parent_path());
        }

        std::ofstream ofs(target, std::ios::binary | std::ios::app);
        if (!ofs.is_open()) {
            return ToolResult::fail("Failed to open file for append: " + relative_path);
        }
        ofs.write(content.data(), static_cast<std::streamsize>(content.size()));
        ofs.close();

        return ToolResult::ok("Successfully appended " + std::to_string(content.size()) + " bytes to " + relative_path);
    } catch (const std::exception& e) {
        return ToolResult::fail(std::string("append_file error: ") + e.what());
    }
}

ToolResult ExecutionSandbox::list_files(const std::string& sub_dir) {
    try {
        std::filesystem::path target = sub_dir.empty() ? root_path_ : resolve_safe_path(sub_dir);
        if (!std::filesystem::exists(target)) {
            return ToolResult::fail("Directory does not exist: " + sub_dir);
        }
        if (!std::filesystem::is_directory(target)) {
            return ToolResult::fail("Path is not a directory: " + sub_dir);
        }

        std::ostringstream oss;
        oss << "Files in '" << (sub_dir.empty() ? "." : sub_dir) << "':\n";
        size_t count = 0;
        for (const auto& entry : std::filesystem::directory_iterator(target)) {
            count++;
            std::string name = entry.path().filename().string();
            if (entry.is_directory()) {
                oss << "  [DIR]  " << name << "/\n";
            } else {
                oss << "  [FILE] " << name << " (" << entry.file_size() << " bytes)\n";
            }
        }
        if (count == 0) {
            oss << "  (directory is empty)\n";
        }
        return ToolResult::ok(oss.str());
    } catch (const std::exception& e) {
        return ToolResult::fail(std::string("list_files error: ") + e.what());
    }
}

ToolResult ExecutionSandbox::file_exists(const std::string& relative_path) {
    try {
        std::filesystem::path target = resolve_safe_path(relative_path);
        bool exists = std::filesystem::exists(target);
        return ToolResult::ok(exists ? "true" : "false");
    } catch (const std::exception& e) {
        return ToolResult::fail(std::string("file_exists error: ") + e.what());
    }
}

ToolResult ExecutionSandbox::delete_file(const std::string& relative_path) {
    if (config_.read_only) {
        return ToolResult::fail("Sandbox is in read-only mode");
    }
    try {
        std::filesystem::path target = resolve_safe_path(relative_path);
        if (!std::filesystem::exists(target)) {
            return ToolResult::fail("File does not exist: " + relative_path);
        }
        bool removed = std::filesystem::remove(target);
        return removed ? ToolResult::ok("Deleted " + relative_path) : ToolResult::fail("Failed to delete " + relative_path);
    } catch (const std::exception& e) {
        return ToolResult::fail(std::string("delete_file error: ") + e.what());
    }
}

ToolResult ExecutionSandbox::execute_command(const std::string& command) {
    if (!config_.allow_subprocesses) {
        return ToolResult::fail("Subprocess execution is disabled in sandbox configuration");
    }
    if (command.empty()) {
        return ToolResult::fail("Cannot execute empty command");
    }

    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) {
        return ToolResult::fail("Failed to spawn process for command: " + command);
    }

    char buffer[256];
    std::string result;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
        if (result.size() > config_.max_output_chars) {
            result = result.substr(0, config_.max_output_chars);
            result += "\n[... stdout truncated ...]";
            break;
        }
    }

    int exit_code = pclose(pipe);
    if (exit_code != 0) {
        return ToolResult::fail("Command exited with code " + std::to_string(exit_code) + "\nOutput: " + result);
    }

    return ToolResult::ok(result);
}

void ExecutionSandbox::reset_workspace() {
    try {
        if (std::filesystem::exists(root_path_)) {
            std::filesystem::remove_all(root_path_);
        }
        std::filesystem::create_directories(root_path_);
    } catch (...) {}
}

std::shared_ptr<Tool> ExecutionSandbox::create_file_writer_tool() {
    auto self = shared_from_this();
    std::vector<ToolParam> params = {
        {"path", ToolParamType::String, "Relative path of file inside sandbox workspace", true},
        {"content", ToolParamType::String, "Text content to write to file", true}
    };
    return std::make_shared<Tool>(
        "write_file",
        "Writes text content to a sandboxed file",
        std::move(params),
        [self](const std::unordered_map<std::string, std::string>& args) -> ToolResult {
            return self->write_file(args.at("path"), args.at("content"));
        }
    );
}

std::shared_ptr<Tool> ExecutionSandbox::create_file_reader_tool() {
    auto self = shared_from_this();
    std::vector<ToolParam> params = {
        {"path", ToolParamType::String, "Relative path of file inside sandbox workspace", true}
    };
    return std::make_shared<Tool>(
        "read_file",
        "Reads text content from a sandboxed file",
        std::move(params),
        [self](const std::unordered_map<std::string, std::string>& args) -> ToolResult {
            return self->read_file(args.at("path"));
        }
    );
}

std::shared_ptr<Tool> ExecutionSandbox::create_file_lister_tool() {
    auto self = shared_from_this();
    std::vector<ToolParam> params = {
        {"directory", ToolParamType::String, "Relative subdirectory to list (optional, defaults to root)", false}
    };
    return std::make_shared<Tool>(
        "list_files",
        "Lists files and directories inside the sandbox workspace",
        std::move(params),
        [self](const std::unordered_map<std::string, std::string>& args) -> ToolResult {
            auto it = args.find("directory");
            return self->list_files(it != args.end() ? it->second : "");
        }
    );
}

std::shared_ptr<Tool> ExecutionSandbox::create_shell_tool() {
    auto self = shared_from_this();
    std::vector<ToolParam> params = {
        {"command", ToolParamType::String, "Shell command to execute in sandbox", true}
    };
    return std::make_shared<Tool>(
        "shell_exec",
        "Executes a sandboxed shell command and returns stdout",
        std::move(params),
        [self](const std::unordered_map<std::string, std::string>& args) -> ToolResult {
            return self->execute_command(args.at("command"));
        }
    );
}

} // namespace aic::autonomous
