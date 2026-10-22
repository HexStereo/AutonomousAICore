// include/server/http_server.hpp
// Production Async Streaming Server & OpenAI-Compatible REST API Engine (Aşama 25 - Final)
// Provides zero-dependency HTTP/1.1 parsing, Server-Sent Events (SSE) token streaming,
// /v1/chat/completions, /v1/models, and integration with ContinuousBatchEngine & Grammar.
#pragma once

#include "core/error.hpp"
#include "data/tokenizer.hpp"
#include "inference/continuous_batch_engine.hpp"
#include "inference/grammar.hpp"
#include "models/transformer_lm.hpp"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace aic::server {

enum class HttpMethod {
    GET,
    POST,
    PUT,
    DEL,
    OPTIONS,
    UNKNOWN
};

const char* http_method_name(HttpMethod method) noexcept;
HttpMethod parse_http_method(const std::string& str) noexcept;

inline std::ostream& operator<<(std::ostream& os, HttpMethod method) {
    return os << http_method_name(method);
}

struct HttpRequest {
    HttpMethod method = HttpMethod::UNKNOWN;
    std::string path;
    std::string query_string;
    std::unordered_map<std::string, std::string> headers;
    std::string body;

    std::string get_header(const std::string& key) const;
    bool has_header(const std::string& key) const;
};

struct HttpResponse {
    int status_code = 200;
    std::string status_text = "OK";
    std::unordered_map<std::string, std::string> headers;
    std::string body;

    static HttpResponse json(int code, const std::string& json_body);
    static HttpResponse ok(const std::string& body, const std::string& content_type = "text/plain");
    static HttpResponse not_found(const std::string& msg = "Not Found");
    static HttpResponse bad_request(const std::string& msg = "Bad Request");

    std::string serialize() const;
};

struct ChatMessage {
    std::string role; // "system", "user", "assistant"
    std::string content;
};

struct ChatCompletionRequest {
    std::string model = "aic-mini-llm";
    std::vector<ChatMessage> messages;
    float temperature = 0.7f;
    float top_p = 0.9f;
    int top_k = 40;
    int64_t max_tokens = 128;
    bool stream = false;
    std::string grammar_json_schema; // Optional JSON schema constraint
};

struct ChatCompletionResponse {
    std::string id;
    std::string model;
    int64_t created_at = 0;
    std::string content;
    std::string finish_reason = "stop";
    int prompt_tokens = 0;
    int completion_tokens = 0;

    std::string to_json() const;
};

struct ChatCompletionChunk {
    std::string id;
    std::string model;
    std::string delta_content;
    std::string finish_reason; // empty or "stop"

    std::string to_sse_event() const;
};

struct ServerConfig {
    std::string host = "127.0.0.1";
    uint16_t port = 8080;
    int num_worker_threads = 4;
    int socket_timeout_ms = 5000;
    std::string default_model = "aic-mini-llm";
};

class AsyncStreamingServer {
public:
    AsyncStreamingServer(
        ServerConfig config,
        std::shared_ptr<models::TransformerLM> model = nullptr,
        std::shared_ptr<data::Tokenizer> tokenizer = nullptr,
        std::shared_ptr<inference::ContinuousBatchEngine> batch_engine = nullptr
    );
    ~AsyncStreamingServer();

    // Start / stop socket listening
    bool start();
    void stop();
    bool is_running() const noexcept { return running_; }

    uint16_t bound_port() const noexcept { return bound_port_; }
    const ServerConfig& config() const noexcept { return config_; }

    // Core HTTP request router
    HttpResponse handle_request(const HttpRequest& req);

    // Parsing utilities
    static HttpRequest parse_http_request(const std::string& raw_request);
    static ChatCompletionRequest parse_chat_completion_request(const std::string& json_body);

    // Simulated HTTP request execution without network sockets
    std::string simulate_http_request(const std::string& raw_request);

private:
    void server_loop();
    void handle_client_socket(uintptr_t client_sock);

    ServerConfig config_;
    std::shared_ptr<models::TransformerLM> model_;
    std::shared_ptr<data::Tokenizer> tokenizer_;
    std::shared_ptr<inference::ContinuousBatchEngine> batch_engine_;

    std::atomic<bool> running_{false};
    uint16_t bound_port_ = 0;
    uintptr_t listen_socket_ = 0;
    std::thread server_thread_;
};

} // namespace aic::server
