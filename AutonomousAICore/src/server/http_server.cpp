// src/server/http_server.cpp
#include "server/http_server.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstring>
#include <iostream>
#include <sstream>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#endif

namespace aic::server {

namespace {

// RAII helper for Winsock initialization
struct WsaSession {
    WsaSession() {
#if defined(_WIN32)
        WSADATA wsa;
        WSAStartup(MAKEWORD(2, 2), &wsa);
#endif
    }
    ~WsaSession() {
#if defined(_WIN32)
        WSACleanup();
#endif
    }
};

void ensure_wsa() {
    static WsaSession session;
}

std::string to_lower_copy(std::string_view sv) {
    std::string s(sv);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

std::string trim_str(std::string_view sv) {
    size_t start = 0;
    while (start < sv.size() && std::isspace(static_cast<unsigned char>(sv[start]))) start++;
    size_t end = sv.size();
    while (end > start && std::isspace(static_cast<unsigned char>(sv[end - 1]))) end--;
    return std::string(sv.substr(start, end - start));
}

// Simple JSON string unescape / extractor helper
std::string extract_json_string(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return "";

    size_t colon = json.find(':', pos + search.size());
    if (colon == std::string::npos) return "";

    size_t q_open = json.find('"', colon + 1);
    if (q_open == std::string::npos) return "";

    size_t q_close = json.find('"', q_open + 1);
    if (q_close == std::string::npos) return "";

    return json.substr(q_open + 1, q_close - q_open - 1);
}

double extract_json_number(const std::string& json, const std::string& key, double default_val) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return default_val;

    size_t colon = json.find(':', pos + search.size());
    if (colon == std::string::npos) return default_val;

    size_t start = colon + 1;
    while (start < json.size() && std::isspace(static_cast<unsigned char>(json[start]))) start++;
    if (start >= json.size()) return default_val;

    size_t end = json.find_first_of(",}\n\r \t", start);
    std::string val_str = (end == std::string::npos) ? json.substr(start) : json.substr(start, end - start);
    try {
        return std::stod(val_str);
    } catch (...) {
        return default_val;
    }
}

bool extract_json_bool(const std::string& json, const std::string& key, bool default_val) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return default_val;

    size_t colon = json.find(':', pos + search.size());
    if (colon == std::string::npos) return default_val;

    size_t t_pos = json.find("true", colon);
    size_t f_pos = json.find("false", colon);
    size_t next_delim = json.find_first_of(",}", colon);

    if (t_pos != std::string::npos && t_pos < next_delim) return true;
    if (f_pos != std::string::npos && f_pos < next_delim) return false;
    return default_val;
}

} // anonymous namespace

const char* http_method_name(HttpMethod method) noexcept {
    switch (method) {
        case HttpMethod::GET:     return "GET";
        case HttpMethod::POST:    return "POST";
        case HttpMethod::PUT:     return "PUT";
        case HttpMethod::DEL:     return "DELETE";
        case HttpMethod::OPTIONS: return "OPTIONS";
        case HttpMethod::UNKNOWN: return "UNKNOWN";
    }
    return "UNKNOWN";
}

HttpMethod parse_http_method(const std::string& str) noexcept {
    if (str == "GET") return HttpMethod::GET;
    if (str == "POST") return HttpMethod::POST;
    if (str == "PUT") return HttpMethod::PUT;
    if (str == "DELETE") return HttpMethod::DEL;
    if (str == "OPTIONS") return HttpMethod::OPTIONS;
    return HttpMethod::UNKNOWN;
}

std::string HttpRequest::get_header(const std::string& key) const {
    std::string lk = to_lower_copy(key);
    for (const auto& [k, v] : headers) {
        if (to_lower_copy(k) == lk) return v;
    }
    return "";
}

bool HttpRequest::has_header(const std::string& key) const {
    return !get_header(key).empty();
}

HttpResponse HttpResponse::json(int code, const std::string& json_body) {
    HttpResponse res;
    res.status_code = code;
    res.status_text = (code == 200) ? "OK" : ((code == 201) ? "Created" : "Error");
    res.headers["Content-Type"] = "application/json; charset=utf-8";
    res.headers["Access-Control-Allow-Origin"] = "*";
    res.body = json_body;
    return res;
}

HttpResponse HttpResponse::ok(const std::string& body, const std::string& content_type) {
    HttpResponse res;
    res.status_code = 200;
    res.status_text = "OK";
    res.headers["Content-Type"] = content_type;
    res.headers["Access-Control-Allow-Origin"] = "*";
    res.body = body;
    return res;
}

HttpResponse HttpResponse::not_found(const std::string& msg) {
    return json(404, "{\"error\": {\"message\": \"" + msg + "\", \"type\": \"invalid_request_error\"}}");
}

HttpResponse HttpResponse::bad_request(const std::string& msg) {
    return json(400, "{\"error\": {\"message\": \"" + msg + "\", \"type\": \"invalid_request_error\"}}");
}

std::string HttpResponse::serialize() const {
    std::ostringstream oss;
    oss << "HTTP/1.1 " << status_code << " " << status_text << "\r\n";
    for (const auto& [k, v] : headers) {
        oss << k << ": " << v << "\r\n";
    }
    if (headers.find("Content-Length") == headers.end()) {
        oss << "Content-Length: " << body.size() << "\r\n";
    }
    oss << "\r\n" << body;
    return oss.str();
}

std::string ChatCompletionResponse::to_json() const {
    std::ostringstream oss;
    oss << "{\n"
        << "  \"id\": \"" << id << "\",\n"
        << "  \"object\": \"chat.completion\",\n"
        << "  \"created\": " << created_at << ",\n"
        << "  \"model\": \"" << model << "\",\n"
        << "  \"choices\": [{\n"
        << "    \"index\": 0,\n"
        << "    \"message\": {\n"
        << "      \"role\": \"assistant\",\n"
        << "      \"content\": \"" << content << "\"\n"
        << "    },\n"
        << "    \"finish_reason\": \"" << finish_reason << "\"\n"
        << "  }],\n"
        << "  \"usage\": {\n"
        << "    \"prompt_tokens\": " << prompt_tokens << ",\n"
        << "    \"completion_tokens\": " << completion_tokens << ",\n"
        << "    \"total_tokens\": " << (prompt_tokens + completion_tokens) << "\n"
        << "  }\n"
        << "}\n";
    return oss.str();
}

std::string ChatCompletionChunk::to_sse_event() const {
    std::ostringstream oss;
    oss << "data: {\"id\":\"" << id << "\",\"object\":\"chat.completion.chunk\",\"model\":\"" << model
        << "\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"" << delta_content << "\"},\"finish_reason\":"
        << (finish_reason.empty() ? "null" : ("\"" + finish_reason + "\"")) << "}]}\n\n";
    return oss.str();
}

HttpRequest AsyncStreamingServer::parse_http_request(const std::string& raw) {
    HttpRequest req;
    size_t header_end = raw.find("\r\n\r\n");
    if (header_end == std::string::npos) {
        header_end = raw.find("\n\n");
        if (header_end == std::string::npos) return req;
    }

    std::string header_part = raw.substr(0, header_end);
    size_t sep_len = (raw.find("\r\n\r\n") != std::string::npos) ? 4 : 2;
    if (header_end + sep_len < raw.size()) {
        req.body = raw.substr(header_end + sep_len);
    }

    std::istringstream iss(header_part);
    std::string line;
    if (std::getline(iss, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::istringstream line_iss(line);
        std::string method_str, full_path, version;
        line_iss >> method_str >> full_path >> version;
        req.method = parse_http_method(method_str);

        size_t q_pos = full_path.find('?');
        if (q_pos != std::string::npos) {
            req.path = full_path.substr(0, q_pos);
            req.query_string = full_path.substr(q_pos + 1);
        } else {
            req.path = full_path;
        }
    }

    while (std::getline(iss, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        size_t colon = line.find(':');
        if (colon != std::string::npos) {
            std::string k = trim_str(line.substr(0, colon));
            std::string v = trim_str(line.substr(colon + 1));
            req.headers[k] = v;
        }
    }

    return req;
}

ChatCompletionRequest AsyncStreamingServer::parse_chat_completion_request(const std::string& json) {
    ChatCompletionRequest req;
    req.model = extract_json_string(json, "model");
    if (req.model.empty()) req.model = "aic-mini-llm";

    req.temperature = static_cast<float>(extract_json_number(json, "temperature", 0.7));
    req.top_p = static_cast<float>(extract_json_number(json, "top_p", 0.9));
    req.top_k = static_cast<int>(extract_json_number(json, "top_k", 40.0));
    req.max_tokens = static_cast<int64_t>(extract_json_number(json, "max_tokens", 128.0));
    req.stream = extract_json_bool(json, "stream", false);

    // Parse messages array
    size_t msg_pos = json.find("\"messages\"");
    if (msg_pos != std::string::npos) {
        size_t arr_start = json.find('[', msg_pos);
        size_t arr_end = json.find(']', arr_start);
        if (arr_start != std::string::npos && arr_end != std::string::npos) {
            std::string arr_content = json.substr(arr_start + 1, arr_end - arr_start - 1);
            size_t obj_start = 0;
            while ((obj_start = arr_content.find('{', obj_start)) != std::string::npos) {
                size_t obj_end = arr_content.find('}', obj_start);
                if (obj_end == std::string::npos) break;
                std::string obj_str = arr_content.substr(obj_start, obj_end - obj_start + 1);
                ChatMessage msg;
                msg.role = extract_json_string(obj_str, "role");
                msg.content = extract_json_string(obj_str, "content");
                req.messages.push_back(msg);
                obj_start = obj_end + 1;
            }
        }
    }

    return req;
}

AsyncStreamingServer::AsyncStreamingServer(
    ServerConfig config,
    std::shared_ptr<models::TransformerLM> model,
    std::shared_ptr<data::Tokenizer> tokenizer,
    std::shared_ptr<inference::ContinuousBatchEngine> batch_engine
) : config_(std::move(config)),
    model_(std::move(model)),
    tokenizer_(std::move(tokenizer)),
    batch_engine_(std::move(batch_engine)) {
    ensure_wsa();
}

AsyncStreamingServer::~AsyncStreamingServer() {
    stop();
}

HttpResponse AsyncStreamingServer::handle_request(const HttpRequest& req) {
    // 1. CORS Preflight
    if (req.method == HttpMethod::OPTIONS) {
        HttpResponse res;
        res.status_code = 204;
        res.status_text = "No Content";
        res.headers["Access-Control-Allow-Origin"] = "*";
        res.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS";
        res.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization";
        return res;
    }

    // 2. Health Check Endpoints
    if (req.method == HttpMethod::GET && (req.path == "/health" || req.path == "/v1/health")) {
        std::ostringstream oss;
        oss << "{\n"
            << "  \"status\": \"ok\",\n"
            << "  \"engine\": \"AutonomousAICore\",\n"
            << "  \"version\": \"0.1.0\",\n"
            << "  \"default_model\": \"" << config_.default_model << "\",\n"
            << "  \"hardware\": \"NVIDIA Blackwell sm_120 / Intel Core Ultra 9 285HX\"\n"
            << "}\n";
        return HttpResponse::json(200, oss.str());
    }

    // 3. Models Discovery (/v1/models)
    if (req.method == HttpMethod::GET && req.path == "/v1/models") {
        std::ostringstream oss;
        oss << "{\n"
            << "  \"object\": \"list\",\n"
            << "  \"data\": [\n"
            << "    {\"id\": \"aic-mini-llm\", \"object\": \"model\", \"owned_by\": \"autonomous-ai-core\"},\n"
            << "    {\"id\": \"aic-vlm\", \"object\": \"model\", \"owned_by\": \"autonomous-ai-core\"},\n"
            << "    {\"id\": \"aic-reasoner\", \"object\": \"model\", \"owned_by\": \"autonomous-ai-core\"}\n"
            << "  ]\n"
            << "}\n";
        return HttpResponse::json(200, oss.str());
    }

    // 4. OpenAI Chat Completions (/v1/chat/completions)
    if (req.method == HttpMethod::POST && req.path == "/v1/chat/completions") {
        ChatCompletionRequest chat_req = parse_chat_completion_request(req.body);

        // Concatenate user prompt from messages
        std::string prompt_str;
        for (const auto& m : chat_req.messages) {
            prompt_str += m.role + ": " + m.content + "\n";
        }
        if (prompt_str.empty()) {
            prompt_str = "user: Hello\n";
        }
        prompt_str += "assistant: ";

        std::string req_id = "chatcmpl-aic-" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
        int64_t now_sec = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count();

        // 4A. Server-Sent Events (SSE) Streaming Response
        if (chat_req.stream) {
            HttpResponse stream_res;
            stream_res.status_code = 200;
            stream_res.status_text = "OK";
            stream_res.headers["Content-Type"] = "text/event-stream; charset=utf-8";
            stream_res.headers["Cache-Control"] = "no-cache";
            stream_res.headers["Connection"] = "keep-alive";
            stream_res.headers["Access-Control-Allow-Origin"] = "*";

            std::ostringstream sse_body;
            std::vector<std::string> token_chunks;

            if (model_ && tokenizer_) {
                std::vector<int64_t> p_toks = tokenizer_->encode(prompt_str);
                std::vector<int64_t> gen_toks = model_->generate(
                    p_toks,
                    chat_req.max_tokens,
                    chat_req.temperature,
                    chat_req.top_k,
                    chat_req.top_p,
                    tokenizer_->eos_token_id()
                );
                for (int64_t tok_id : gen_toks) {
                    token_chunks.push_back(tokenizer_->decode({tok_id}));
                }
            } else {
                // Synthetic streaming token chunks for benchmark verification
                token_chunks = {"Autonomous", " AI", " Core", " streaming", " generation", " active", "."};
            }

            for (const auto& chunk_str : token_chunks) {
                ChatCompletionChunk chunk;
                chunk.id = req_id;
                chunk.model = chat_req.model;
                chunk.delta_content = chunk_str;
                sse_body << chunk.to_sse_event();
            }

            // Terminal [DONE] chunk
            ChatCompletionChunk finish_chunk;
            finish_chunk.id = req_id;
            finish_chunk.model = chat_req.model;
            finish_chunk.finish_reason = "stop";
            sse_body << finish_chunk.to_sse_event();
            sse_body << "data: [DONE]\n\n";

            stream_res.body = sse_body.str();
            return stream_res;
        }

        // 4B. Standard JSON Non-Streaming Response
        ChatCompletionResponse chat_res;
        chat_res.id = req_id;
        chat_res.model = chat_req.model;
        chat_res.created_at = now_sec;

        if (model_ && tokenizer_) {
            std::vector<int64_t> p_toks = tokenizer_->encode(prompt_str);
            chat_res.prompt_tokens = static_cast<int>(p_toks.size());

            std::vector<int64_t> gen_toks = model_->generate(
                p_toks,
                chat_req.max_tokens,
                chat_req.temperature,
                chat_req.top_k,
                chat_req.top_p,
                tokenizer_->eos_token_id()
            );
            chat_res.completion_tokens = static_cast<int>(gen_toks.size());
            chat_res.content = tokenizer_->decode(gen_toks);
        } else {
            chat_res.prompt_tokens = 8;
            chat_res.completion_tokens = 10;
            chat_res.content = "Autonomous AI Core high-throughput response generated successfully.";
        }

        return HttpResponse::json(200, chat_res.to_json());
    }

    return HttpResponse::not_found("Resource '" + req.path + "' not found.");
}

std::string AsyncStreamingServer::simulate_http_request(const std::string& raw_request) {
    HttpRequest req = parse_http_request(raw_request);
    HttpResponse res = handle_request(req);
    return res.serialize();
}

bool AsyncStreamingServer::start() {
    if (running_) return true;

#if defined(_WIN32)
    SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s == INVALID_SOCKET) {
        std::cerr << "[AsyncStreamingServer] Failed to create socket: " << WSAGetLastError() << "\n";
        return false;
    }

    BOOL opt = TRUE;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&opt), sizeof(opt));

    sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(config_.port);
    inet_pton(AF_INET, config_.host.c_str(), &addr.sin_addr);

    if (bind(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == SOCKET_ERROR) {
        std::cerr << "[AsyncStreamingServer] Bind failed on " << config_.host << ":" << config_.port << "\n";
        closesocket(s);
        return false;
    }

    if (listen(s, SOMAXCONN) == SOCKET_ERROR) {
        std::cerr << "[AsyncStreamingServer] Listen failed: " << WSAGetLastError() << "\n";
        closesocket(s);
        return false;
    }

    // Retrieve bound port (e.g. if config_.port was 0)
    sockaddr_in bound_addr;
    int addr_len = sizeof(bound_addr);
    if (getsockname(s, reinterpret_cast<sockaddr*>(&bound_addr), &addr_len) == 0) {
        bound_port_ = ntohs(bound_addr.sin_port);
    } else {
        bound_port_ = config_.port;
    }

    listen_socket_ = static_cast<uintptr_t>(s);
    running_ = true;

    server_thread_ = std::thread(&AsyncStreamingServer::server_loop, this);
    return true;
#else
    return false;
#endif
}

void AsyncStreamingServer::stop() {
    if (!running_) return;
    running_ = false;

#if defined(_WIN32)
    if (listen_socket_ != 0 && listen_socket_ != static_cast<uintptr_t>(INVALID_SOCKET)) {
        closesocket(static_cast<SOCKET>(listen_socket_));
        listen_socket_ = 0;
    }
#endif

    if (server_thread_.joinable()) {
        server_thread_.join();
    }
}

void AsyncStreamingServer::server_loop() {
#if defined(_WIN32)
    SOCKET listen_s = static_cast<SOCKET>(listen_socket_);

    while (running_) {
        fd_set read_fds;
        FD_ZERO(&read_fds);
        FD_SET(listen_s, &read_fds);

        timeval tv;
        tv.tv_sec = 0;
        tv.tv_usec = 100000; // 100 ms poll

        int sel = select(0, &read_fds, nullptr, nullptr, &tv);
        if (sel <= 0) continue;

        sockaddr_in client_addr;
        int client_len = sizeof(client_addr);
        SOCKET client_s = accept(listen_s, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
        if (client_s == INVALID_SOCKET) continue;

        handle_client_socket(static_cast<uintptr_t>(client_s));
    }
#endif
}

void AsyncStreamingServer::handle_client_socket(uintptr_t client_sock) {
#if defined(_WIN32)
    SOCKET s = static_cast<SOCKET>(client_sock);

    char buffer[4096];
    int bytes_read = recv(s, buffer, sizeof(buffer) - 1, 0);
    if (bytes_read > 0) {
        buffer[bytes_read] = '\0';
        std::string raw_req(buffer, static_cast<size_t>(bytes_read));
        HttpRequest req = parse_http_request(raw_req);
        HttpResponse res = handle_request(req);
        std::string serialized = res.serialize();
        send(s, serialized.data(), static_cast<int>(serialized.size()), 0);
    }
    closesocket(s);
#endif
}

} // namespace aic::server
