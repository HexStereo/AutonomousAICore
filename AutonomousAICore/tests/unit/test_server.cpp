// tests/unit/test_server.cpp
// Unit tests for Production Async Streaming Server & OpenAI-Compatible REST API Engine (Aşama 25 - Final)
#include "tests/test_framework.hpp"
#include "server/http_server.hpp"
#include "core/device.hpp"

#include <chrono>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

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

using namespace aic;
using namespace aic::server;

AIC_TEST(ServerTest, HttpRequestParserAndHeaderExtraction) {
    std::string raw = 
        "POST /v1/chat/completions?version=2 HTTP/1.1\r\n"
        "Host: localhost:8080\r\n"
        "User-Agent: curl/8.4.0\r\n"
        "Accept: application/json\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: 26\r\n"
        "\r\n"
        "{\"model\": \"aic-mini-llm\"}";

    HttpRequest req = AsyncStreamingServer::parse_http_request(raw);

    ASSERT_EQ(req.method, HttpMethod::POST);
    ASSERT_EQ(req.path, "/v1/chat/completions");
    ASSERT_EQ(req.query_string, "version=2");
    ASSERT_EQ(req.get_header("Host"), "localhost:8080");
    ASSERT_EQ(req.get_header("content-type"), "application/json"); // Case-insensitive
    ASSERT_TRUE(req.has_header("Content-Length"));
    ASSERT_FALSE(req.has_header("Authorization"));
    ASSERT_EQ(req.body, "{\"model\": \"aic-mini-llm\"}");
}

AIC_TEST(ServerTest, OpenAIChatCompletionRequestJsonParsing) {
    std::string json_body = 
        "{\n"
        "  \"model\": \"aic-mini-llm\",\n"
        "  \"messages\": [\n"
        "    {\"role\": \"system\", \"content\": \"You are a helpful assistant.\"},\n"
        "    {\"role\": \"user\", \"content\": \"What is 25 * 16?\"}\n"
        "  ],\n"
        "  \"temperature\": 0.4,\n"
        "  \"top_p\": 0.95,\n"
        "  \"top_k\": 50,\n"
        "  \"max_tokens\": 64,\n"
        "  \"stream\": true\n"
        "}";

    ChatCompletionRequest chat_req = AsyncStreamingServer::parse_chat_completion_request(json_body);

    ASSERT_EQ(chat_req.model, "aic-mini-llm");
    ASSERT_TRUE(chat_req.stream);
    ASSERT_NEAR(chat_req.temperature, 0.4f, 1e-3f);
    ASSERT_NEAR(chat_req.top_p, 0.95f, 1e-3f);
    ASSERT_EQ(chat_req.top_k, 50);
    ASSERT_EQ(chat_req.max_tokens, 64);
    ASSERT_EQ(chat_req.messages.size(), 2);
    ASSERT_EQ(chat_req.messages[0].role, "system");
    ASSERT_EQ(chat_req.messages[0].content, "You are a helpful assistant.");
    ASSERT_EQ(chat_req.messages[1].role, "user");
    ASSERT_EQ(chat_req.messages[1].content, "What is 25 * 16?");
}

AIC_TEST(ServerTest, HealthEndpointAndModelsDiscovery) {
    ServerConfig cfg;
    AsyncStreamingServer server(cfg);

    // 1. Test /health
    HttpRequest req_health;
    req_health.method = HttpMethod::GET;
    req_health.path = "/health";
    HttpResponse res_health = server.handle_request(req_health);

    ASSERT_EQ(res_health.status_code, 200);
    ASSERT_TRUE(res_health.body.find("\"status\": \"ok\"") != std::string::npos);
    ASSERT_TRUE(res_health.body.find("AutonomousAICore") != std::string::npos);

    // 2. Test /v1/models
    HttpRequest req_models;
    req_models.method = HttpMethod::GET;
    req_models.path = "/v1/models";
    HttpResponse res_models = server.handle_request(req_models);

    ASSERT_EQ(res_models.status_code, 200);
    ASSERT_TRUE(res_models.body.find("\"object\": \"list\"") != std::string::npos);
    ASSERT_TRUE(res_models.body.find("aic-mini-llm") != std::string::npos);
    ASSERT_TRUE(res_models.body.find("aic-vlm") != std::string::npos);
    ASSERT_TRUE(res_models.body.find("aic-reasoner") != std::string::npos);
}

AIC_TEST(ServerTest, ChatCompletionsNonStreamingJsonResponse) {
    ServerConfig cfg;
    AsyncStreamingServer server(cfg);

    HttpRequest req;
    req.method = HttpMethod::POST;
    req.path = "/v1/chat/completions";
    req.body = 
        "{\n"
        "  \"model\": \"aic-mini-llm\",\n"
        "  \"messages\": [{\"role\": \"user\", \"content\": \"Say hello\"}],\n"
        "  \"stream\": false\n"
        "}";

    HttpResponse res = server.handle_request(req);
    ASSERT_EQ(res.status_code, 200);
    ASSERT_EQ(res.headers["Content-Type"], "application/json; charset=utf-8");

    // Verify OpenAI compatible fields
    ASSERT_TRUE(res.body.find("\"object\": \"chat.completion\"") != std::string::npos);
    ASSERT_TRUE(res.body.find("\"role\": \"assistant\"") != std::string::npos);
    ASSERT_TRUE(res.body.find("\"usage\"") != std::string::npos);
    ASSERT_TRUE(res.body.find("\"finish_reason\": \"stop\"") != std::string::npos);
}

AIC_TEST(ServerTest, ChatCompletionsSseTokenStreaming) {
    ServerConfig cfg;
    AsyncStreamingServer server(cfg);

    HttpRequest req;
    req.method = HttpMethod::POST;
    req.path = "/v1/chat/completions";
    req.body = 
        "{\n"
        "  \"model\": \"aic-mini-llm\",\n"
        "  \"messages\": [{\"role\": \"user\", \"content\": \"Stream response\"}],\n"
        "  \"stream\": true\n"
        "}";

    HttpResponse res = server.handle_request(req);
    ASSERT_EQ(res.status_code, 200);
    ASSERT_EQ(res.headers["Content-Type"], "text/event-stream; charset=utf-8");
    ASSERT_EQ(res.headers["Cache-Control"], "no-cache");

    // Verify SSE structure
    ASSERT_TRUE(res.body.find("data: {\"id\":") != std::string::npos);
    ASSERT_TRUE(res.body.find("\"object\":\"chat.completion.chunk\"") != std::string::npos);
    ASSERT_TRUE(res.body.find("\"delta\":{\"content\":") != std::string::npos);
    ASSERT_TRUE(res.body.find("data: [DONE]\n\n") != std::string::npos);
}

AIC_TEST(ServerTest, CORSPreflightOptionsAndErrorHandling) {
    ServerConfig cfg;
    AsyncStreamingServer server(cfg);

    // 1. CORS OPTIONS preflight
    HttpRequest req_opt;
    req_opt.method = HttpMethod::OPTIONS;
    req_opt.path = "/v1/chat/completions";
    HttpResponse res_opt = server.handle_request(req_opt);

    ASSERT_EQ(res_opt.status_code, 204);
    ASSERT_EQ(res_opt.headers["Access-Control-Allow-Origin"], "*");
    ASSERT_TRUE(res_opt.headers["Access-Control-Allow-Methods"].find("POST") != std::string::npos);

    // 2. 404 for unknown endpoint
    HttpRequest req_404;
    req_404.method = HttpMethod::GET;
    req_404.path = "/v1/unknown_resource";
    HttpResponse res_404 = server.handle_request(req_404);

    ASSERT_EQ(res_404.status_code, 404);
    ASSERT_TRUE(res_404.body.find("invalid_request_error") != std::string::npos);
}

AIC_TEST(ServerTest, LiveLoopbackSocketHttpExchange) {
#if defined(_WIN32)
    ServerConfig cfg;
    cfg.host = "127.0.0.1";
    cfg.port = 18888; // Use test loopback port
    AsyncStreamingServer server(cfg);

    bool started = server.start();
    ASSERT_TRUE(started);
    ASSERT_TRUE(server.is_running());

    // Small sleep to ensure accept loop is ready
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Connect real client socket to 127.0.0.1:18888
    SOCKET client = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    ASSERT_TRUE(client != INVALID_SOCKET);

    sockaddr_in server_addr;
    std::memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(server.bound_port());
    inet_pton(AF_INET, "127.0.0.1", &server_addr.sin_addr);

    int conn = connect(client, reinterpret_cast<sockaddr*>(&server_addr), sizeof(server_addr));
    ASSERT_EQ(conn, 0);

    // Send HTTP GET /health
    std::string http_req = "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n";
    send(client, http_req.data(), static_cast<int>(http_req.size()), 0);

    // Receive HTTP response
    char buf[2048];
    int n = recv(client, buf, sizeof(buf) - 1, 0);
    ASSERT_TRUE(n > 0);
    buf[n] = '\0';
    std::string http_res(buf, static_cast<size_t>(n));

    ASSERT_TRUE(http_res.find("HTTP/1.1 200 OK") != std::string::npos);
    ASSERT_TRUE(http_res.find("\"status\": \"ok\"") != std::string::npos);

    closesocket(client);
    server.stop();
    ASSERT_FALSE(server.is_running());
#endif
}
