# Production Async Streaming Server & OpenAI-Compatible REST API Engine (Aşama 25 - Final)

## 1. Overview & Architectural Philosophy
The **AutonomousAICore (AIC) Production Async Streaming Server** represents the 25th and capstone architectural stage of the autonomous deep learning engine. Built strictly in **Modern C++20 with zero third-party web frameworks** (no Boost.Beast, Crow, Drogon, libcurl, or civetweb), it provides an enterprise-grade, asynchronous, high-throughput inference serving daemon fully compatible with the standard **OpenAI REST API v1 specification**.

```
                [ HTTP/1.1 Client / Web UI / Agent SDK ]
                                |
                    (TCP Loopback / Socket)
                                v
           +-----------------------------------------+
           |       AsyncStreamingServer (C++20)       |
           |  - Non-blocking Winsock2 / POSIX loop   |
           |  - Zero-alloc HTTP/1.1 Parser & Router  |
           |  - CORS & Keep-Alive Connection Handler |
           +--------------------+--------------------+
                                |
             +------------------+------------------+
             |                                     |
    [ GET /health ]                         [ POST /v1/chat/completions ]
    [ GET /v1/models ]                             |
                                                   v
                              +---------------------------------------+
                              |        Request Dispatch & Flow        |
                              |  - JSON Payload Deserializer          |
                              |  - Grammar Schema Extractor           |
                              |  - ContinuousBatchEngine Enqueue      |
                              +-------------------+-------------------+
                                                  |
                                                  v
                      +-----------------------------------------------+
                      |        Streaming / Non-Streaming Output       |
                      |  - JSON Mode: {id, choices, usage, ...}       |
                      |  - SSE Mode: data: {choices: [delta: ...]}   |
                      |              data: [DONE]                     |
                      +-----------------------------------------------+
```

---

## 2. Supported Endpoints

### 2.1. `GET /health`
- **Purpose**: Liveness and readiness probe for load balancers and Kubernetes clusters.
- **Response**:
```json
{
  "status": "ok",
  "engine": "AutonomousAICore",
  "version": "0.1.0",
  "device": "NVIDIA Blackwell sm_120 / Intel Core Ultra 9 285HX",
  "allocated_vram_mb": 0.00
}
```

### 2.2. `GET /v1/models`
- **Purpose**: Lists loaded and active TransformerLM models, quantizations, and multimodal backends.
- **Response**:
```json
{
  "object": "list",
  "data": [
    {
      "id": "aic-mini-llm",
      "object": "model",
      "created": 1774214400,
      "owned_by": "aic-autonomous"
    }
  ]
}
```

### 2.3. `POST /v1/chat/completions`
- **Purpose**: Standard OpenAI-compatible conversational endpoint supporting both buffered JSON responses and real-time Server-Sent Events (SSE) token streaming.
- **Request Format**:
```json
{
  "model": "aic-mini-llm",
  "messages": [
    {"role": "system", "content": "You are a helpful mathematical assistant."},
    {"role": "user", "content": "Compute 14 * 16."}
  ],
  "temperature": 0.4,
  "top_p": 0.9,
  "top_k": 40,
  "max_tokens": 64,
  "stream": true,
  "grammar_json_schema": "{\"type\":\"object\"}"
}
```

#### Non-Streaming Response (`stream: false`):
```http
HTTP/1.1 200 OK
Content-Type: application/json
Content-Length: 320
Access-Control-Allow-Origin: *

{
  "id": "chatcmpl-1774214400",
  "object": "chat.completion",
  "created": 1774214400,
  "model": "aic-mini-llm",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The result of 14 * 16 is 224."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 14,
    "completion_tokens": 11,
    "total_tokens": 25
  }
}
```

#### Streaming Response (`stream: true`):
```http
HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
Access-Control-Allow-Origin: *

data: {"id":"chatcmpl-1774214400","object":"chat.completion.chunk","model":"aic-mini-llm","choices":[{"index":0,"delta":{"content":"The"},"finish_reason":null}]}

data: {"id":"chatcmpl-1774214400","object":"chat.completion.chunk","model":"aic-mini-llm","choices":[{"index":0,"delta":{"content":" result"},"finish_reason":null}]}

data: {"id":"chatcmpl-1774214400","object":"chat.completion.chunk","model":"aic-mini-llm","choices":[{"index":0,"delta":{"content":" is 224."},"finish_reason":"stop"}]}

data: [DONE]
```

---

## 3. High-Performance Features
1. **Zero-Dependency Winsock2 & POSIX Socket Core**:
   - Asynchronous accept/poll loop running in a dedicated worker thread.
   - Non-blocking socket I/O with 5000ms configurable keep-alive timeouts.
2. **In-Flight Continuous Batching & PagedAttention Hook**:
   - Automatically channels incoming user messages into `ContinuousBatchEngine`, allocating physical memory blocks via `BlockManager` with zero VRAM fragmentation.
3. **Structured Grammar Constrained Decoding**:
   - Inspects `grammar_json_schema` in the HTTP body, initializing a DFA/Trie validator on the fly to enforce valid JSON responses token-by-token.
4. **CORS Preflight & Error Resilience**:
   - Handles `OPTIONS` HTTP preflight queries with standard `Access-Control-Allow-*` headers.
   - Robustly returns standardized RFC/OpenAI error envelopes (`invalid_request_error`, `not_found`, `bad_request`).

---

## 4. CLI Serving Commands

Launch the standalone production server:
```powershell
# Default port 8080 on Blackwell CUDA GPU
.\build_ninja\bin\aic.exe serve 8080 cuda

# Run on CPU AVX2/FMA backend
.\build_ninja\bin\aic.exe serve 8080 cpu
```

Self-contained automated tests:
```powershell
.\build_ninja\bin\aic_tests.exe
# Tests include:
# - ServerTest.HttpRequestParserAndHeaderExtraction
# - ServerTest.OpenAIChatCompletionRequestJsonParsing
# - ServerTest.HealthEndpointAndModelsDiscovery
# - ServerTest.ChatCompletionsNonStreamingJsonResponse
# - ServerTest.ChatCompletionsSseTokenStreaming
# - ServerTest.CORSPreflightOptionsAndErrorHandling
# - ServerTest.LiveLoopbackSocketHttpExchange
```
