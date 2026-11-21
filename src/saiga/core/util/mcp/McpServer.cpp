/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/util/mcp/McpServer.h"

#include "saiga/core/util/assert.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <future>
#include <iostream>
#include <sstream>
#include <thread>

#include <system_error>

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <winsock2.h>
#    include <ws2tcpip.h>
using socket_t                          = SOCKET;
static constexpr socket_t InvalidSocket = INVALID_SOCKET;
static void closeSocket(socket_t s)
{
    closesocket(s);
}
#else
#    include <arpa/inet.h>
#    include <fcntl.h>
#    include <netinet/in.h>
#    include <netinet/tcp.h>
#    include <poll.h>
#    include <sys/socket.h>
#    include <unistd.h>
using socket_t                          = int;
static constexpr socket_t InvalidSocket = -1;
static void closeSocket(socket_t s)
{
    ::close(s);
}
#    ifndef MSG_NOSIGNAL
#        define MSG_NOSIGNAL 0
#    endif
#endif

namespace Saiga
{
using Json = McpServer::Json;

namespace
{
#ifdef _WIN32
struct WinsockInit
{
    WinsockInit()
    {
        WSADATA data;
        WSAStartup(MAKEWORD(2, 2), &data);
    }
    ~WinsockInit() { WSACleanup(); }
};
#endif

// Waits until the socket is readable. Returns false on timeout.
bool waitReadable(socket_t s, int timeoutMs)
{
#ifdef _WIN32
    WSAPOLLFD p = {};
    p.fd        = s;
    p.events    = POLLRDNORM;
    return WSAPoll(&p, 1, timeoutMs) > 0;
#else
    pollfd p = {};
    p.fd     = s;
    p.events = POLLIN;
    return ::poll(&p, 1, timeoutMs) > 0;
#endif
}

// Waits until the socket can take more data. Returns false on timeout.
bool waitWritable(socket_t s, int timeoutMs)
{
#ifdef _WIN32
    WSAPOLLFD p = {};
    p.fd        = s;
    p.events    = POLLWRNORM;
    return WSAPoll(&p, 1, timeoutMs) > 0;
#else
    pollfd p = {};
    p.fd     = s;
    p.events = POLLOUT;
    return ::poll(&p, 1, timeoutMs) > 0;
#endif
}

// Blocking (accepted sockets; on Windows they inherit the non-blocking mode of the listener) or non-blocking mode
// Apple has no MSG_NOSIGNAL: suppress SIGPIPE per socket instead (a client that disconnects during a response would
// otherwise terminate the process)
void suppressSigPipe(socket_t s)
{
#ifdef SO_NOSIGPIPE
    int one = 1;
    setsockopt(s, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#else
    (void)s;
#endif
}

void setBlocking(socket_t s, bool blocking)
{
#ifdef _WIN32
    u_long mode = blocking ? 0 : 1;
    ioctlsocket(s, FIONBIO, &mode);
#else
    int flags = fcntl(s, F_GETFL, 0);
    if (flags < 0) return;
    fcntl(s, F_SETFL, blocking ? (flags & ~O_NONBLOCK) : (flags | O_NONBLOCK));
#endif
}

// Sends all data with non-blocking sends. Gives up (returns false) after the deadline or when the server stops: a
// slow reader must not block stop(). A blocking send() could wait for its whole timeout even after poll() reported the
// socket writable, so the socket is never blocked on: POSIX uses MSG_DONTWAIT, Windows switches the socket to
// non-blocking mode for the duration of the call.
bool sendAll(socket_t s, const std::string& data, const std::atomic_bool* stopping = nullptr,
             std::chrono::seconds deadlineAfter = std::chrono::seconds(30))
{
    auto deadline = std::chrono::steady_clock::now() + deadlineAfter;
    size_t sent   = 0;
#ifdef _WIN32
    setBlocking(s, false);
#endif
    bool ok = true;
    while (sent < data.size())
    {
        if ((stopping && *stopping) || std::chrono::steady_clock::now() > deadline)
        {
            ok = false;
            break;
        }
        size_t piece = std::min<size_t>(data.size() - sent, 64 * 1024);
#ifdef _WIN32
        int n           = ::send(s, data.data() + sent, (int)piece, 0);
        bool wouldBlock = n < 0 && WSAGetLastError() == WSAEWOULDBLOCK;
#else
        ssize_t n       = ::send(s, data.data() + sent, piece, MSG_DONTWAIT | MSG_NOSIGNAL);
        bool wouldBlock = n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR);
#endif
        if (wouldBlock)
        {
            // Short waits, so the stop flag and the deadline are checked regularly
            waitWritable(s, 50);
            continue;
        }
        if (n <= 0)
        {
            ok = false;
            break;
        }
        sent += size_t(n);
    }
#ifdef _WIN32
    setBlocking(s, true);
#endif
    return ok;
}

std::string toLower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    return s;
}

std::string trim(const std::string& s)
{
    size_t b = s.find_first_not_of(" \t\r\n");
    size_t e = s.find_last_not_of(" \t\r\n");
    return b == std::string::npos ? std::string() : s.substr(b, e - b + 1);
}

Json rpcResult(const Json& id, const Json& result)
{
    return Json::object{{"jsonrpc", "2.0"}, {"id", id}, {"result", result}};
}

Json rpcError(const Json& id, int code, const std::string& message)
{
    return Json::object{{"jsonrpc", "2.0"}, {"id", id}, {"error", Json::object{{"code", code}, {"message", message}}}};
}

const char* statusText(int status)
{
    switch (status)
    {
        case 200:
            return "OK";
        case 202:
            return "Accepted";
        case 400:
            return "Bad Request";
        case 403:
            return "Forbidden";
        case 404:
            return "Not Found";
        case 405:
            return "Method Not Allowed";
        case 413:
            return "Payload Too Large";
        case 415:
            return "Unsupported Media Type";
        case 501:
            return "Not Implemented";
        case 503:
            return "Service Unavailable";
        default:
            return "Error";
    }
}

std::string httpResponse(int status, const std::string& body, const std::string& contentType, bool keepAlive,
                         const std::string& extraHeaders = "")
{
    std::ostringstream str;
    str << "HTTP/1.1 " << status << " " << statusText(status) << "\r\n";
    if (!body.empty()) str << "Content-Type: " << contentType << "\r\n";
    str << "Content-Length: " << body.size() << "\r\n";
    str << "Connection: " << (keepAlive ? "keep-alive" : "close") << "\r\n";
    str << extraHeaders;
    str << "\r\n";
    str << body;
    return str.str();
}

// Allows requests without Origin (non-browser clients) and from local pages only (DNS rebinding protection). The
// opaque origin "null" (sandboxed iframes, data: pages) is rejected.
bool originAllowed(const std::map<std::string, std::string>& headers)
{
    auto it = headers.find("origin");
    if (it == headers.end() || it->second.empty()) return true;
    const std::string& o = it->second;
    for (const char* prefix : {"http://localhost", "https://localhost", "http://127.0.0.1", "https://127.0.0.1",
                               "http://[::1]", "https://[::1]"})
    {
        size_t n = std::strlen(prefix);
        if (o.compare(0, n, prefix) == 0 && (o.size() == n || o[n] == ':' || o[n] == '/')) return true;
    }
    return false;
}

// The Host header must name the loopback interface (DNS rebinding protection), optionally with the server's port.
bool hostAllowed(const std::map<std::string, std::string>& headers, int port)
{
    auto it = headers.find("host");
    if (it == headers.end()) return false;
    std::string host = toLower(it->second);
    for (std::string name : {"127.0.0.1", "localhost", "[::1]"})
    {
        if (host == name || host == name + ":" + std::to_string(port)) return true;
    }
    return false;
}

// application/json, optionally with parameters (e.g. "; charset=utf-8")
bool isJsonContentType(const std::map<std::string, std::string>& headers)
{
    auto it = headers.find("content-type");
    if (it == headers.end()) return false;
    std::string type = toLower(trim(it->second.substr(0, it->second.find(';'))));
    return type == "application/json";
}

// Parses a Content-Length value (decimal digits only). Returns false if it is invalid or too large.
bool parseContentLength(const std::string& value, size_t maxValue, size_t& result)
{
    if (value.empty() || value.size() > 18) return false;
    if (!std::all_of(value.begin(), value.end(), [](unsigned char c) { return c >= '0' && c <= '9'; })) return false;
    unsigned long long v = std::strtoull(value.c_str(), nullptr, 10);
    if (v > maxValue) return false;
    result = size_t(v);
    return true;
}

void setSendTimeout(socket_t s, int seconds)
{
#ifdef _WIN32
    DWORD ms = DWORD(seconds) * 1000;
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, (const char*)&ms, sizeof(ms));
#else
    timeval tv = {};
    tv.tv_sec  = seconds;
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, (const char*)&tv, sizeof(tv));
#endif
}

}  // namespace

//==================================================================================================
// ToolResult

McpServer::ToolResult McpServer::ToolResult::Text(const std::string& text)
{
    ToolResult r;
    r.addText(text);
    return r;
}

McpServer::ToolResult McpServer::ToolResult::Error(const std::string& message)
{
    ToolResult r = Text(message);
    r.isError    = true;
    return r;
}

// Indented JSON (json11 only writes compact JSON), easier to read for the agent.
static void prettyJson(const Json& j, std::string& out, int indent)
{
    std::string pad(indent * 2, ' ');
    std::string padInner((indent + 1) * 2, ' ');
    if (j.is_object() && !j.object_items().empty())
    {
        out += "{\n";
        bool first = true;
        for (auto& kv : j.object_items())
        {
            if (!first) out += ",\n";
            first = false;
            out += padInner + Json(kv.first).dump() + ": ";
            prettyJson(kv.second, out, indent + 1);
        }
        out += "\n" + pad + "}";
    }
    else if (j.is_array() && !j.array_items().empty())
    {
        // Short arrays of numbers (vectors) in one line
        bool numbers =
            std::all_of(j.array_items().begin(), j.array_items().end(), [](const Json& e) { return e.is_number(); });
        if (numbers && j.array_items().size() <= 4)
        {
            out += "[";
            for (size_t i = 0; i < j.array_items().size(); ++i)
            {
                if (i > 0) out += ", ";
                prettyJson(j.array_items()[i], out, 0);
            }
            out += "]";
            return;
        }
        out += "[\n";
        bool first = true;
        for (auto& e : j.array_items())
        {
            if (!first) out += ",\n";
            first = false;
            out += padInner;
            prettyJson(e, out, indent + 1);
        }
        out += "\n" + pad + "]";
    }
    else if (j.is_number())
    {
        // Integers in full, other numbers with 6 significant digits (plenty for the agent; json11 writes 17)
        char buf[40];
        double v = j.number_value();
        if (!std::isfinite(v))
            std::snprintf(buf, sizeof(buf), "null");  // JSON has no inf or NaN
        else if (v == std::floor(v) && std::abs(v) < 9007199254740992.0)
            std::snprintf(buf, sizeof(buf), "%.0f", v);
        else
            std::snprintf(buf, sizeof(buf), "%.6g", v);
        out += buf;
    }
    else
    {
        out += j.dump();
    }
}

// Compact JSON with short numbers
static void compactJson(const Json& j, std::string& out)
{
    if (j.is_object())
    {
        out += "{";
        bool first = true;
        for (auto& kv : j.object_items())
        {
            if (!first) out += ", ";
            first = false;
            out += Json(kv.first).dump() + ": ";
            compactJson(kv.second, out);
        }
        out += "}";
    }
    else if (j.is_array())
    {
        out += "[";
        bool first = true;
        for (auto& e : j.array_items())
        {
            if (!first) out += ", ";
            first = false;
            compactJson(e, out);
        }
        out += "]";
    }
    else
    {
        prettyJson(j, out, 0);
    }
}

std::string McpServer::FormatJson(const Json& json, bool pretty)
{
    std::string text;
    if (pretty)
        prettyJson(json, text, 0);
    else
        compactJson(json, text);
    return text;
}

McpServer::ToolResult McpServer::ToolResult::FromJson(const Json& json)
{
    return Text(FormatJson(json, true));
}

McpServer::ToolResult McpServer::ToolResult::Image(const std::vector<unsigned char>& data, const std::string& mimeType)
{
    ToolResult r;
    r.addImage(data, mimeType);
    return r;
}

McpServer::ToolResult& McpServer::ToolResult::addText(const std::string& text)
{
    content.push_back(Json::object{{"type", "text"}, {"text", text}});
    return *this;
}

McpServer::ToolResult& McpServer::ToolResult::addImage(const std::vector<unsigned char>& data,
                                                       const std::string& mimeType)
{
    content.push_back(
        Json::object{{"type", "image"}, {"data", Base64Encode(data.data(), data.size())}, {"mimeType", mimeType}});
    return *this;
}

//==================================================================================================
// Schema helpers

Json McpServer::ObjectSchema(const std::map<std::string, Json>& properties, const std::vector<std::string>& required)
{
    Json::object schema{{"type", "object"}, {"properties", Json::object(properties.begin(), properties.end())}};
    if (!required.empty())
    {
        Json::array r(required.begin(), required.end());
        schema["required"] = r;
    }
    return schema;
}

Json McpServer::NumberProperty(const std::string& description)
{
    return Json::object{{"type", "number"}, {"description", description}};
}

Json McpServer::IntegerProperty(const std::string& description)
{
    return Json::object{{"type", "integer"}, {"description", description}};
}

Json McpServer::BooleanProperty(const std::string& description)
{
    return Json::object{{"type", "boolean"}, {"description", description}};
}

Json McpServer::StringProperty(const std::string& description, const std::vector<std::string>& enumValues)
{
    Json::object p{{"type", "string"}, {"description", description}};
    if (!enumValues.empty()) p["enum"] = Json::array(enumValues.begin(), enumValues.end());
    return p;
}

Json McpServer::Vec3Property(const std::string& description)
{
    return Json::object{{"type", "array"},
                        {"items", Json::object{{"type", "number"}}},
                        {"minItems", 3},
                        {"maxItems", 3},
                        {"description", description}};
}

std::string McpServer::Base64Encode(const unsigned char* data, size_t size)
{
    static const char* table = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((size + 2) / 3 * 4);
    size_t i = 0;
    for (; i + 2 < size; i += 3)
    {
        uint32_t v = (uint32_t(data[i]) << 16) | (uint32_t(data[i + 1]) << 8) | uint32_t(data[i + 2]);
        out += table[(v >> 18) & 63];
        out += table[(v >> 12) & 63];
        out += table[(v >> 6) & 63];
        out += table[v & 63];
    }
    if (i < size)
    {
        uint32_t v = uint32_t(data[i]) << 16;
        if (i + 1 < size) v |= uint32_t(data[i + 1]) << 8;
        out += table[(v >> 18) & 63];
        out += table[(v >> 12) & 63];
        out += (i + 1 < size) ? table[(v >> 6) & 63] : '=';
        out += '=';
    }
    return out;
}

//==================================================================================================
// Server

struct McpServer::PendingCall
{
    Handler* handler = nullptr;
    Json arguments;
    std::promise<ToolResult> promise;
    std::atomic_bool answered{false};
    // Set once by whoever gets the call first: poll() (it runs) or callTool() on a timeout (it never runs)
    std::atomic_bool claimed{false};
    // When the caller stops waiting (the call's timeout, from when it was queued)
    std::chrono::steady_clock::time_point deadline;

    void answer(ToolResult result)
    {
        if (!answered.exchange(true)) promise.set_value(std::move(result));
    }
};

McpServer::McpServer(std::string name, std::string version, std::string instructions)
    : name(std::move(name)), version(std::move(version)), instructions(std::move(instructions))
{
}

McpServer::~McpServer()
{
    stop();
}

void McpServer::addTool(const std::string& toolName, const std::string& description, const Json& inputSchema,
                        Handler handler)
{
    SAIGA_ASSERT(!listening, "McpServer: add the tools before start()");
    // A second registration of a name would list the tool twice (every build)
    for (const Tool& t : tools)
        if (t.name == toolName) SAIGA_EXIT_ERROR("McpServer: the tool \"" + toolName + "\" is already registered");
    tools.push_back({toolName, description, inputSchema, std::move(handler)});
}

bool McpServer::start(int port, const std::string& address)
{
    SAIGA_ASSERT(!listening, "McpServer: already started");
    // Only 127.0.0.1: the Host check accepts only loopback names (127.0.0.1, localhost, [::1]), and the server has no
    // authentication
    SAIGA_ASSERT(address == "127.0.0.1", "McpServer: only 127.0.0.1 is supported");
#ifdef _WIN32
    static WinsockInit winsock;
#endif

    socket_t s = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s == InvalidSocket) return false;

    int yes = 1;
#ifdef _WIN32
    // SO_REUSEADDR would allow a second process to bind the same port on Windows
    setsockopt(s, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, (const char*)&yes, sizeof(yes));
#else
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, (const char*)&yes, sizeof(yes));
#endif

    sockaddr_in addr = {};
    addr.sin_family  = AF_INET;
    addr.sin_port    = htons((uint16_t)port);
    if (inet_pton(AF_INET, address.c_str(), &addr.sin_addr) != 1)
    {
        closeSocket(s);
        return false;
    }
    if (::bind(s, (sockaddr*)&addr, sizeof(addr)) != 0 || ::listen(s, 16) != 0)
    {
        closeSocket(s);
        return false;
    }

    // Non-blocking: a connection that is gone between poll() and accept() must not block the accept thread (and stop())
    suppressSigPipe(s);
    setBlocking(s, false);

    sockaddr_in bound = {};
    socklen_t len     = sizeof(bound);
    getsockname(s, (sockaddr*)&bound, &len);
    boundPort = ntohs(bound.sin_port);

    listenSocket = (intptr_t)s;
    stopping     = false;
    listening    = true;
    acceptThread = std::thread([this]() { acceptLoop(); });
    return true;
}

void McpServer::stop()
{
    if (!listening) return;
    stopping = true;
    if (acceptThread.joinable()) acceptThread.join();
    closeSocket((socket_t)listenSocket);
    listenSocket = -1;

    {
        std::lock_guard<std::mutex> lock(pendingMutex);
        for (auto& c : pendingCalls) c->answer(ToolResult::Error("The application is shutting down"));
        pendingCalls.clear();
    }
    joinFinishedConnections(true);
    listening = false;
}

void McpServer::poll()
{
    std::deque<std::shared_ptr<PendingCall>> calls;
    {
        std::lock_guard<std::mutex> lock(pendingMutex);
        calls.swap(pendingCalls);
    }
    for (auto& call : calls)
    {
        auto c = call;
        // Claimed by a timeout: the client was told that the call failed, so it must not run anymore
        if (c->claimed.exchange(true)) continue;
        // Handlers can read how long the caller still waits (callDeadline)
        currentDeadline = c->deadline;
        try
        {
            (*c->handler)(c->arguments, [c](ToolResult result) { c->answer(std::move(result)); });
        }
        catch (const std::exception& e)
        {
            c->answer(ToolResult::Error(std::string("Exception: ") + e.what()));
        }
    }
    // Listeners may add listeners (kept) and remove themselves (return false). A throwing listener is removed; the
    // others still run, then the first exception is rethrown.
    std::vector<std::function<bool()>> current;
    current.swap(pollListeners);
    std::exception_ptr first;
    for (auto& l : current)
    {
        try
        {
            if (l()) pollListeners.push_back(std::move(l));
        }
        catch (...)
        {
            if (!first) first = std::current_exception();
        }
    }
    if (first) std::rethrow_exception(first);
}

McpServer::ToolResult McpServer::callTool(const std::string& toolName, const Json& arguments)
{
    auto it = std::find_if(tools.begin(), tools.end(), [&](const Tool& t) { return t.name == toolName; });
    SAIGA_ASSERT(it != tools.end());

    auto call       = std::make_shared<PendingCall>();
    call->handler   = &it->handler;
    call->arguments = arguments.is_object() ? arguments : Json(Json::object{});
    auto future     = call->promise.get_future();
    call->deadline = std::chrono::steady_clock::now() + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                                            std::chrono::duration<double>(toolTimeoutSeconds));
    {
        std::lock_guard<std::mutex> lock(pendingMutex);
        if (stopping) return ToolResult::Error("The application is shutting down");
        // Bounded: if the main loop does not call poll(), calls must not pile up
        if (pendingCalls.size() >= maxPendingCalls)
            return ToolResult::Error("Too many pending tool calls (is the main loop running and calling poll()?)");
        pendingCalls.push_back(call);
    }

    // On a timeout the call is claimed: if poll() has not started it yet, it never runs; if it has started, the error
    // says so (its effects may have happened, a deferred answer arriving later is dropped).
    const auto deadline = call->deadline;
    while (future.wait_for(std::chrono::milliseconds(50)) != std::future_status::ready)
    {
        if (stopping)
        {
            call->answer(ToolResult::Error("The application is shutting down"));
            break;
        }
        if (std::chrono::steady_clock::now() > deadline)
        {
            bool started = call->claimed.exchange(true);
            if (!started)
            {
                // Never runs: remove it (with its arguments) from the queue
                std::lock_guard<std::mutex> lock(pendingMutex);
                pendingCalls.erase(std::remove(pendingCalls.begin(), pendingCalls.end(), call), pendingCalls.end());
            }
            call->answer(ToolResult::Error(
                started ? "Timeout: the tool started but did not answer in time (its effects may have happened)"
                        : "Timeout: the tool call was not run (is the main loop running and calling poll()?)"));
            break;
        }
    }
    return future.get();
}

Json McpServer::handleMessage(const Json& message)
{
    if (!message.is_object()) return rpcError(nullptr, -32600, "Invalid request");

    const Json& id     = message["id"];
    std::string method = message["method"].string_value();
    // Notifications (e.g. notifications/initialized) and responses have no answer.
    if (method.empty() || id.is_null()) return Json();

    const Json& params = message["params"];

    if (method == "initialize")
    {
        std::string protocol = params["protocolVersion"].string_value();
        if (protocol.empty()) protocol = "2025-06-18";
        Json::object result{{"protocolVersion", protocol},
                            {"capabilities", Json::object{{"tools", Json::object{{"listChanged", false}}}}},
                            {"serverInfo", Json::object{{"name", name}, {"version", version}}}};
        if (!instructions.empty()) result["instructions"] = instructions;
        return rpcResult(id, result);
    }
    if (method == "ping")
    {
        return rpcResult(id, Json::object{});
    }
    if (method == "tools/list")
    {
        Json::array list;
        for (auto& t : tools)
        {
            list.push_back(
                Json::object{{"name", t.name}, {"description", t.description}, {"inputSchema", t.inputSchema}});
        }
        return rpcResult(id, Json::object{{"tools", list}});
    }
    if (method == "tools/call")
    {
        std::string toolName = params["name"].string_value();
        bool known = std::any_of(tools.begin(), tools.end(), [&](const Tool& t) { return t.name == toolName; });
        if (!known) return rpcError(id, -32602, "Unknown tool: " + toolName);

        ToolResult r = callTool(toolName, params["arguments"]);
        return rpcResult(id, Json::object{{"content", r.content}, {"isError", r.isError}});
    }
    return rpcError(id, -32601, "Method not found: " + method);
}

std::string McpServer::handleHttp(const std::string& method, const std::map<std::string, std::string>& headers,
                                  const std::string& body, bool& keepAlive)
{
    if (!originAllowed(headers)) return httpResponse(403, "Forbidden origin", "text/plain", keepAlive);
    if (!hostAllowed(headers, boundPort)) return httpResponse(403, "Forbidden host", "text/plain", keepAlive);

    if (method != "POST")
    {
        // No server initiated SSE stream (GET) and no sessions (DELETE)
        return httpResponse(405, "", "", keepAlive, "Allow: POST\r\n");
    }
    // Requiring JSON makes every browser request a CORS preflight (OPTIONS), which is refused above
    if (!isJsonContentType(headers))
        return httpResponse(415, "Content-Type must be application/json", "text/plain", keepAlive);

    std::string err;
    Json message = Json::parse(body, err);
    if (!err.empty())
    {
        return httpResponse(400, rpcError(nullptr, -32700, "Parse error: " + err).dump(), "application/json",
                            keepAlive);
    }

    Json response;
    if (message.is_array())
    {
        // Batch (protocol versions <= 2025-03-26)
        Json::array responses;
        for (auto& m : message.array_items())
        {
            Json r = handleMessage(m);
            if (!r.is_null()) responses.push_back(r);
        }
        if (!responses.empty()) response = responses;
    }
    else
    {
        response = handleMessage(message);
    }

    if (response.is_null()) return httpResponse(202, "", "", keepAlive);
    return httpResponse(200, response.dump(), "application/json", keepAlive);
}

void McpServer::acceptLoop()
{
    socket_t ls = (socket_t)listenSocket;
    while (!stopping)
    {
        joinFinishedConnections(false);
        if (!waitReadable(ls, 100)) continue;
        socket_t s = ::accept(ls, nullptr, nullptr);
        if (s == InvalidSocket)
        {
            // No connection (it was gone again; the listener is non-blocking) or e.g. out of file descriptors: the
            // pending connection keeps the socket readable, do not spin
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        suppressSigPipe(s);
        setBlocking(s, true);

        int yes = 1;
        setsockopt(s, IPPROTO_TCP, TCP_NODELAY, (const char*)&yes, sizeof(yes));
        // A client that stops reading must not block its connection thread (and stop()) forever
        setSendTimeout(s, 10);

        size_t open;
        {
            std::lock_guard<std::mutex> lock(connectionMutex);
            open = connections.size();
        }
        if (open >= maxConnections)
        {
            sendAll(s, httpResponse(503, "Too many connections", "text/plain", false), &stopping);
            closeSocket(s);
            continue;
        }

        auto connection    = std::make_unique<Connection>();
        connection->socket = (intptr_t)s;
        Connection* c      = connection.get();
        std::lock_guard<std::mutex> lock(connectionMutex);
        try
        {
            connection->thread = std::thread([this, c]() { connectionLoop(c); });
        }
        catch (const std::system_error&)
        {
            // No thread could be created: drop this connection instead of terminating the application
            closeSocket(s);
            continue;
        }
        connections.push_back(std::move(connection));
    }
}

void McpServer::joinFinishedConnections(bool all)
{
    std::vector<std::unique_ptr<Connection>> finished;
    {
        std::lock_guard<std::mutex> lock(connectionMutex);
        for (auto& c : connections)
        {
            if (all || c->done) finished.push_back(std::move(c));
        }
        connections.erase(std::remove(connections.begin(), connections.end(), nullptr), connections.end());
    }
    for (auto& c : finished)
    {
        if (c->thread.joinable()) c->thread.join();
    }
}

void McpServer::connectionLoop(Connection* connection)
{
    try
    {
        serveConnection(connection);
    }
    catch (const std::exception& e)
    {
        std::cerr << "McpServer: connection error: " << e.what() << std::endl;
    }
    closeSocket((socket_t)connection->socket);
    connection->done = true;
}

void McpServer::serveConnection(Connection* connection)
{
    socket_t s = (socket_t)connection->socket;
    std::string buffer;
    // Requests larger than this are rejected
    const size_t maxRequestSize = 64 * 1024 * 1024;
    // Idle keep-alive connections are closed after this time
    const auto idleTimeout = std::chrono::seconds(120);
    auto lastActivity      = std::chrono::steady_clock::now();
    bool keepAlive         = true;

    while (keepAlive && !stopping)
    {
        // Read the header
        size_t headerEnd;
        while ((headerEnd = buffer.find("\r\n\r\n")) == std::string::npos)
        {
            if (stopping || std::chrono::steady_clock::now() - lastActivity > idleTimeout) return;
            if (!waitReadable(s, 100)) continue;
            char chunk[16384];
            auto n = ::recv(s, chunk, sizeof(chunk), 0);
            if (n <= 0) return;
            buffer.append(chunk, size_t(n));
            lastActivity = std::chrono::steady_clock::now();
            if (buffer.size() > maxRequestSize) return;
        }

        {
            std::istringstream header(buffer.substr(0, headerEnd));
            std::string requestLine;
            std::getline(header, requestLine);
            std::istringstream rl(requestLine);
            std::string method, target, httpVersion;
            rl >> method >> target >> httpVersion;

            std::map<std::string, std::string> headers;
            std::string line;
            while (std::getline(header, line))
            {
                auto colon = line.find(':');
                if (colon == std::string::npos) continue;
                headers[toLower(trim(line.substr(0, colon)))] = trim(line.substr(colon + 1));
            }

            if (headers.count("transfer-encoding"))
            {
                sendAll(s, httpResponse(501, "Transfer-Encoding is not supported", "text/plain", false), &stopping);
                return;
            }
            size_t contentLength = 0;
            if (headers.count("content-length"))
            {
                // Invalid, or larger than any size_t: 400. Valid but too large: 413.
                if (!parseContentLength(headers["content-length"], ~size_t(0), contentLength))
                {
                    sendAll(s, httpResponse(400, "Invalid Content-Length", "text/plain", false), &stopping);
                    return;
                }
            }
            if (contentLength > maxRequestSize)
            {
                sendAll(s, httpResponse(413, "", "", false), &stopping);
                return;
            }
            keepAlive = toLower(headers["connection"]) != "close" && httpVersion != "HTTP/1.0";

            // Read the body
            size_t requestSize = headerEnd + 4 + contentLength;
            while (buffer.size() < requestSize)
            {
                if (stopping || std::chrono::steady_clock::now() - lastActivity > idleTimeout) return;
                if (!waitReadable(s, 100)) continue;
                char chunk[16384];
                auto n = ::recv(s, chunk, sizeof(chunk), 0);
                if (n <= 0) return;
                buffer.append(chunk, size_t(n));
                lastActivity = std::chrono::steady_clock::now();
            }
            std::string body = buffer.substr(headerEnd + 4, contentLength);
            buffer.erase(0, requestSize);

            std::string response = handleHttp(method, headers, body, keepAlive);
            if (!sendAll(s, response, &stopping)) return;
            lastActivity = std::chrono::steady_clock::now();
        }
    }
}

}  // namespace Saiga
