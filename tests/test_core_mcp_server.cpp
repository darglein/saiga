/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/util/mcp/McpServer.h"

#include "gtest/gtest.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <future>
#include <limits>
#include <thread>

#ifndef _WIN32
#    include <arpa/inet.h>
#    include <netinet/in.h>
#    include <sys/socket.h>
#    include <unistd.h>
#endif

using namespace Saiga;
using Json = McpServer::Json;

static Json Request(const std::string& method, const Json& params, int id = 1)
{
    return Json::object{{"jsonrpc", "2.0"}, {"id", id}, {"method", method}, {"params", params}};
}

// Calls poll() in the background, like the main loop of an application.
struct Poller
{
    McpServer& server;
    std::atomic_bool stop{false};
    std::thread thread;
    explicit Poller(McpServer& s) : server(s)
    {
        thread = std::thread(
            [this]()
            {
                while (!stop)
                {
                    server.poll();
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            });
    }
    ~Poller()
    {
        stop = true;
        thread.join();
    }
};

TEST(McpServer, Base64)
{
    auto enc = [](const std::string& s) { return McpServer::Base64Encode((const unsigned char*)s.data(), s.size()); };
    EXPECT_EQ(enc(""), "");
    EXPECT_EQ(enc("f"), "Zg==");
    EXPECT_EQ(enc("fo"), "Zm8=");
    EXPECT_EQ(enc("foo"), "Zm9v");
    EXPECT_EQ(enc("foobar"), "Zm9vYmFy");
}

TEST(McpServer, InitializeAndList)
{
    McpServer server("test", "1.2", "Test instructions");
    server.addTool("add", "Adds two numbers", McpServer::ObjectSchema({{"a", McpServer::NumberProperty("a")}}, {"a"}),
                   [](const Json&, McpServer::Respond r) { r(McpServer::ToolResult::Text("x")); });

    Json init = server.handleMessage(Request("initialize", Json::object{{"protocolVersion", "2025-03-26"}}));
    EXPECT_EQ(init["result"]["protocolVersion"].string_value(), "2025-03-26");
    EXPECT_EQ(init["result"]["serverInfo"]["name"].string_value(), "test");
    EXPECT_EQ(init["result"]["instructions"].string_value(), "Test instructions");
    EXPECT_TRUE(init["result"]["capabilities"]["tools"].is_object());

    // Notifications have no response
    Json note = server.handleMessage(Json::object{{"jsonrpc", "2.0"}, {"method", "notifications/initialized"}});
    EXPECT_TRUE(note.is_null());

    Json list = server.handleMessage(Request("tools/list", Json::object{}));
    ASSERT_EQ(list["result"]["tools"].array_items().size(), 1u);
    EXPECT_EQ(list["result"]["tools"][0]["name"].string_value(), "add");
    EXPECT_EQ(list["result"]["tools"][0]["inputSchema"]["required"][0].string_value(), "a");

    EXPECT_EQ(server.handleMessage(Request("unknown/method", Json::object{}))["error"]["code"].int_value(), -32601);
    EXPECT_EQ(server.handleMessage(Request("tools/call", Json::object{{"name", "nope"}}))["error"]["code"].int_value(),
              -32602);
}

TEST(McpServer, ToolCallRunsInPoll)
{
    McpServer server("test");
    std::thread::id handlerThread;
    server.addTool(
        "add", "Adds a and b", McpServer::ObjectSchema({}),
        [&](const Json& args, McpServer::Respond respond)
        {
            handlerThread = std::this_thread::get_id();
            respond(McpServer::ToolResult::Text(std::to_string(args["a"].int_value() + args["b"].int_value())));
        });

    std::thread::id pollThread;
    {
        Poller poller(server);
        pollThread = poller.thread.get_id();
        Json r     = server.handleMessage(
            Request("tools/call", Json::object{{"name", "add"}, {"arguments", Json::object{{"a", 2}, {"b", 3}}}}));
        EXPECT_FALSE(r["result"]["isError"].bool_value());
        EXPECT_EQ(r["result"]["content"][0]["text"].string_value(), "5");
    }
    EXPECT_EQ(handlerThread, pollThread);
}

TEST(McpServer, DeferredResponseAndTimeout)
{
    McpServer server("test");
    McpServer::Respond pending;
    server.addTool("later", "Answers on the next poll", McpServer::ObjectSchema({}),
                   [&](const Json&, McpServer::Respond respond) { pending = respond; });
    server.addTool("never", "Never answers", McpServer::ObjectSchema({}), [](const Json&, McpServer::Respond) {});
    server.toolTimeoutSeconds = 1.0;

    std::atomic_bool stop{false};
    std::thread poller(
        [&]()
        {
            while (!stop)
            {
                server.poll();
                // Answer a deferred call one poll later (like a screenshot after the next frame)
                if (pending)
                {
                    auto r  = pending;
                    pending = nullptr;
                    r(McpServer::ToolResult::Image({1, 2, 3}, "image/png"));
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });

    Json r = server.handleMessage(Request("tools/call", Json::object{{"name", "later"}}));
    EXPECT_EQ(r["result"]["content"][0]["type"].string_value(), "image");
    EXPECT_EQ(r["result"]["content"][0]["data"].string_value(), "AQID");

    Json t = server.handleMessage(Request("tools/call", Json::object{{"name", "never"}}));
    EXPECT_TRUE(t["result"]["isError"].bool_value());

    stop = true;
    poller.join();
}

// Regression: a call that timed out was still executed by a later poll(), although the client was told it failed.
TEST(McpServer, TimedOutCallIsNotExecuted)
{
    McpServer server("test");
    std::atomic_int executed{0};
    server.addTool("count", "Counts", McpServer::ObjectSchema({}),
                   [&](const Json&, McpServer::Respond r) { executed++, r(McpServer::ToolResult::Text("ok")); });
    server.toolTimeoutSeconds = 0.1;

    // No poll() while the call waits: it times out
    Json t = server.handleMessage(Request("tools/call", Json::object{{"name", "count"}}));
    EXPECT_TRUE(t["result"]["isError"].bool_value());
    server.poll();
    EXPECT_EQ(executed, 0);
}

// Regression: poll() could start a call after callTool() had already reported its timeout. Now a timeout claims the
// call: a call that was not started never runs, and the error for a started one says that it ran.
TEST(McpServer, TimeoutOfStartedCall)
{
    McpServer server("test");
    McpServer::Respond deferred;
    server.addTool("later", "Answers too late", McpServer::ObjectSchema({}),
                   [&](const Json&, McpServer::Respond r) { deferred = r; });
    // Long enough that poll() starts the call before the timeout also on a loaded machine
    server.toolTimeoutSeconds = 2;

    auto result = std::async(std::launch::async, [&]
                             { return server.handleMessage(Request("tools/call", Json::object{{"name", "later"}})); });
    // Bounded: if the call timed out before poll() started it, the handler never runs (fail instead of hanging)
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!deferred && std::chrono::steady_clock::now() < deadline)
    {
        server.poll();
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    ASSERT_TRUE(deferred) << "the tool call was never started";
    Json t = result.get();
    EXPECT_TRUE(t["result"]["isError"].bool_value());
    EXPECT_NE(t["result"]["content"][0]["text"].string_value().find("started"), std::string::npos);
    // The late answer is dropped
    deferred(McpServer::ToolResult::Text("late"));
}

TEST(McpServer, FormatJson)
{
    Json j = Json::object{{"a", 0.1 + 0.2}, {"v", Json::array{1.0 / 3.0, 2, 3}}};
    EXPECT_EQ(McpServer::FormatJson(j, false), "{\"a\": 0.3, \"v\": [0.333333, 2, 3]}");
    // Regression: integers were rounded to 6 significant digits (12345678 -> 1.23457e+07)
    EXPECT_EQ(McpServer::FormatJson(Json::array{12345678, -2000000000.0, 1e20}, false),
              "[12345678, -2000000000, 1e+20]");
    // JSON has no inf or NaN: non-finite numbers are written as null
    const double inf = std::numeric_limits<double>::infinity();
    EXPECT_EQ(McpServer::FormatJson(Json::array{inf, -inf, std::nan("")}, false), "[null, null, null]");
    EXPECT_EQ(McpServer::FormatJson(Json::object{{"x", inf}}, true), "{\n  \"x\": null\n}");
}

#ifndef _WIN32
// Sends a raw HTTP request to the server and returns the response (until the server closes the connection)
static std::string HttpRaw(int port, const std::string& req)
{
    int s            = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr = {};
    addr.sin_family  = AF_INET;
    addr.sin_port    = htons(uint16_t(port));
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
    if (connect(s, (sockaddr*)&addr, sizeof(addr)) != 0)
    {
        close(s);
        return "";
    }
    send(s, req.data(), req.size(), 0);
    std::string response;
    char buf[4096];
    ssize_t n;
    while ((n = recv(s, buf, sizeof(buf), 0)) > 0) response.append(buf, size_t(n));
    close(s);
    return response;
}

// A POST request with the default headers (Host, Content-Type) unless they are replaced
static std::string HttpPost(int port, const std::string& body, const std::string& extraHeaders = "",
                            const std::string& host = "127.0.0.1", const std::string& contentType = "application/json")
{
    return HttpRaw(port, "POST /mcp HTTP/1.1\r\nHost: " + host + "\r\nContent-Type: " + contentType +
                             "\r\nConnection: close\r\n" + extraHeaders +
                             "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body);
}

TEST(McpServer, Http)
{
    McpServer server("test");
    server.addTool("echo", "Echo", McpServer::ObjectSchema({}), [](const Json& args, McpServer::Respond r)
                   { r(McpServer::ToolResult::Text(args["text"].string_value())); });
    ASSERT_TRUE(server.start(0));
    ASSERT_GT(server.port(), 0);
    Poller poller(server);

    std::string r = HttpPost(
        server.port(),
        Request("tools/call", Json::object{{"name", "echo"}, {"arguments", Json::object{{"text", "hello"}}}}).dump());
    ASSERT_EQ(r.compare(0, 15, "HTTP/1.1 200 OK"), 0) << r;
    std::string err;
    Json body = Json::parse(r.substr(r.find("\r\n\r\n") + 4), err);
    EXPECT_TRUE(err.empty());
    EXPECT_EQ(body["result"]["content"][0]["text"].string_value(), "hello");

    // Notifications: 202 without body
    r = HttpPost(server.port(), Json(Json::object{{"jsonrpc", "2.0"}, {"method", "notifications/initialized"}}).dump());
    EXPECT_EQ(r.compare(0, 12, "HTTP/1.1 202"), 0) << r;

    // DNS rebinding protection: foreign origins are rejected
    r = HttpPost(server.port(), Request("tools/list", Json::object{}).dump(), "Origin: http://evil.example\r\n");
    EXPECT_EQ(r.compare(0, 12, "HTTP/1.1 403"), 0) << r;

    // Invalid JSON
    r = HttpPost(server.port(), "{not json");
    EXPECT_EQ(r.compare(0, 12, "HTTP/1.1 400"), 0) << r;

    std::string list = Request("tools/list", Json::object{}).dump();

    // Regression: the opaque origin "null" (sandboxed iframes) was allowed
    r = HttpPost(server.port(), list, "Origin: null\r\n");
    EXPECT_EQ(r.compare(0, 12, "HTTP/1.1 403"), 0) << r;

    // Regression: a cross-site "simple request" (text/plain, no CORS preflight) could call tools
    r = HttpPost(server.port(), list, "", "127.0.0.1", "text/plain");
    EXPECT_EQ(r.compare(0, 12, "HTTP/1.1 415"), 0) << r;
    r = HttpPost(server.port(), list, "", "127.0.0.1", "application/json; charset=utf-8");
    EXPECT_EQ(r.compare(0, 12, "HTTP/1.1 200"), 0) << r;

    // DNS rebinding: the Host must be the loopback interface
    r = HttpPost(server.port(), list, "", "evil.example");
    EXPECT_EQ(r.compare(0, 12, "HTTP/1.1 403"), 0) << r;
    r = HttpPost(server.port(), list, "", "localhost:" + std::to_string(server.port()));
    EXPECT_EQ(r.compare(0, 12, "HTTP/1.1 200"), 0) << r;

    // Regression: an invalid Content-Length threw on the connection thread and terminated the application
    for (std::string length : {"abc", "", "99999999999999999999999", "-1"})
    {
        r = HttpRaw(server.port(),
                    "POST /mcp HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                    "Content-Length: " +
                        length + "\r\n\r\n");
        EXPECT_EQ(r.compare(0, 12, "HTTP/1.1 400"), 0) << length << ": " << r;
    }
    // Chunked bodies are not supported (they were silently read as empty)
    r = HttpRaw(server.port(),
                "POST /mcp HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                "Transfer-Encoding: chunked\r\n\r\n0\r\n\r\n");
    EXPECT_EQ(r.compare(0, 12, "HTTP/1.1 501"), 0) << r;
    // Still serving
    r = HttpPost(server.port(), list);
    EXPECT_EQ(r.compare(0, 12, "HTTP/1.1 200"), 0) << r;

    server.stop();
    EXPECT_FALSE(server.running());
}
#endif
