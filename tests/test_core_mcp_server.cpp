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

#include "mcp_test_helper.h"

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
    return McpRequest(method, params, id);
}

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

// Regression (S14): a second registration of a name listed the tool twice; it stops the program in every build
TEST(McpServerDeathTest, DuplicateToolName)
{
    auto twice = []()
    {
        McpServer server("test");
        auto handler = [](const Json&, McpServer::Respond respond) { respond(McpServer::ToolResult::Text("")); };
        server.addTool("a", "first", McpServer::ObjectSchema({}), handler);
        server.addTool("a", "second", McpServer::ObjectSchema({}), handler);
    };
    EXPECT_DEATH(twice(), "already registered");
}

TEST(McpServer, DeferredResponseAndTimeout)
{
    McpServer server("test");
    McpServer::Respond pending;
    server.addTool("later", "Answers on the next poll", McpServer::ObjectSchema({}),
                   [&](const Json&, McpServer::Respond respond) { pending = respond; });
    server.addTool("never", "Never answers", McpServer::ObjectSchema({}), [](const Json&, McpServer::Respond) {});
    // "later" is answered one poll (~1 ms) after its call, "never" waits for the timeout
    server.toolTimeoutSeconds = 0.5;

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

// Regression: timed-out calls stayed in the queue while poll() was not called, so the queue grew without bound. They
// are removed now, and the queue is limited (a full queue rejects new calls).
TEST(McpServer, TimedOutCallsDoNotFillTheQueue)
{
    McpServer server("test");
    std::atomic_int executed{0};
    server.addTool("count", "Counts", McpServer::ObjectSchema({}),
                   [&](const Json&, McpServer::Respond r) { executed++, r(McpServer::ToolResult::Text("ok")); });
    server.toolTimeoutSeconds = 0.005;

    // More timed-out calls than the queue holds: each reports "not run", none reports a full queue
    for (int i = 0; i < 80; ++i)
    {
        Json t = server.handleMessage(Request("tools/call", Json::object{{"name", "count"}}));
        ASSERT_TRUE(t["result"]["isError"].bool_value());
        ASSERT_NE(t["result"]["content"][0]["text"].string_value().find("not run"), std::string::npos) << i;
    }
    server.poll();
    EXPECT_EQ(executed, 0);

    // A full queue (poll() is not called while the calls wait) rejects further calls: of 70 concurrent calls 64 wait
    // and 6 are rejected at once (only the rejected ones can finish before poll())
    executed                  = 0;
    server.toolTimeoutSeconds = 30;
    std::vector<std::future<Json>> calls;
    for (int i = 0; i < 70; ++i)
        calls.push_back(
            std::async(std::launch::async,
                       [&] { return server.handleMessage(Request("tools/call", Json::object{{"name", "count"}})); }));
    auto ready = [&]
    {
        int n = 0;
        for (auto& c : calls) n += c.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
        return n;
    };
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    while (ready() < 6 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    ASSERT_EQ(ready(), 6);
    while (ready() < 70 && std::chrono::steady_clock::now() < deadline)
    {
        server.poll();
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    int rejected = 0;
    for (auto& c : calls)
        rejected += c.get()["result"]["content"][0]["text"].string_value().find("Too many") != std::string::npos;
    EXPECT_EQ(rejected, 6);
    EXPECT_EQ(executed, 64);
}

// Regression: poll() could start a call after callTool() had already reported its timeout. Now a timeout claims the
// call: a call that was not started never runs, and the error for a started one says that it ran.
TEST(McpServer, TimeoutOfStartedCall)
{
    McpServer server("test");
    McpServer::Respond deferred;
    server.addTool("later", "Answers too late", McpServer::ObjectSchema({}),
                   [&](const Json&, McpServer::Respond r) { deferred = r; });
    // Long enough that poll() (every 5 ms) starts the call before the timeout also on a loaded machine
    server.toolTimeoutSeconds = 1;

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

// stop() returns promptly while a client that never reads blocks a large response (sends are non-blocking and check
// the stop flag; a blocking send() could wait for its whole send timeout).
TEST(McpServer, StopWithSlowReader)
{
#ifdef _WIN32
    GTEST_SKIP() << "raw client sockets are only implemented for POSIX in this test";
#else
    McpServer server("test");
    std::string big(32 * 1024 * 1024, 'x');
    server.addTool("big", "Big result", McpServer::ObjectSchema({}),
                   [&](const Json&, McpServer::Respond r) { r(McpServer::ToolResult::Text(big)); });
    ASSERT_TRUE(server.start(0));
    Poller poller(server);

    // A client with a small receive buffer that sends the request and never reads
    int c   = ::socket(AF_INET, SOCK_STREAM, 0);
    int buf = 4096;
    setsockopt(c, SOL_SOCKET, SO_RCVBUF, &buf, sizeof(buf));
    sockaddr_in addr = {};
    addr.sin_family  = AF_INET;
    addr.sin_port    = htons(uint16_t(server.port()));
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
    ASSERT_EQ(::connect(c, (sockaddr*)&addr, sizeof(addr)), 0);
    std::string body = Request("tools/call", Json::object{{"name", "big"}, {"arguments", Json::object{}}}).dump();
    std::string request =
        "POST /mcp HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: " +
        std::to_string(body.size()) + "\r\n\r\n" + body;
    ASSERT_EQ(::send(c, request.data(), request.size(), 0), (ssize_t)request.size());

    // The server is now writing the response into the full socket buffer
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    auto t0 = std::chrono::steady_clock::now();
    server.stop();
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    EXPECT_LT(seconds, 3.0) << "stop() waited for the slow reader";
    ::close(c);
#endif
}

// callLocal (the debug UI's console, S19c): queued like a remote call, run by poll(), answered on the polling thread
TEST(McpServer, CallLocal)
{
    McpServer server("test");
    McpServer::Respond later;
    std::atomic_int executed{0};
    server.addTool("count", "Counts", McpServer::ObjectSchema({}),
                   [&](const Json& a, McpServer::Respond r)
                   {
                       executed++;
                       r(McpServer::ToolResult::Text(a.dump()));
                   });
    server.addTool("later", "Answers later", McpServer::ObjectSchema({}),
                   [&](const Json&, McpServer::Respond r) { later = std::move(r); });
    EXPECT_EQ(server.toolNames(), (std::vector<std::string>{"count", "later"}));
    std::string description;
    Json schema;
    EXPECT_TRUE(server.toolInfo("later", description, schema));
    EXPECT_EQ(description, "Answers later");
    EXPECT_FALSE(server.toolInfo("nope", description, schema));

    std::vector<McpServer::ToolResult> answers;
    std::vector<std::thread::id> threads;
    auto respond = [&](McpServer::ToolResult r)
    {
        answers.push_back(r);
        threads.push_back(std::this_thread::get_id());
    };
    server.callLocal("count", Json::object{{"x", 1}}, respond);
    EXPECT_EQ(executed, 0);
    EXPECT_TRUE(answers.empty());
    server.poll();
    EXPECT_EQ(executed, 1);
    ASSERT_EQ(answers.size(), 1u);
    EXPECT_FALSE(answers[0].isError);
    EXPECT_EQ(answers[0].content[0]["text"].string_value(), R"({"x": 1})");

    // Unknown tool: an error through respond, in the next poll
    server.callLocal("nope", Json::object{}, respond);
    EXPECT_EQ(answers.size(), 1u);
    server.poll();
    ASSERT_EQ(answers.size(), 2u);
    EXPECT_TRUE(answers[1].isError);

    // A deferred answer from another thread is delivered by poll() on the polling thread
    server.callLocal("later", Json::object{}, respond);
    server.poll();
    ASSERT_TRUE(later);
    std::thread([&]() { later(McpServer::ToolResult::Text("late")); }).join();
    EXPECT_EQ(answers.size(), 2u);
    server.poll();
    ASSERT_EQ(answers.size(), 3u);
    EXPECT_EQ(threads[2], std::this_thread::get_id());

    // The timeout is enforced by poll() (no thread waits for a local call)
    server.toolTimeoutSeconds = 0.05;
    later                     = {};
    server.callLocal("later", Json::object{}, respond);
    server.poll();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    server.poll();
    ASSERT_EQ(answers.size(), 4u);
    EXPECT_TRUE(answers[3].isError);
    EXPECT_NE(answers[3].content[0]["text"].string_value().find("Timeout"), std::string::npos);
    // The late answer is dropped
    later(McpServer::ToolResult::Text("too late"));
    server.poll();
    EXPECT_EQ(answers.size(), 4u);

    // stop() answers pending local calls although the server never listened
    server.callLocal("count", Json::object{}, respond);
    server.stop();
    ASSERT_EQ(answers.size(), 5u);
    EXPECT_TRUE(answers[4].isError);
    EXPECT_EQ(executed, 1);
}

// The call observer (the debug UI's MCP call log, S19c): records for remote and local calls, delivered by poll()
TEST(McpServer, CallObserver)
{
    McpServer server("test");
    server.addTool("ok", "Answers", McpServer::ObjectSchema({}),
                   [](const Json&, McpServer::Respond r) { r(McpServer::ToolResult::Text("fine\n  answer")); });
    server.addTool("fail", "Fails", McpServer::ObjectSchema({}),
                   [](const Json&, McpServer::Respond r) { r(McpServer::ToolResult::Error("bad")); });
    std::vector<McpServer::CallRecord> records;
    std::vector<std::thread::id> threads;
    std::atomic_int recordCount{0};  // read by the main thread while the poller appends
    server.setCallObserver(
        [&](const McpServer::CallRecord& r)
        {
            records.push_back(r);
            threads.push_back(std::this_thread::get_id());
            recordCount++;
        });
    using R = McpServer::CallRecord;

    // A remote call (handleMessage blocks: a poller answers it)
    {
        std::atomic_bool stop{false};
        std::thread poller(
            [&]()
            {
                while (!stop)
                {
                    server.poll();
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            });
        server.handleMessage(
            Request("tools/call", Json::object{{"name", "ok"}, {"arguments", Json::object{{"a", 1}}}}));
        // The Answered record is delivered by the poll after the answer
        for (int i = 0; i < 1000 && recordCount < 2; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        stop = true;
        poller.join();
    }
    ASSERT_EQ(records.size(), 2u);
    EXPECT_EQ(records[0].phase, R::Phase::Started);
    EXPECT_EQ(records[1].phase, R::Phase::Answered);
    EXPECT_EQ(records[0].id, records[1].id);
    EXPECT_EQ(records[1].tool, "ok");
    EXPECT_EQ(records[1].source, R::Source::Remote);
    EXPECT_EQ(records[1].arguments["a"], Json(1));
    EXPECT_FALSE(records[1].isError);
    EXPECT_GE(records[1].durationMs, 0);
    EXPECT_EQ(records[1].answer, "fine answer");
    EXPECT_EQ(threads[0], threads[1]);

    // A local call with an error
    records.clear();
    threads.clear();
    server.callLocal("fail", Json::object{}, [](McpServer::ToolResult) {});
    server.poll();
    ASSERT_EQ(records.size(), 2u);
    EXPECT_EQ(records[1].source, R::Source::Local);
    EXPECT_TRUE(records[1].isError);
    EXPECT_EQ(records[1].answer, "bad");
    EXPECT_EQ(threads[1], std::this_thread::get_id());
    EXPECT_NE(records[1].id, 0u);

    // A remote call that times out without poll(): the HTTP thread gives up, the record arrives with the next poll()
    // on the polling thread
    records.clear();
    threads.clear();
    server.toolTimeoutSeconds = 0.05;
    std::thread caller([&]() { server.handleMessage(Request("tools/call", Json::object{{"name", "ok"}})); });
    caller.join();
    EXPECT_TRUE(records.empty());
    server.poll();
    ASSERT_EQ(records.size(), 1u);
    EXPECT_EQ(records[0].phase, R::Phase::Answered);
    EXPECT_TRUE(records[0].isError);
    EXPECT_EQ(threads[0], std::this_thread::get_id());
}

// Regressions (CodeRabbit, S19c): a local call queued before more tools were registered kept a pointer into the tool
// list (reallocated by addTool); a local call that waited past its deadline still ran its handler
TEST(McpServer, CallLocalQueuedBeforeMoreToolsAndExpired)
{
    McpServer server("test");
    int executed = 0;
    server.addTool("count", "Counts", McpServer::ObjectSchema({}),
                   [&](const Json&, McpServer::Respond r)
                   {
                       executed++;
                       r(McpServer::ToolResult::Text("ok"));
                   });
    std::vector<McpServer::ToolResult> answers;
    server.callLocal("count", Json::object{}, [&](McpServer::ToolResult r) { answers.push_back(r); });
    for (int i = 0; i < 64; ++i)
        server.addTool("tool" + std::to_string(i), "More", McpServer::ObjectSchema({}),
                       [](const Json&, McpServer::Respond r) { r(McpServer::ToolResult::Text("x")); });
    server.poll();
    ASSERT_EQ(answers.size(), 1u);
    EXPECT_FALSE(answers[0].isError);
    EXPECT_EQ(executed, 1);

    server.toolTimeoutSeconds = 0.02;
    server.callLocal("count", Json::object{}, [&](McpServer::ToolResult r) { answers.push_back(r); });
    std::this_thread::sleep_for(std::chrono::milliseconds(60));
    server.poll();
    ASSERT_EQ(answers.size(), 2u);
    EXPECT_TRUE(answers[1].isError);
    EXPECT_EQ(executed, 1);
}

// Regression (CodeRabbit): queued calls copied the handler, so a handler with state (a counter) started anew per call
TEST(McpServer, StatefulHandlerKeepsItsState)
{
    McpServer server("test");
    server.addTool("next", "Counts up", McpServer::ObjectSchema({}), [n = 0](const Json&, McpServer::Respond r) mutable
                   { r(McpServer::ToolResult::Text(std::to_string(++n))); });
    std::vector<std::string> answers;
    auto respond = [&](McpServer::ToolResult r) { answers.push_back(r.content[0]["text"].string_value()); };
    server.callLocal("next", Json::object{}, respond);
    server.poll();
    server.callLocal("next", Json::object{}, respond);
    server.poll();
    EXPECT_EQ(answers, (std::vector<std::string>{"1", "2"}));
}

// Regression (CodeRabbit): a call taken from the queue by poll() could still run after stop() drained the queue; no
// handler runs after stop(), and later local calls are refused
TEST(McpServer, NoHandlerRunsAfterStop)
{
    McpServer server("test");
    int executed = 0;
    server.addTool("count", "Counts", McpServer::ObjectSchema({}),
                   [&](const Json&, McpServer::Respond r)
                   {
                       executed++;
                       r(McpServer::ToolResult::Text("ok"));
                   });
    server.stop();
    std::vector<McpServer::ToolResult> answers;
    server.callLocal("count", Json::object{}, [&](McpServer::ToolResult r) { answers.push_back(r); });
    server.poll();
    ASSERT_EQ(answers.size(), 1u);
    EXPECT_TRUE(answers[0].isError);
    EXPECT_EQ(executed, 0);
}

// Regression (CodeRabbit): stop() on another thread could drain the queues while poll() had taken a call but not
// started it yet; the handler then ran after stop() returned. Now a handler that started finishes before stop()
// returns, its answer is delivered, and no handler starts afterwards.
TEST(McpServer, StopWaitsForADispatchingPoll)
{
    McpServer server("test");
    std::atomic_bool started{false}, finished{false}, stop{false};
    std::atomic_int executed{0}, answered{0};
    server.addTool("slow", "Sleeps", McpServer::ObjectSchema({}),
                   [&](const Json&, McpServer::Respond r)
                   {
                       executed++;
                       started = true;
                       std::this_thread::sleep_for(std::chrono::milliseconds(100));
                       finished = true;
                       r(McpServer::ToolResult::Text("ok"));
                   });
    server.callLocal("slow", Json::object{}, [&](McpServer::ToolResult) { answered++; });
    std::thread poller(
        [&]()
        {
            while (!stop)
            {
                server.poll();
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });
    while (!started) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    server.callLocal("slow", Json::object{}, [&](McpServer::ToolResult) { answered++; });
    server.stop();
    EXPECT_TRUE(finished);
    EXPECT_EQ(answered, 2);  // the first answered by its handler, the second by stop()
    stop = true;
    poller.join();
    EXPECT_EQ(executed, 1);
}
