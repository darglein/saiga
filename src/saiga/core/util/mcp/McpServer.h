/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/util/json11.hpp"

#include <atomic>
#include <chrono>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace Saiga
{
/**
 * A minimal Model Context Protocol (MCP) server, so that an AI agent can control a running application through tools
 * (e.g. take a screenshot, move the camera, change settings).
 *
 * Transport: "streamable HTTP" with plain JSON responses (no SSE streams, no sessions). Only the tools capability is
 * implemented. The server listens on 127.0.0.1 by default. Against requests from web pages (DNS rebinding, cross-site
 * POSTs) it rejects a non-local or "null" Origin, a Host header that is not the loopback interface and a POST body that
 * is not Content-Type application/json (so browsers must send a CORS preflight, which is refused). Add it to Claude
 * Code with
 *
 *   claude mcp add --transport http saiga http://127.0.0.1:8765/mcp
 *
 * Threading: the HTTP server runs in background threads. The tool handlers are executed by poll(), on the thread that
 * calls it (e.g. once per update of the main loop), so they can access the application state without locking. A
 * handler answers by calling the Respond callback, either directly or later (e.g. a screenshot after the next frame
 * was rendered); it must be called exactly once.
 *
 *   McpServer mcp("my-app");
 *   mcp.addTool("get_state", "Returns the state", McpServer::ObjectSchema({}),
 *               [&](const McpServer::Json& args, McpServer::Respond respond) { respond(ToolResult::Text("...")); });
 *   mcp.start(8765);
 *   // main loop:
 *   mcp.poll();
 */
class SAIGA_CORE_API McpServer
{
   public:
    using Json = json11::Json;

    struct SAIGA_CORE_API ToolResult
    {
        // MCP content blocks ({"type": "text", "text": ...} or {"type": "image", "data": base64, "mimeType": ...})
        std::vector<Json> content;
        bool isError = false;

        static ToolResult Text(const std::string& text);
        static ToolResult Error(const std::string& message);
        // The JSON as (pretty) text
        static ToolResult FromJson(const Json& json);
        // Encoded image data (e.g. png or jpg file content)
        static ToolResult Image(const std::vector<unsigned char>& data, const std::string& mimeType);

        ToolResult& addText(const std::string& text);
        ToolResult& addImage(const std::vector<unsigned char>& data, const std::string& mimeType);
    };

    using Respond = std::function<void(ToolResult)>;
    using Handler = std::function<void(const Json& arguments, Respond respond)>;

    /**
     * name, version: reported to the client (serverInfo).
     * instructions: optional text that tells the agent how to use the tools.
     */
    McpServer(std::string name, std::string version = "1.0", std::string instructions = "");
    ~McpServer();

    McpServer(const McpServer&)            = delete;
    McpServer& operator=(const McpServer&) = delete;

    // inputSchema: a JSON schema of the arguments object, see ObjectSchema().
    void addTool(const std::string& name, const std::string& description, const Json& inputSchema, Handler handler);

    /**
     * Starts listening (port 0: an unused port, see port()). Returns false if the socket could not be bound (e.g. the
     * port is in use). Only 127.0.0.1 is supported (asserted): the server has no authentication and accepts only
     * loopback Host headers.
     */
    bool start(int port, const std::string& address = "127.0.0.1");

    // Stops the server. Pending tool calls fail. Called by the destructor.
    void stop();

    bool running() const { return listening; }
    int port() const { return boundPort; }

    // While a handler runs (in poll()): when its caller stops waiting (queued time + toolTimeoutSeconds). A handler
    // that works long (or answers later) finishes before it, e.g. with a partial result.
    std::chrono::steady_clock::time_point callDeadline() const { return currentDeadline; }

    // Executes the pending tool calls on the calling thread, then the poll listeners.
    void poll();
    // Runs fn at the end of every poll() (on the polling thread), e.g. a deferred tool that checks a deadline while
    // nothing else happens; fn returns false to be removed
    void addPollListener(std::function<bool()> fn) { pollListeners.push_back(std::move(fn)); }

    // A tool call that is not answered within this time (e.g. because poll() is not called) fails.
    double toolTimeoutSeconds = 60;
    // The share of toolTimeoutSeconds that a tool which stops at a deadline leaves for its answer: the engine's
    // answerDeadline is callDeadline() minus answerMarginFraction * toolTimeoutSeconds (0.1: "the timeout minus a
    // tenth"). Tests with short timeouts raise it, so that the deadline comes early while the margin stays large in
    // absolute time (e.g. a 0.8 s timeout with 0.5: the tool answers after 0.4 s, 0.4 s before the timeout).
    double answerMarginFraction = 0.1;

    /**
     * Helper for the input schemas: an object with the given properties, e.g.
     *   ObjectSchema({{"x", NumberProperty("x coordinate")}}, {"x"})
     */
    static Json ObjectSchema(const std::map<std::string, Json>& properties,
                             const std::vector<std::string>& required = {});
    static Json NumberProperty(const std::string& description);
    static Json IntegerProperty(const std::string& description);
    static Json BooleanProperty(const std::string& description);
    static Json StringProperty(const std::string& description, const std::vector<std::string>& enumValues = {});
    static Json Vec3Property(const std::string& description);

    // JSON text with integers in full and other numbers shortened to 6 significant digits, indented if pretty.
    static std::string FormatJson(const Json& json, bool pretty);

    // Base64 encoding (RFC 4648, with padding), as used for the image content.
    static std::string Base64Encode(const unsigned char* data, size_t size);

    // Handles one JSON-RPC message (request or notification) and returns the response, or a null Json for a
    // notification. Blocks until the tool call is answered. Used by the HTTP connections (and the tests).
    Json handleMessage(const Json& message);

    /**
     * A tool call from inside the application (e.g. a console): queued like an HTTP tools/call and run by the next
     * poll(), in arrival order with the remote calls. Does not block. respond gets the answer on the polling thread
     * (also a deferred answer from another thread, delivered by a later poll()), an error for an unknown tool or a full
     * queue, and a timeout error once toolTimeoutSeconds have passed (checked by poll()). stop() and the destructor
     * answer the pending local calls with an error on their calling thread.
     */
    void callLocal(const std::string& toolName, const Json& arguments, Respond respond);

    // One tool call event for the call observer
    struct CallRecord
    {
        enum class Phase
        {
            Started,  // poll() started the handler
            Answered  // answered (also errors, timeouts and shutdown)
        };
        enum class Source
        {
            Remote,  // HTTP (an agent)
            Local    // callLocal
        };
        uint64_t id = 0;  // assigned by the server (1, 2, ...)
        std::string tool;
        Json arguments;
        Source source = Source::Remote;
        Phase phase   = Phase::Started;
        bool isError  = false;
        // Answered: from the start of the handler (or from queueing, if it never ran) to the answer
        double durationMs = 0;
        // Answered: the first text part of the answer, whitespace collapsed, at most 200 characters
        std::string answer;
    };
    /**
     * Called on the polling thread at the end of each poll() with the records since the last poll (in order). Answers
     * that happen on other threads (timeouts on the HTTP threads, deferred answers) are queued and delivered by the
     * next poll(); records not delivered when the server stops are dropped. One observer; empty: none.
     */
    void setCallObserver(std::function<void(const CallRecord&)> observer);

    // The registered tools (registration order) and one tool's description and input schema (false if unknown)
    std::vector<std::string> toolNames() const;
    bool toolInfo(const std::string& toolName, std::string& description, Json& inputSchema) const;

   private:
    struct Tool
    {
        std::string name, description;
        Json inputSchema;
        // Shared with the queued calls: stable when more tools are registered, and a stateful handler keeps its state
        std::shared_ptr<Handler> handler;
    };

    struct PendingCall;
    // The records and local answers made on any thread, delivered by poll(); shared with the answer callbacks, which
    // may outlive the server
    struct CallState;

    std::string name, version, instructions;
    std::vector<Tool> tools;
    std::vector<std::function<bool()>> pollListeners;
    std::chrono::steady_clock::time_point currentDeadline;

    std::mutex pendingMutex;
    std::deque<std::shared_ptr<PendingCall>> pendingCalls;
    std::shared_ptr<CallState> callState;
    // Local calls that were started and not answered yet (polling thread): their deadline is checked by poll()
    std::vector<std::shared_ptr<PendingCall>> localInFlight;
    std::mutex inFlightMutex;  // localInFlight: poll() and stop() may run on different threads
    // poll() dispatches the calls under it and stop() drains under it: a handler that started finishes before stop()
    // returns, and none starts after (recursive: a handler may call stop())
    std::recursive_mutex dispatchMutex;
    std::function<void(const CallRecord&)> callObserver;
    std::atomic<uint64_t> nextCallId{1};

    std::atomic_bool stopping  = false;
    std::atomic_bool listening = false;
    int boundPort              = 0;
    // The listening socket (SOCKET on Windows)
    intptr_t listenSocket = -1;
    std::thread acceptThread;

    struct Connection
    {
        std::thread thread;
        intptr_t socket = -1;
        std::atomic_bool done{false};
    };
    std::mutex connectionMutex;
    std::vector<std::unique_ptr<Connection>> connections;

    // Further connections are answered with 503 and closed
    static constexpr size_t maxConnections = 32;
    // Tool calls waiting for poll()
    static constexpr size_t maxPendingCalls = 64;

    void acceptLoop();
    // Serves the connection, then closes its socket and marks it done
    void connectionLoop(Connection* connection);
    void serveConnection(Connection* connection);
    void joinFinishedConnections(bool all);

    ToolResult callTool(const std::string& toolName, const Json& arguments);
    std::shared_ptr<PendingCall> makeCall(Tool& tool, const Json& arguments, bool local);
    // Delivers the queued local answers and records (polling thread, or the thread of stop())
    void deliverCallState(bool records);
    void answerLocalCallsOnStop();
    // HTTP request -> HTTP response (status line, headers and body)
    std::string handleHttp(const std::string& method, const std::map<std::string, std::string>& headers,
                           const std::string& body, bool& keepAlive);
};

}  // namespace Saiga
