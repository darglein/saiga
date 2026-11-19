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

   private:
    struct Tool
    {
        std::string name, description;
        Json inputSchema;
        Handler handler;
    };

    struct PendingCall;

    std::string name, version, instructions;
    std::vector<Tool> tools;
    std::vector<std::function<bool()>> pollListeners;
    std::chrono::steady_clock::time_point currentDeadline;

    std::mutex pendingMutex;
    std::deque<std::shared_ptr<PendingCall>> pendingCalls;

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
    // HTTP request -> HTTP response (status line, headers and body)
    std::string handleHttp(const std::string& method, const std::map<std::string, std::string>& headers,
                           const std::string& body, bool& keepAlive);
};

}  // namespace Saiga
