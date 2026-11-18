/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Helpers for MCP tool tests (docs/engine_design.md §16.2): JSON-RPC requests for McpServer::handleMessage and a
// background poller (handleMessage blocks until a tool answers; the tools run in poll()).

#pragma once

#include "saiga/core/util/mcp/McpServer.h"

#include <atomic>
#include <chrono>
#include <functional>
#include <string>
#include <thread>

namespace Saiga
{
inline McpServer::Json McpRequest(const std::string& method, const McpServer::Json& params, int id = 1)
{
    return McpServer::Json::object{{"jsonrpc", "2.0"}, {"id", id}, {"method", method}, {"params", params}};
}

// Calls poll() in the background, like the main loop of an application; tick (optional) runs after every poll on the
// same thread (e.g. a world step)
struct Poller
{
    McpServer& server;
    std::atomic_bool stop{false};
    std::thread thread;
    explicit Poller(McpServer& s, std::function<void()> tick = {}) : server(s)
    {
        thread = std::thread(
            [this, tick]()
            {
                while (!stop)
                {
                    server.poll();
                    if (tick) tick();
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

}  // namespace Saiga
