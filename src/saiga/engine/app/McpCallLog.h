/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/mcp/McpServer.h"
#include "saiga/export.h"

#include <functional>
#include <map>
#include <string>

namespace Saiga
{
namespace Engine
{
/**
 * The MCP call log (docs/engine_design.md §17.1; S19c): the call observer of the app's McpServer writes one line per
 * answered tool call into the engine log, category "mcp" (shown by the debug UI's console; the logs tool returns them
 * only when asked for category "mcp"):
 *
 *   spawn -> ok 1.2 ms: {"entity": ...}  args {"class": "car"}
 *   (console) screenshot -> error 0.1 ms: No device
 *
 * A call still running after 0.5 s gets a "<tool> ... running" line. The read-only tools agents poll in loops (logs,
 * app_status, world_status, get_profile) are logged at most once per 2 s per tool: the calls in between are counted
 * and named on the next line ("(+N calls)"), or on a line of their own once 2 s have passed (tick()). Local calls (the
 * console) are never throttled. Main thread (the observer runs in McpServer::poll).
 */
class SAIGA_ENGINE_API McpCallLog
{
   public:
    using Clock = std::function<double()>;  // seconds
    using Sink  = std::function<void(const std::string& line)>;

    // Default clock: steady clock; default sink: SAIGA_LOG("mcp", ...)
    explicit McpCallLog(Clock clock = Clock(), Sink sink = Sink());

    // The observer (McpServer::setCallObserver)
    void record(const McpServer::CallRecord& r);
    // Running lines and the flush of counted polling calls (called every frame, also from a poll listener)
    void tick();

    // The longest running call (not screenshot: it would show in its own image), or empty; and since when (s)
    std::string longestRunning(double* seconds = nullptr) const;

    static constexpr double kThrottleSeconds = 2.0;
    static constexpr double kRunningSeconds  = 0.5;
    static bool isPollingTool(const std::string& tool);

   private:
    struct Running
    {
        std::string tool;
        double start = 0;
        bool logged  = false;
    };
    struct Throttle
    {
        double lastLine = -1e30;
        int counted     = 0;
    };
    Clock now;
    Sink sink;
    std::map<uint64_t, Running> running;
    std::map<std::string, Throttle> throttles;
};

}  // namespace Engine
}  // namespace Saiga
