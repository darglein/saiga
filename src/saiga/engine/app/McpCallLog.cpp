/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/app/McpCallLog.h"

#include "saiga/engine/core/Log.h"

#include <algorithm>
#include <chrono>
#include <cstdio>

namespace Saiga
{
namespace Engine
{
namespace
{
std::string cut(const std::string& s, size_t n)
{
    return s.size() <= n ? s : s.substr(0, n) + "...";
}
}  // namespace

McpCallLog::McpCallLog(Clock clock, Sink sinkFn) : now(std::move(clock)), sink(std::move(sinkFn))
{
    if (!now)
    {
        const auto start = std::chrono::steady_clock::now();
        now = [start]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(); };
    }
    if (!sink) sink = [](const std::string& line) { SAIGA_LOG("mcp", "%s", line.c_str()); };
}

bool McpCallLog::isPollingTool(const std::string& tool)
{
    return tool == "logs" || tool == "app_status" || tool == "world_status" || tool == "get_profile";
}

void McpCallLog::record(const McpServer::CallRecord& r)
{
    using Phase    = McpServer::CallRecord::Phase;
    const double t = now();
    if (r.phase == Phase::Started)
    {
        // A call answered before its start was recorded (a timeout racing the start) is not running
        if (std::find(recentlyAnswered.begin(), recentlyAnswered.end(), r.id) == recentlyAnswered.end())
            running[r.id] = Running{r.tool, t, false};
        return;
    }
    running.erase(r.id);
    recentlyAnswered.push_back(r.id);
    if (recentlyAnswered.size() > 64) recentlyAnswered.pop_front();
    const bool local = r.source == McpServer::CallRecord::Source::Local;
    int counted      = 0;
    if (!local && isPollingTool(r.tool))
    {
        Throttle& th = throttles[r.tool];
        if (t - th.lastLine < kThrottleSeconds)
        {
            th.counted++;
            return;
        }
        counted     = th.counted;
        th.counted  = 0;
        th.lastLine = t;
    }
    // The result first (what matters when reading the log), the arguments last
    const bool noArgs = r.arguments.is_null() || (r.arguments.is_object() && r.arguments.object_items().empty());
    char ms[48];
    std::snprintf(ms, sizeof(ms), " %.1f ms", r.durationMs);
    std::string line = std::string(local ? "(console) " : "") + r.tool + " -> " + (r.isError ? "error" : "ok") + ms;
    if (!r.answer.empty()) line += ": " + cut(r.answer, 120);
    if (counted > 0) line += " (+" + std::to_string(counted) + " calls)";
    if (!noArgs) line += "  args " + cut(r.arguments.dump(), 200);
    sink(line);
}

void McpCallLog::tick()
{
    const double t = now();
    for (auto& [id, run] : running)
    {
        if (run.logged || t - run.start < kRunningSeconds) continue;
        run.logged = true;
        sink(run.tool + " ... running");
    }
    for (auto& [tool, th] : throttles)
    {
        if (th.counted == 0 || t - th.lastLine < kThrottleSeconds) continue;
        sink(tool + ": +" + std::to_string(th.counted) + " calls");
        th.counted  = 0;
        th.lastLine = t;
    }
}

std::string McpCallLog::longestRunning(double* seconds) const
{
    const double t = now();
    std::string tool;
    double longest = -1;
    for (const auto& [id, run] : running)
    {
        if (run.tool == "screenshot" || t - run.start <= longest) continue;
        longest = t - run.start;
        tool    = run.tool;
    }
    if (seconds) *seconds = std::max(longest, 0.0);
    return tool;
}

}  // namespace Engine
}  // namespace Saiga
