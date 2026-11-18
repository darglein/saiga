/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/json11.hpp"
#include "saiga/export.h"

#include <chrono>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace Saiga
{
namespace Engine
{
/**
 * Timings and counters per sample (one sample per world step; from S13 also per frame), for the world API, MCP and
 * later an ImGui window (docs/engine_design.md §17.1, S10). Scopes form a tree by their path ("step",
 * "step/phase:gameplay", "step/phase:gameplay/system:move"); a scope may be timed several times in one sample (the
 * times add up). Each scope and counter keeps a ring of the last historySize samples. Wall-clock time (steady clock)
 * only for these numbers, never for simulation (R-DET-1). Main thread only.
 */
class SAIGA_ENGINE_API Profiler
{
   public:
    struct Stats
    {
        int samples = 0;
        double avg = 0, min = 0, p50 = 0, p95 = 0, max = 0;
    };

    explicit Profiler(int historySize = 600);

    // The ID of a scope or counter (created on first use; the parent of a scope is its path without the last part)
    int scope(const std::string& path);
    int counter(const std::string& name);
    // Adds a duration to the scope in the current sample
    void add(int scope, double ms);
    void setCounter(int counter, double value);
    void addCounter(int counter, double value);
    // Ends the sample: the accumulated values enter the rings (scopes not timed in this sample count as 0)
    void endSample();
    // Drops the values of the current sample (an aborted step)
    void discardSample();
    // Drops all history (the scopes and counters stay)
    void reset();

    bool enabled = true;
    // Samples since the start (not reset by reset())
    uint64_t sampleCount() const { return samples; }
    int historySize() const { return capacity; }

    // Statistics over the last n samples (n <= 0 or larger than the history: all of it)
    Stats scopeStats(int scope, int lastSamples = 0) const;
    Stats counterStats(int counter, int lastSamples = 0) const;
    // The tree of scopes with their statistics and the share of their parent, and the counters, as JSON
    json11::Json report(int lastSamples = 0) const;


    // Times a scope (no clock calls when the profiler is disabled)
    class Timer
    {
       public:
        Timer(Profiler& p, int scope) : profiler(p), id(scope), active(p.enabled)
        {
            if (active) start = std::chrono::steady_clock::now();
        }
        ~Timer()
        {
            if (active)
                profiler.add(
                    id, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
        }
        Timer(const Timer&)            = delete;
        Timer& operator=(const Timer&) = delete;

       private:
        Profiler& profiler;
        int id;
        bool active;  // enabled at construction (enabling it mid-scope must not add a bogus duration)
        std::chrono::steady_clock::time_point start;
    };

   private:
    struct Series
    {
        std::string name;
        int parent     = -1;
        double current = 0;
        std::vector<double> ring;
    };
    Stats stats(const Series& s, int lastSamples) const;
    void push(Series& s);

    int capacity;
    uint64_t samples = 0;
    int filled       = 0;  // valid samples in the rings
    int head         = 0;  // next ring position
    std::vector<Series> scopes, counters;
};

}  // namespace Engine
}  // namespace Saiga
