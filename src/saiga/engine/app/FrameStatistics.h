/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/export.h"

#include <array>
#include <cstddef>
#include <deque>
#include <string>
#include <vector>

namespace Saiga
{
namespace Engine
{
/**
 * The frame times of the last seconds for the "Engine statistics" window (docs/engine_design.md §17.1): per frame the
 * interval (from this frame's start to the next one's: what the player sees), the CPU work (the frame without the
 * render's waits for the frame slot, the swap chain image and the present) and per GPU frame its GPU time. The time
 * base is the sum of the intervals (deterministic with fixed intervals, e.g. in tests). summarize() gives statistics
 * over a time window, so the shown numbers do not change every frame and single long frames (stutter) show in p99, max,
 * the 1% low and the long frames. Main thread only.
 */
class SAIGA_ENGINE_API FrameStatistics
{
   public:
    // Statistics of one series over the window (ms)
    struct Series
    {
        int samples = 0;
        double mean = 0, p10 = 0, median = 0, p99 = 0, max = 0;
        // The mean of the slowest 1% (ceil(samples / 100) values): 1000 / it is the "1% low" frame rate of the
        // interval. Only meaningful from 100 samples on (lowValid)
        double worst1Percent = 0;
        int worstCount       = 0;
        bool lowValid        = false;
    };
    struct Summary
    {
        double seconds = 0;  // the time the window's frames cover
        Series interval, work, idle, gpu;
        // Frames longer than max(1.5 x the median, the median + 1 ms) (a missed refresh with vsync is 2 x; the 1 ms
        // keeps sub-millisecond jitter at high frame rates out)
        int longFrames = 0;
        // The refreshes missed (shownRefreshes - 1 per frame) against the refresh period (vsync; else the median)
        int missedRefreshes = 0;
        // Frame pacing: the p99 of the change of the interval from one frame to the next (alternating short and long
        // frames stutter although mean and median look fine)
        double pacingP99 = 0;
        // The median of the frames that made a refresh (interval <= 1.5 x the p10): the refresh period with vsync
        double onTimeMedian = 0;
        // With a refresh period (vsync): the frames as the display shows them, each interval rounded to whole
        // refreshes (at least one). The CPU-side intervals jitter around the refresh while the display shows every
        // frame for whole refreshes: fps, 1% low and max of the shown frames are what the player sees
        Series shown;
        // The intervals alternate between short and long (most changes between neighbouring frames flip their sign):
        // uneven frame pacing, not single long frames. The means of the intervals below and above the mean
        bool alternating = false;
        double shortMs = 0, longMs = 0;
        // The CPU wait's parts (means): the frame slot (the GPU), the swap chain image acquire, the present
        std::array<double, 3> waitParts = {};
    };

    explicit FrameStatistics(double keepSeconds = 30, size_t maxFrames = 30000);

    // A frame: its interval, its CPU work (the work is clamped to the interval) and the parts of its wait (ms)
    void addFrame(double intervalMs, double workMs, const std::array<double, 3>& waitParts = {});
    // A completed GPU frame (ms) of the frame that ended at the given time (now() when it was rendered; the GPU times
    // arrive a few frames later)
    void addGpu(double gpuMs, double atTime);
    void clear();

    // Over the frames of the last windowSeconds; refreshPeriodMs > 0: the display's refresh period for the missed
    // refreshes
    Summary summarize(double windowSeconds, double refreshPeriodMs = 0) const;
    // The newest n intervals (oldest first)
    std::vector<float> recentIntervals(size_t n) const;

    // The samples (the graphs): a frame ends at time (seconds, the sum of the intervals)
    struct Frame
    {
        double time, interval, work;
        std::array<double, 3> waitParts;
    };
    struct Gpu
    {
        double time, ms;
    };
    const std::deque<Frame>& frameSamples() const { return frameLog; }
    const std::deque<Gpu>& gpuSamples() const { return gpuLog; }

    double now() const { return time; }
    // The start time of the oldest kept frame (now() without frames)
    double firstTime() const
    {
        return frameLog.empty() ? time : frameLog.front().time - frameLog.front().interval / 1000;
    }
    size_t frames() const { return frameLog.size(); }

    /**
     * The refresh period (ms) of frames presented with vsync: the median of the frames that made a refresh (a window
     * manager may report another monitor's rate), or the reported period if that median is a whole multiple of it
     * within 5% of the reported period (every frame misses refreshes, e.g. a slow GPU at 83 Hz on a 165 Hz display).
     * 0 without frames.
     */
    static double refreshPeriod(const Summary& summary, double reportedHz);

    /**
     * The refresh period of the session: the window's period (refreshPeriod), unless it is a whole multiple of the
     * session's shorter period and does not match the reported rate (every frame of the window missed refreshes);
     * updates sessionPeriodMs (the shortest period, or the reported one after a display change; 0: none yet).
     */
    static double sessionRefreshPeriod(double windowPeriodMs, double reportedHz, double& sessionPeriodMs);

    /**
     * The refreshes each frame is shown for (vsync, oldest first): its end on the refresh grid, round(T / period),
     * minus the previous frame's. A catch-up frame that ends before the next refresh (the swap chain queue absorbed a
     * late frame) takes the extra refresh back from a late frame of the last kQueueFrames, so a 9 ms frame followed by
     * a 3 ms frame at 6 ms refreshes misses nothing; each frame is shown for at least one refresh.
     */
    static constexpr size_t kQueueFrames = 3;
    static std::vector<int> shownRefreshes(const std::vector<double>& intervals, double periodMs);

    // Statistics of a set of values (nearest-rank percentiles, as Profiler)
    static Series stats(std::vector<double> values);

   private:
    void trim();

    double keep;
    size_t maxFrames;
    double time = 0;  // seconds: the sum of the intervals
    std::deque<Frame> frameLog;
    std::deque<Gpu> gpuLog;
};

}  // namespace Engine
}  // namespace Saiga
