/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/app/FrameStatistics.h"

#include <algorithm>
#include <cmath>

namespace Saiga
{
namespace Engine
{
FrameStatistics::FrameStatistics(double keepSeconds, size_t maxFrames) : keep(keepSeconds), maxFrames(maxFrames) {}

void FrameStatistics::addFrame(double intervalMs, double workMs, const std::array<double, 3>& waitParts)
{
    time += std::max(intervalMs, 0.0) / 1000;
    frameLog.push_back(
        {time, std::max(intervalMs, 0.0), std::clamp(workMs, 0.0, std::max(intervalMs, 0.0)), waitParts});
    trim();
}

void FrameStatistics::addGpu(double gpuMs, double atTime)
{
    // In time order (the frames complete in order)
    if (!gpuLog.empty()) atTime = std::max(atTime, gpuLog.back().time);
    gpuLog.push_back({atTime, std::max(gpuMs, 0.0)});
    trim();
}

void FrameStatistics::clear()
{
    frameLog.clear();
    gpuLog.clear();
}

void FrameStatistics::trim()
{
    while (!frameLog.empty() && (frameLog.size() > maxFrames || frameLog.front().time < time - keep))
        frameLog.pop_front();
    while (!gpuLog.empty() && (gpuLog.size() > maxFrames || gpuLog.front().time < time - keep)) gpuLog.pop_front();
}

FrameStatistics::Series FrameStatistics::stats(std::vector<double> v)
{
    Series s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    auto rank      = [&](double q) { return v[size_t(std::min(double(n - 1), std::ceil(q * double(n)) - 1))]; };
    double sum     = 0;
    for (double x : v) sum += x;
    s.samples    = int(n);
    s.mean       = sum / double(n);
    s.p10        = rank(0.1);
    s.median     = rank(0.5);
    s.p99        = rank(0.99);
    s.max        = v.back();
    s.worstCount = int((n + 99) / 100);
    double worst = 0;
    for (size_t i = n - size_t(s.worstCount); i < n; ++i) worst += v[i];
    s.worst1Percent = worst / s.worstCount;
    s.lowValid      = n >= 100;
    return s;
}

std::vector<int> FrameStatistics::shownRefreshes(const std::vector<double>& intervals, double periodMs)
{
    std::vector<int> out(intervals.size(), 1);
    if (periodMs <= 0) return out;
    // The frame ends on the refresh grid: round(T / period) of the time since the first frame's start
    double t       = 0;
    long long last = 0;
    for (size_t i = 0; i < intervals.size(); ++i)
    {
        t += intervals[i];
        const long long k = std::llround(t / periodMs);
        int shown         = int(k - last);
        if (shown < 1)
        {
            // A catch-up frame (it came before the next refresh): the swap chain queue absorbed an earlier late frame
            // of the last kQueueFrames, which then missed no refresh; else the frame is shown for one refresh
            for (size_t j = i; j-- > 0 && i - j <= kQueueFrames;)
            {
                if (out[j] > 1)
                {
                    out[j]--;
                    break;
                }
            }
            shown = 1;
        }
        out[i] = shown;
        last   = std::max(k, last + 1);
    }
    return out;
}

double FrameStatistics::refreshPeriod(const Summary& summary, double reportedHz)
{
    const double measured = summary.onTimeMedian;
    if (measured <= 0) return 0;
    if (reportedHz <= 0) return measured;
    const double reported = 1000.0 / reportedHz;
    // The measured period agrees with the reported one, or is a whole multiple of it (every frame misses refreshes)
    const double whole = std::round(measured / reported);
    // An absolute tolerance: a relative one grows with the multiple (12.1 ms is not 3 x 4.17 ms)
    return whole >= 1 && std::abs(measured - whole * reported) < 0.05 * reported ? reported : measured;
}

double FrameStatistics::sessionRefreshPeriod(double windowPeriodMs, double reportedHz, double& sessionPeriodMs)
{
    if (windowPeriodMs <= 0) return sessionPeriodMs;
    // A period that matches the reported rate is the display's (e.g. after a switch to a 72 Hz monitor)
    const bool reported = reportedHz > 0 && std::abs(windowPeriodMs - 1000.0 / reportedHz) < 0.05 * 1000.0 / reportedHz;
    if (!reported && sessionPeriodMs > 0)
    {
        // A whole multiple of the session's period: every frame of the window misses refreshes
        const double whole = std::round(windowPeriodMs / sessionPeriodMs);
        if (whole >= 2 && std::abs(windowPeriodMs - whole * sessionPeriodMs) < 0.05 * sessionPeriodMs)
            return sessionPeriodMs;
    }
    if (reported || sessionPeriodMs <= 0 || windowPeriodMs < sessionPeriodMs) sessionPeriodMs = windowPeriodMs;
    return windowPeriodMs;
}

FrameStatistics::Summary FrameStatistics::summarize(double windowSeconds, double refreshPeriodMs) const
{
    Summary out;
    const double from = time - windowSeconds;
    std::vector<double> interval, work, idle, gpu;
    for (auto it = frameLog.rbegin(); it != frameLog.rend(); ++it)
    {
        // A frame belongs to the window if it ended inside it
        if (it->time <= from) break;
        interval.push_back(it->interval);
        work.push_back(it->work);
        idle.push_back(it->interval - it->work);
        out.seconds += it->interval / 1000;
        for (size_t k = 0; k < 3; ++k) out.waitParts[k] += it->waitParts[k];
    }
    if (!interval.empty())
        for (double& w : out.waitParts) w /= double(interval.size());
    for (auto it = gpuLog.rbegin(); it != gpuLog.rend() && it->time > from; ++it) gpu.push_back(it->ms);
    out.interval        = stats(interval);
    out.work            = stats(work);
    out.idle            = stats(idle);
    out.gpu             = stats(gpu);
    const double median = out.interval.median;
    const double period = refreshPeriodMs > 0 ? refreshPeriodMs : median;
    if (median > 0)
    {
        for (double ms : interval)
        {
            out.longFrames += ms > std::max(1.5 * median, median + 1.0) ? 1 : 0;
        }
    }
    // interval is newest first: the changes between neighbours
    std::vector<double> change;
    for (size_t i = 1; i < interval.size(); ++i) change.push_back(std::abs(interval[i] - interval[i - 1]));
    out.pacingP99 = stats(change).p99;
    // Alternation: most of the changes (above 0.1 ms) flip their sign from one frame to the next
    int flips = 0, changes = 0;
    for (size_t i = 2; i < interval.size(); ++i)
    {
        const double a = interval[i - 1] - interval[i - 2], b = interval[i] - interval[i - 1];
        if (std::abs(a) < 0.1 || std::abs(b) < 0.1) continue;
        changes++;
        flips += (a > 0) != (b > 0) ? 1 : 0;
    }
    out.alternating = changes >= 20 && flips > 0.7 * changes && out.pacingP99 > 0.25 * median;
    double below = 0, above = 0;
    int nBelow = 0, nAbove = 0;
    for (double ms : interval)
    {
        if (ms <= out.interval.mean)
            below += ms, nBelow++;
        else
            above += ms, nAbove++;
    }
    out.shortMs = nBelow > 0 ? below / nBelow : 0;
    out.longMs  = nAbove > 0 ? above / nAbove : 0;
    // The shown frames and the missed refreshes (with the refresh period; without vsync against the median)
    const std::vector<int> refreshes = shownRefreshes(std::vector<double>(interval.rbegin(), interval.rend()), period);
    std::vector<double> shownFrames;
    for (int r : refreshes)
    {
        out.missedRefreshes += r - 1;
        shownFrames.push_back(r * period);
    }
    if (refreshPeriodMs > 0) out.shown = stats(shownFrames);
    std::vector<double> onTime;
    for (double ms : interval)
        if (ms <= 1.5 * out.interval.p10) onTime.push_back(ms);
    out.onTimeMedian = stats(onTime).median;
    return out;
}

std::vector<float> FrameStatistics::recentIntervals(size_t n) const
{
    std::vector<float> out;
    const size_t take = std::min(n, frameLog.size());
    for (size_t i = frameLog.size() - take; i < frameLog.size(); ++i) out.push_back(float(frameLog[i].interval));
    return out;
}

}  // namespace Engine
}  // namespace Saiga
