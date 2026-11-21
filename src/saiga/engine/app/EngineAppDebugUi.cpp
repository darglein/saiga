/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// EngineApp's debug UI (docs/engine_design.md §17.1; S19c): the frame statistics, the Performance and Engine tabs, the
// bar's summary and the MCP call log

#include "saiga/core/imgui/imgui.h"
#include "saiga/engine/app/DebugUiStyle.h"
#include "saiga/engine/app/EngineApp.h"
#include "saiga/engine/terrain/TerrainSystem.h"
#include "saiga/vulkan/scene/RenderScene.h"
#include "saiga/vulkan/window/Window.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>

namespace Saiga
{
namespace Engine
{
namespace
{
constexpr ImU32 kCpuColor   = DebugUiStyle::kCpu;
constexpr ImU32 kGpuColor   = DebugUiStyle::kGpu;
constexpr ImU32 kFrameColor = DebugUiStyle::kFrame;
constexpr ImU32 kRefColor   = DebugUiStyle::kRef;
constexpr ImU32 kWarnColor  = DebugUiStyle::kWarn;
constexpr ImU32 kWhite      = DebugUiStyle::kWhite;
using DebugUiStyle::cellRight;
using DebugUiStyle::coloredText;

// A time in ms for the tables: 2 decimals, 3 below 0.1 ms (the GPU sections and systems are often that short)
std::string msText(double ms)
{
    char text[32];
    std::snprintf(text, sizeof(text), ms >= 0.1 || ms == 0 ? "%.2f" : "%.3f", ms);
    return text;
}

struct GraphLine
{
    float value;
    ImU32 color;
    const char* label;
};

// The frame time graph of the statistics window over the frames of the window (the same frames as the numbers; until
// the window is full, the time since the statistics started): per pixel column the longest frame interval (a single
// long frame stays visible) on a scale that shows the refresh; below it a strip of the CPU busy time (area) and the
// GPU busy time (line) scaled to them, so their size shows also when they are tiny against the refresh. Reference
// lines with their labels in a gutter on the right; intervals above the scale are clipped and marked red at the top.
void frameGraph(const FrameStatistics& fs, double span, float frameScale, const std::vector<GraphLine>& lines,
                float busyScale, float budgetMs, float height)
{
    const float gutter = 62;
    const float width  = std::max(ImGui::GetContentRegionAvail().x, 140.0f);
    const float lineH0 = ImGui::GetTextLineHeight();
    // The plots share the height (without the time axis): 62% the frame intervals, 38% the busy times
    const float plots = std::max(height - lineH0 - 4, 100.0f);
    const float gap = 4, frameH = (plots - gap) * 0.55f, busyH = plots - gap - frameH;
    const float lineH   = ImGui::GetTextLineHeight();
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton("##frameGraph", ImVec2(width, frameH + gap + busyH + lineH0));
    const bool hovered = ImGui::IsItemHovered();
    const float plotW  = width - gutter;
    ImDrawList* dl     = ImGui::GetWindowDrawList();
    const ImU32 dim    = ImGui::GetColorU32(ImGuiCol_TextDisabled);

    // Per pixel column: the longest interval and the largest CPU and GPU busy times (-1: no frame in the column)
    const int columns = std::max(int(plotW), 1);
    std::vector<float> frameMax(size_t(columns), -1), workMax(size_t(columns), -1), gpuMax(size_t(columns), -1);
    const double from = fs.now() - span;
    auto column       = [&](double t) { return std::clamp(int((t - from) / span * columns), 0, columns - 1); };
    for (const auto& f : fs.frameSamples())
    {
        if (f.time <= from) continue;
        const size_t c = size_t(column(f.time));
        frameMax[c]    = std::max(frameMax[c], float(f.interval));
        workMax[c]     = std::max(workMax[c], float(f.work));
    }
    for (const auto& g : fs.gpuSamples())
    {
        if (g.time <= from) continue;
        const size_t c = size_t(column(g.time));
        gpuMax[c]      = std::max(gpuMax[c], float(g.ms));
    }
    auto plot = [&](float top, float height, float scale, auto&& draw)
    {
        const ImVec2 p0(origin.x, top), p1(origin.x + plotW, top + height);
        dl->AddRectFilled(p0, p1, ImGui::GetColorU32(ImGuiCol_FrameBg));
        auto y = [=](double v) { return p1.y - float(std::min(v, double(scale)) / scale) * (height - 2) - 1; };
        draw(p0, p1, y);
    };
    auto polyline = [&](const std::vector<float>& v, ImU32 color, auto&& y)
    {
        int last = -1;
        for (int c = 0; c < columns; ++c)
        {
            if (v[size_t(c)] < 0) continue;
            if (last >= 0)
                dl->AddLine(ImVec2(origin.x + float(last), y(v[size_t(last)])),
                            ImVec2(origin.x + float(c), y(v[size_t(c)])), color, 1.5f);
            last = c;
        }
    };
    // The frame intervals
    plot(origin.y, frameH, frameScale,
         [&](ImVec2 p0, ImVec2 p1, auto&& y)
         {
             polyline(frameMax, kFrameColor, y);
             for (int c = 0; c < columns; ++c)
                 if (frameMax[size_t(c)] > frameScale)
                     dl->AddRectFilled(ImVec2(p0.x + float(c) - 1.5f, p0.y), ImVec2(p0.x + float(c) + 1.5f, p0.y + 6),
                                       kWarnColor);
             for (const GraphLine& l : lines)
             {
                 if (l.value <= 0 || l.value > frameScale) continue;
                 const float ly = y(l.value);
                 dl->AddLine(ImVec2(p0.x, ly), ImVec2(p1.x, ly), l.color, 1.5f);
                 // Below the scale and the legend at the top of the gutter
                 if (ly - lineH / 2 > p0.y + 2 * lineH) dl->AddText(ImVec2(p1.x + 4, ly - lineH / 2), l.color, l.label);
             }
         });
    // The busy times on their own scale
    const float busyTop = origin.y + frameH + gap;
    plot(busyTop, busyH, busyScale,
         [&](ImVec2 p0, ImVec2 p1, auto&& y)
         {
             for (int c = 0; c < columns; ++c)
                 if (workMax[size_t(c)] > 0)
                     dl->AddLine(ImVec2(p0.x + float(c) + 0.5f, p1.y),
                                 ImVec2(p0.x + float(c) + 0.5f, y(workMax[size_t(c)])),
                                 (kCpuColor & 0x00FFFFFF) | 0xA0000000);
             polyline(gpuMax, kGpuColor, y);
             // Busy times above the scale: marked at the top as in the frame plot
             for (int c = 0; c < columns; ++c)
                 if (workMax[size_t(c)] > busyScale || gpuMax[size_t(c)] > busyScale)
                     dl->AddRectFilled(ImVec2(p0.x + float(c) - 1.5f, p0.y), ImVec2(p0.x + float(c) + 1.5f, p0.y + 6),
                                       kWarnColor);
             // The scale and the legend at the top of the gutter; the budget line (the frame) when it is on the scale
             char scaleText[32];
             std::snprintf(scaleText, sizeof(scaleText), "%.3g ms", busyScale);
             dl->AddText(ImVec2(p1.x + 4, p0.y), dim, scaleText);
             dl->AddText(ImVec2(p1.x + 4, p0.y + lineH), kCpuColor, "CPU");
             dl->AddText(ImVec2(p1.x + 4, p0.y + 2 * lineH), kGpuColor, "GPU");
             if (budgetMs > 0 && budgetMs <= busyScale)
             {
                 const float ly = y(budgetMs);
                 dl->AddLine(ImVec2(p0.x, ly), ImVec2(p1.x, ly), kRefColor, 1.0f);
                 if (ly > p0.y + 3 * lineH) dl->AddText(ImVec2(p1.x + 4, ly - lineH / 2), kRefColor, "budget");
             }
         });
    // The frame plot's scale and legend at the top of its gutter (the reference lines below them)
    char text[96];
    std::snprintf(text, sizeof(text), "%.3g ms", frameScale);
    dl->AddText(ImVec2(origin.x + plotW + 4, origin.y), dim, text);
    dl->AddText(ImVec2(origin.x + plotW + 4, origin.y + lineH), kFrameColor, "frame");
    // The time axis
    std::snprintf(text, sizeof(text), "-%.0f s", span);
    dl->AddText(ImVec2(origin.x, busyTop + busyH), dim, text);
    dl->AddText(ImVec2(origin.x + plotW - ImGui::CalcTextSize("now").x, busyTop + busyH), dim, "now");
    if (hovered)
    {
        const int c = int(ImGui::GetIO().MousePos.x - origin.x);
        if (c >= 0 && c < columns && frameMax[size_t(c)] >= 0)
        {
            char gpu[32] = "-";
            if (gpuMax[size_t(c)] >= 0) std::snprintf(gpu, sizeof(gpu), "%.2f ms", gpuMax[size_t(c)]);
            ImGui::SetTooltip(
                "%.1f s ago\nframe interval %.2f ms (the longest in this pixel column)\nCPU busy %.2f ms\nGPU busy %s",
                span * (1.0 - (c + 0.5) / columns), frameMax[size_t(c)], workMax[size_t(c)], gpu);
        }
    }
}
}  // namespace

void EngineApp::checkStatisticsReset()
{
    // A profile reset (reset_profile) also starts the frame statistics anew
    if (frameProfiler.resetCount() == statisticsResets) return;
    statisticsResets = frameProfiler.resetCount();
    clearFrameStatistics();
}

void EngineApp::clearFrameStatistics()
{
    frameStats.clear();
    renderedFrameTimes.clear();
    sessionPeriodMs       = 0;
    shownStatistics.valid = false;
}

void EngineApp::refreshShownStatistics()
{
    checkStatisticsReset();
    ShownStatistics& s = shownStatistics;
    s.at               = std::chrono::steady_clock::now();
    s.valid            = true;
    // The display's refresh rate as reported (the window may have moved to another monitor)
    s.refreshHz = appWindow ? appWindow->refreshRateHz() : 0.0;
    // A display with another refresh rate (the window moved): its period does not apply to the old frames
    if (s.refreshHz != lastRefreshHz)
    {
        if (lastRefreshHz > 0 && s.refreshHz > 0) clearFrameStatistics();
        lastRefreshHz = s.refreshHz;
    }
    const double window = kStatisticsWindows[size_t(statisticsWindowIndex)];
    const bool vsync    = renderScene && renderScene->vsync();
    s.summary           = frameStats.summarize(window);
    if (vsync)
    {
        // The refresh period from the frames that made a refresh (FrameStatistics::refreshPeriod). The display rarely
        // changes: the shortest period of the session is kept, and used while the window has no frame that made a
        // refresh (every frame misses one: the window's estimate is a multiple of it)
        double period = 0;
        if (s.refreshHz > 0)
            period = FrameStatistics::sessionRefreshPeriod(FrameStatistics::refreshPeriod(s.summary, s.refreshHz),
                                                           s.refreshHz, sessionPeriodMs);
        else
        {
            // An unknown rate (a window on Wayland): only this window's frames, no period kept from earlier (it may
            // be of another display: regression, a 120 Hz period made every 60 Hz frame a missed refresh)
            sessionPeriodMs = 0;
            period          = FrameStatistics::refreshPeriod(s.summary, 0);
            if (period <= 0) period = s.summary.interval.median;
        }
        s.periodMs = period;
        s.summary  = frameStats.summarize(window, period);
    }
    else
        s.periodMs = s.summary.interval.median;
    // The time the graph covers: the window, or the time since the statistics started until it is full
    s.graphSpan = std::max(std::min(window, frameStats.now() - frameStats.firstTime()), 1.0);
    // The newest steps, as many as the window's frames ran: about the window's time (the steps of a frame that was not
    // presented are among them; none while the world is paused), without the current loop's steps (pendingSteps: their
    // frame is not in the window yet)
    s.stepReport = s.summary.steps > 0 ? gameWorld->profiler().report(s.summary.steps, pendingSteps) : json11::Json();
    // Once per refresh (the tab and the bar read them every frame)
    shownVerdict   = statisticsVerdict();
    s.stepMs       = gameWorld->profiler().scopeStats(gameWorld->profiler().scope("step"), 120).p50;
    s.droppedSteps = frameProfiler.counterStats(counterDropped, 120).max > 0;
}

EngineApp::Verdict EngineApp::statisticsVerdict() const
{
    const ShownStatistics& s            = shownStatistics;
    const FrameStatistics::Summary& sum = s.summary;
    const bool vsync                    = renderScene && renderScene->vsync();
    const double period                 = s.periodMs;
    const double busy                   = std::max(sum.work.median, sum.gpu.median);
    const char* limit                   = sum.gpu.median >= sum.work.median ? "GPU" : "CPU";
    Verdict v;
    v.period = period;
    char text[400];
    // The largest of the CPU's waits (frame slot, image acquire, present): what a frame waits for when neither the
    // CPU nor the GPU fills it
    const char* waitNames[] = {"the frame slot (an earlier GPU frame)", "the swap chain image acquire", "the present"};
    const size_t waitMax = size_t(std::max_element(sum.waitParts.begin(), sum.waitParts.end()) - sum.waitParts.begin());
    if (!renderScene || !renderingEnabled || renderScene->headless())
    {
        std::snprintf(text, sizeof(text), "Not presenting: paced to the step rate (%.0f steps/s)",
                      1.0 / gameWorld->dt());
        v.tag = "NOT PRESENTING";
    }
    else if (sum.interval.samples == 0)
    {
        std::snprintf(text, sizeof(text), "Measuring (the first second is left out)");
        v.tag = "MEASURING";
    }
    else if (vsync && sum.missedRefreshes > 0 && busy >= 0.85 * period)
    {
        std::snprintf(text, sizeof(text), "Limited by the %s: frames miss refreshes (%.0f Hz display)", limit,
                      1000.0 / period);
        v.warn = true;
        v.tag  = std::string(limit) + "-BOUND";
    }
    else if (vsync && sum.seconds > 0 && sum.missedRefreshes / sum.seconds > kStutterWarnPerSecond &&
             busy < 0.85 * period)
    {
        // Frames miss refreshes although neither the CPU nor the GPU fills the frame: hitches (a blocking call, a
        // load, the window system), not a steady load
        std::snprintf(text, sizeof(text),
                      "Stutter: %.1f missed refreshes/s while the CPU and the GPU are mostly idle (%.0f%% of the "
                      "refresh). Look for hitches: blocking calls (e.g. MCP screenshots), loads, the largest wait: %s",
                      sum.missedRefreshes / sum.seconds, 100 * busy / period, waitNames[waitMax]);
        v.tag  = "STUTTER";
        v.warn = true;
    }
    else if (vsync && busy >= 0.85 * period)
    {
        std::snprintf(text, sizeof(text), "Limited by the display, the %s near its limit (%.0f%% of the refresh)",
                      limit, 100 * busy / period);
        v.tag  = std::string("VSYNC, ") + limit + " NEAR LIMIT";
        v.warn = true;
    }
    else if (vsync)
    {
        std::snprintf(text, sizeof(text), "Limited by the display: vsync at %.0f Hz (%.2f ms per refresh)",
                      1000.0 / period, period);
        v.displayLimited = true;
        v.tag            = "VSYNC-BOUND";
    }
    else if (busy >= 0.85 * period)
    {
        std::snprintf(text, sizeof(text), "Limited by the %s (no vsync)", limit);
        v.tag = std::string(limit) + "-BOUND";
    }
    else if (sum.waitParts[waitMax] >= 0.05)
    {
        std::snprintf(text, sizeof(text), "Limited by waits for %s (%.2f ms per frame; no vsync)", waitNames[waitMax],
                      sum.waitParts[waitMax]);
        v.tag = "WAIT-BOUND";
    }
    else
    {
        std::snprintf(text, sizeof(text), "Limit undetermined: neither the CPU, the GPU nor a wait fills the frame");
        v.tag = "UNDETERMINED";
    }
    v.text = text;
    return v;
}


void EngineApp::perfSummary()
{
    // The headline: what limits the frame rate as a short tag (the sentence in its tooltip), fps and stutter, the CPU's
    // and the GPU's busy time against the frame (of the refresh with vsync)
    const FrameStatistics::Summary& sum = shownStatistics.summary;
    const bool vsync                    = renderScene && renderScene->vsync();
    const Verdict& verdict              = shownVerdict;
    const double period                 = verdict.period;
    char text[160];
    coloredText(verdict.warn ? kWarnColor : kWhite, verdict.tag.c_str());
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "%s\n\nVSYNC: the CPU and the GPU finish early and wait for the next refresh.\n"
            "GPU / CPU: its busy time fills the frame (with vsync: frames miss refreshes).",
            verdict.text.c_str());
    if (period > 0)
    {
        ImGui::SameLine();
        if (vsync)
            ImGui::TextDisabled("%.0f Hz, %.2f ms per refresh", 1000.0 / period, period);
        else
            ImGui::TextDisabled("no vsync, median frame %.2f ms", period);
    }

    // fps, 1% low, the longest frame, stutter (with vsync of the shown frames: what the player sees)
    const FrameStatistics::Series& frames = vsync && sum.shown.samples > 0 ? sum.shown : sum.interval;
    if (frames.samples > 0)
        std::snprintf(text, sizeof(text), "%.0f fps", frames.mean > 0 ? 1000.0 / frames.mean : 0.0);
    else
        std::snprintf(text, sizeof(text), "-- fps");
    coloredText(kFrameColor, text);
    ImGui::SameLine(0, ImGui::GetFontSize());
    if (frames.lowValid)
        std::snprintf(text, sizeof(text), "1%% low %.0f", 1000.0 / frames.worst1Percent);
    else
        std::snprintf(text, sizeof(text), "1%% low --");
    ImGui::TextUnformatted(text);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("1000 / the mean of the slowest 1%% of the frames (here %d of %d; from 100 frames on).%s",
                          frames.worstCount, frames.samples,
                          vsync ? "\nWith vsync of the shown frames: each interval rounded to whole refreshes." : "");
    ImGui::SameLine(0, ImGui::GetFontSize());
    std::snprintf(text, sizeof(text), "max %.1f ms", frames.max);
    ImGui::TextUnformatted(text);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("The longest frame in the window");
    ImGui::SameLine(0, ImGui::GetFontSize());
    if (!vsync && sum.alternating)
    {
        std::snprintf(text, sizeof(text), "uneven pacing %.1f / %.1f ms", sum.shortMs, sum.longMs);
        coloredText(kWarnColor, text);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "The frame intervals alternate between short and long (p99 change between neighbouring\n"
                "frames %.2f ms): the motion is uneven although the mean and the median look fine.",
                sum.pacingP99);
    }
    else
    {
        const int stutters = vsync ? sum.missedRefreshes : sum.longFrames;
        const double rate  = sum.seconds > 0 ? stutters / sum.seconds : 0.0;
        std::snprintf(text, sizeof(text), "%s %.1f/s", vsync ? "missed" : "long", rate);
        coloredText(rate > kStutterWarnPerSecond ? kWarnColor : ImGui::GetColorU32(ImGuiCol_TextDisabled), text);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "%d in the window: stutter the player notices. With vsync: the refreshes the frames missed\n"
                "(an interval of 2 refreshes is one). Without vsync: frames longer than 1.5 x the median\n"
                "(and the median + 1 ms). Frame pacing, the p99 change between neighbouring frames: %.2f ms.",
                stutters, sum.pacingP99);
    }

    // CPU and GPU busy: the name, a bar of the share of the frame, the value beside it (never over the fill)
    if (ImGui::BeginTable("##busy", 3, ImGuiTableFlags_SizingFixedFit))
    {
        const float em = ImGui::GetFontSize();
        ImGui::TableSetupColumn("name", ImGuiTableColumnFlags_WidthFixed, 2.6f * em);
        ImGui::TableSetupColumn("bar", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthFixed, 7.5f * em);
        auto row = [&](const char* name, double ms, ImU32 color, const char* tooltip)
        {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            coloredText(color, name);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tooltip);
            ImGui::TableNextColumn();
            const float share = period > 0 ? float(ms / period) : 0.0f;
            const ImVec2 p0   = ImGui::GetCursorScreenPos();
            const float w = ImGui::GetContentRegionAvail().x, h = ImGui::GetTextLineHeight() * 0.6f;
            const float y  = p0.y + (ImGui::GetTextLineHeight() - h) / 2;
            ImDrawList* dl = ImGui::GetWindowDrawList();
            dl->AddRectFilled(ImVec2(p0.x, y), ImVec2(p0.x + w, y + h), ImGui::GetColorU32(ImGuiCol_FrameBg));
            dl->AddRectFilled(ImVec2(p0.x, y), ImVec2(p0.x + w * std::clamp(share, 0.0f, 1.0f), y + h), color);
            ImGui::Dummy(ImVec2(w, ImGui::GetTextLineHeight()));
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("%.0f%% of the %.2f ms %s", 100 * share, period, vsync ? "refresh" : "frame");
            ImGui::TableNextColumn();
            // Next to its bar (left-aligned)
            std::snprintf(text, sizeof(text), "%.2f ms  %.0f%%", ms, 100 * share);
            ImGui::TextUnformatted(text);
        };
        row("CPU", sum.work.median, kCpuColor,
            "CPU busy (median): input, world steps, extract, recording and submit.\n"
            "Without the waits for the GPU and the display.");
        row("GPU", sum.gpu.median, kGpuColor, "GPU busy (median, timestamps)");
        ImGui::EndTable();
    }
}

void EngineApp::perfTable()
{
    const FrameStatistics::Summary& sum = shownStatistics.summary;
    const bool vsync                    = renderScene && renderScene->vsync();
    char text[400];
    // ------------------------------------------------------------------------------------ The table
    if (ImGui::BeginTable("frame times", 5, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp))
    {
        ImGui::TableSetupColumn("per frame (ms)", ImGuiTableColumnFlags_WidthStretch, 1.8f);
        for (const char* c : {"mean", "median", "p99", "max"})
            ImGui::TableSetupColumn(c, ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
        ImGui::TableNextColumn();
        ImGui::TableHeader("per frame (ms)");
        for (const char* c : {"mean", "median", "p99", "max"}) cellRight(c);
        auto row = [&](const char* name, ImU32 color, const FrameStatistics::Series& st, const char* tooltip)
        {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            coloredText(color, name);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tooltip);
            for (double v : {st.mean, st.median, st.p99, st.max})
            {
                std::snprintf(text, sizeof(text), "%.2f", v);
                cellRight(text);
            }
        };
        if (vsync && sum.shown.samples > 0)
            row("shown frame", kFrameColor, sum.shown,
                "The frames as the display shows them: each for whole refreshes (a late frame and the\n"
                "catch-up frame after it, absorbed by the swap chain queue, miss no refresh). fps, 1% low and\n"
                "max above are of these.");
        row(vsync ? "frame interval (CPU)" : "frame interval", kFrameColor, sum.interval,
            "From one frame's start to the next one's on the CPU. With vsync it jitters around the\n"
            "refresh; the display shows every frame for whole refreshes (fps and 1% low above).");
        row("CPU busy", kCpuColor, sum.work, "The CPU time of the frame without the waits for the GPU and the display");
        std::snprintf(text, sizeof(text),
                      "The frame interval without the CPU busy time: idle time, not work. Its parts (means):\n"
                      "frame slot %.2f ms (the GPU still runs an earlier frame), image acquire %.2f ms,\n"
                      "present %.2f ms (with vsync the display paces the acquire or the present).",
                      sum.waitParts[0], sum.waitParts[1], sum.waitParts[2]);
        row("CPU wait", ImGui::GetColorU32(ImGuiCol_TextDisabled), sum.idle, text);
        row("GPU busy", kGpuColor, sum.gpu, "The GPU time of the frame (timestamps)");
        ImGui::EndTable();
    }
}

void EngineApp::perfGraph(float height)
{
    // Every frame of the window: the frame intervals above, the busy times below (own scale); the legend and the
    // reference lines in the gutter on the right, the scales on the time axis (nothing over the data)
    const FrameStatistics::Summary& sum = shownStatistics.summary;
    const bool vsync                    = renderScene && renderScene->vsync();
    const double period                 = shownVerdict.period;
    const float frameScale              = float(std::max(2.2 * period, sum.interval.p99 * 1.2));
    std::vector<GraphLine> lines        = {{float(period), kRefColor, vsync ? "refresh" : "median"}};
    if (vsync) lines.push_back({float(2 * period), kWarnColor, "missed"});
    const float busyScale = float(std::max({sum.work.p99, sum.gpu.p99, 0.05}) * 1.5);
    frameGraph(frameStats, shownStatistics.graphSpan, frameScale, lines, busyScale, float(period), height);
}

void EngineApp::perfSimulation()
{
    // The world step: its time, then the systems sorted by their mean time (where a slow system shows), the counters
    const FrameStatistics::Summary& sum = shownStatistics.summary;
    const json11::Json& report          = shownStatistics.stepReport;
    if (report.is_null())
    {
        ImGui::TextDisabled("no world steps in the window (paused)");
        return;
    }
    const int shownSteps = report["samples"].int_value();
    for (const auto& sc : report["scopes"].array_items())
        if (sc["scope"].string_value() == "step")
            ImGui::Text("step %s ms mean, %s ms p99", msText(sc["avg"].number_value()).c_str(),
                        msText(sc["p99"].number_value()).c_str());
    ImGui::SameLine();
    ImGui::TextDisabled("(%d steps at %.0f/s%s)", shownSteps, 1.0 / gameWorld->dt(),
                        shownSteps < sum.steps ? ", all the profile keeps" : "");
    struct Row
    {
        std::string phase, system;
        double avg, p99;
    };
    std::vector<Row> rows;
    for (const auto& sc : report["scopes"].array_items())
    {
        const std::string& path = sc["scope"].string_value();
        const size_t sys        = path.find("/system:");
        if (path.rfind("step/phase:", 0) != 0 || sys == std::string::npos) continue;
        rows.push_back(
            {path.substr(11, sys - 11), path.substr(sys + 8), sc["avg"].number_value(), sc["p99"].number_value()});
    }
    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.avg > b.avg; });
    if (rows.empty())
        ImGui::TextDisabled("no systems timed (WorldSetup::profiling off)");
    else if (ImGui::BeginTable("##systems", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp))
    {
        ImGui::TableSetupColumn("system", ImGuiTableColumnFlags_WidthStretch, 2.4f);
        ImGui::TableSetupColumn("phase", ImGuiTableColumnFlags_WidthStretch, 1.4f);
        ImGui::TableSetupColumn("mean ms", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableSetupColumn("p99 ms", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        DebugUiStyle::headerRow({"system (slowest first)", "phase", "mean ms", "p99 ms"});
        for (const Row& r : rows)
        {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(r.system.c_str());
            ImGui::TableNextColumn();
            ImGui::TextDisabled("%s", r.phase.c_str());
            cellRight(msText(r.avg).c_str());
            cellRight(msText(r.p99).c_str());
        }
        ImGui::EndTable();
    }
    std::string counters;
    for (const auto& c : report["counters"].array_items())
    {
        char text[96];
        std::snprintf(text, sizeof(text), "%s%s %.0f", counters.empty() ? "" : ",  ",
                      c["counter"].string_value().c_str(), c["avg"].number_value());
        counters += text;
    }
    if (!counters.empty()) ImGui::TextWrapped("%s", counters.c_str());
}

void EngineApp::ensureShownStatistics()
{
    if (!shownStatistics.valid || std::chrono::steady_clock::now() - shownStatistics.at >=
                                      std::chrono::duration<double>(kStatisticsRefreshSeconds))
        refreshShownStatistics();
}

void EngineApp::perfGpu()
{
    // The GPU sections of the frame (RenderScene's timestamps): median and p95 over the last 2 s of GPU frames
    if (!renderScene)
    {
        ImGui::TextDisabled("no device");
        return;
    }
    if (!ImGui::BeginTable("gpu sections", 3, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) return;
    ImGui::TableSetupColumn("GPU section", ImGuiTableColumnFlags_WidthStretch, 2.0f);
    ImGui::TableSetupColumn("median ms", ImGuiTableColumnFlags_WidthStretch, 1.0f);
    ImGui::TableSetupColumn("p95 ms", ImGuiTableColumnFlags_WidthStretch, 1.0f);
    DebugUiStyle::headerRow({"GPU section", "median ms", "p95 ms"});
    const int samples = 120;
    std::vector<int> ids{scopeGpuFrame};
    ids.insert(ids.end(), scopeGpuSections.begin(), scopeGpuSections.end());
    for (int id : ids)
    {
        const Profiler::Stats st = gpuProfiler.scopeStats(id, samples);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        const std::string& name = gpuScopeNames[size_t(id)];
        coloredText(id == scopeGpuFrame ? kGpuColor : ImGui::GetColorU32(ImGuiCol_Text),
                    id == scopeGpuFrame ? "GPU frame" : name.c_str());
        cellRight(msText(st.p50).c_str());
        cellRight(msText(st.p95).c_str());
    }
    ImGui::EndTable();
}

void EngineApp::performanceTab()
{
    // docs/engine_design.md §17.1: what matters at a glance first (what limits the frame rate, fps and stutter, CPU and
    // GPU busy, the graph of every frame of the window), statistics over a time window refreshed every
    // kStatisticsRefreshSeconds (numbers that change every frame cannot be read). The details start collapsed
    ensureShownStatistics();
    perfSummary();
    ImGui::Spacing();
    // The graph gets the height the collapsed sections leave (measured before them: expanding one scrolls the panel,
    // the graph keeps its size)
    {
        const float em      = ImGui::GetFontSize();
        const int sections  = 2 + ui->sectionCount(DebugTab::Performance) + (showEngineGpuSections() ? 1 : 0);
        const float reserve = float(sections + 2) * ImGui::GetFrameHeightWithSpacing();
        perfGraph(std::max(ImGui::GetContentRegionAvail().y - reserve, 9 * em));
    }
    // The window, copy and reset in one line under the graph (wrapped when the panel is narrow)
    const FrameStatistics::Summary& sum = shownStatistics.summary;
    auto sameLineIfFits                 = [](const char* next)
    {
        const float w = ImGui::CalcTextSize(next).x + 2 * ImGui::GetStyle().FramePadding.x;
        if (ImGui::GetContentRegionAvail().x > w + ImGui::GetStyle().ItemSpacing.x) ImGui::SameLine();
    };
    ImGui::AlignTextToFramePadding();
    ImGui::TextDisabled("window");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(4.5f * ImGui::GetFontSize());
    const char* windows[] = {"3 s", "10 s", "30 s"};
    if (ImGui::Combo("##window", &statisticsWindowIndex, windows, 3)) shownStatistics.valid = false;
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("The time window of the numbers and the graph (now %.1f s, %d frames)", sum.seconds,
                          sum.interval.samples);
    sameLineIfFits("reset");
    if (ImGui::SmallButton("reset")) clearFrameStatistics();
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Starts the statistics anew");
    sameLineIfFits("copy JSON");
    if (ImGui::SmallButton("copy JSON"))
    {
        const ApiResult r = worldApi->call("get_profile", json11::Json::object{});
        ImGui::SetClipboardText((r.ok ? McpServer::FormatJson(r.value, true) : r.error).c_str());
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Copies the get_profile answer (step, frame and GPU profiles) to the clipboard");

    if (ui->section("performance/Frame times", "Frame timings", false)) perfTable();
    if (ui->section("performance/Simulation", "World step (systems)", false)) perfSimulation();
    // The engine's GPU sections only when no game section breaks the GPU down itself (the world prototype does)
    if (showEngineGpuSections() && ui->section("performance/GPU", "GPU sections", false)) perfGpu();
}

bool EngineApp::showEngineGpuSections() const
{
    return ui->sectionCount(DebugTab::Performance) == 0;
}

void EngineApp::barSummary(float width)
{
    // The bar's line (replaces PR #66's overlay): fps, frame, CPU and GPU busy, step; warnings; dropped from the right
    // when the bar is narrow (fps and the warnings stay)
    ensureShownStatistics();
    const FrameStatistics::Summary& sum   = shownStatistics.summary;
    const bool vsync                      = renderScene && renderScene->vsync();
    const FrameStatistics::Series& frames = vsync && sum.shown.samples > 0 ? sum.shown : sum.interval;
    const double stepMs                   = shownStatistics.stepMs;
    struct Item
    {
        std::string text;
        ImU32 color;
        int drop;  // drop order (higher: dropped first), 0: never
    };
    std::vector<Item> items;
    char text[96];
    const ImU32 normal = ImGui::GetColorU32(ImGuiCol_Text);
    if (frames.samples > 0)
        std::snprintf(text, sizeof(text), "%.0f fps", frames.mean > 0 ? 1000.0 / frames.mean : 0.0);
    else
        std::snprintf(text, sizeof(text), "-- fps");
    items.push_back({text, kFrameColor, 0});
    std::snprintf(text, sizeof(text), "CPU %.2f ms", sum.work.median);
    items.push_back({text, kCpuColor, 2});
    std::snprintf(text, sizeof(text), "GPU %.2f ms", sum.gpu.median);
    items.push_back({text, kGpuColor, 3});
    std::snprintf(text, sizeof(text), "step %.2f ms", stepMs);
    items.push_back({text, normal, 4});
    const int stutters = vsync ? sum.missedRefreshes : sum.longFrames;
    if (sum.seconds > 0 && stutters / sum.seconds > kStutterWarnPerSecond)
    {
        std::snprintf(text, sizeof(text), "%s %.1f/s", vsync ? "missed" : "long", stutters / sum.seconds);
        items.push_back({text, kWarnColor, 0});
    }
    if (shownStatistics.droppedSteps) items.push_back({"dropped steps", kWarnColor, 0});
    double runningFor     = 0;
    const std::string mcp = callLog ? callLog->longestRunning(&runningFor) : std::string();
    if (!mcp.empty() && runningFor > McpCallLog::kRunningSeconds)
        items.push_back({"MCP: " + mcp, DebugUiStyle::kMcp, 0});

    // Each number item at least as wide as its widest value, so the right-aligned items do not move
    const float gap = ImGui::GetFontSize();
    auto itemWidth  = [](const Item& i)
    {
        float w = ImGui::CalcTextSize(i.text.c_str()).x;
        if (i.drop > 0 || i.color == kFrameColor) w = std::max(w, ImGui::CalcTextSize("GPU 00.00 ms").x * 0.8f);
        return w;
    };
    auto total = [&]()
    {
        float w = 0;
        for (const Item& i : items) w += itemWidth(i) + gap;
        return w - gap;
    };
    for (int drop = 4; drop >= 1 && total() > width; --drop)
        items.erase(std::remove_if(items.begin(), items.end(), [&](const Item& i) { return i.drop == drop; }),
                    items.end());
    // PAUSED: a button (resumes) before the numbers
    const bool paused   = gameWorld->isPaused();
    const char* pausedL = "PAUSED (Pause)";
    const float pausedW = paused ? ImGui::CalcTextSize(pausedL).x + 2 * ImGui::GetStyle().FramePadding.x + gap : 0.0f;
    const float w       = total() + pausedW;
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(width - w, 0.0f));
    if (paused)
    {
        ImGui::PushStyleColor(ImGuiCol_Button, IM_COL32(150, 60, 50, 255));
        if (ImGui::Button(pausedL)) queueBetweenSteps([this]() { worldApi->call("resume", json11::Json::object{}); });
        ImGui::PopStyleColor();
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("The world is paused: click or press Pause to resume");
        ImGui::SameLine(0, gap);
    }
    ImGui::BeginGroup();
    for (size_t i = 0; i < items.size(); ++i)
    {
        if (i > 0) ImGui::SameLine(0, gap);
        ImGui::AlignTextToFramePadding();
        // Right-aligned in its fixed width
        const float pad = itemWidth(items[i]) - ImGui::CalcTextSize(items[i].text.c_str()).x;
        if (pad > 0) ImGui::SetCursorPosX(ImGui::GetCursorPosX() + pad);
        coloredText(items[i].color, items[i].text.c_str());
    }
    ImGui::EndGroup();
    if (ImGui::IsItemHovered())
    {
        std::string tip = vsync && shownStatistics.periodMs > 0
                              ? "vsync " + std::to_string(int(1000.0 / shownStatistics.periodMs + 0.5)) + " Hz\n"
                              : std::string("no vsync\n");
        if (frames.lowValid)
        {
            std::snprintf(text, sizeof(text), "1%% low %.0f fps\n", 1000.0 / frames.worst1Percent);
            tip += text;
        }
        std::snprintf(text, sizeof(text), "max frame %.1f ms\nmedians over %.0f s; click: Performance (F2)", frames.max,
                      kStatisticsWindows[size_t(statisticsWindowIndex)]);
        tip += text;
        ImGui::SetTooltip("%s", tip.c_str());
    }
    if (ImGui::IsItemClicked()) ui->select(DebugUi::kSummaryTab);
}

void EngineApp::togglePause()
{
    // The state when it runs (two toggles in one frame cancel out)
    queueBetweenSteps([this]() { worldApi->call(gameWorld->isPaused() ? "resume" : "pause", json11::Json::object{}); });
}

void EngineApp::engineTab()
{
    World& w = *gameWorld;
    if (ui->section("engine/Simulation", "Simulation"))
    {
        // Pause, step and time scale change the world: queued, applied between steps (after the next poll)
        if (ImGui::SmallButton(w.isPaused() ? "resume" : "pause")) togglePause();
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Pause key");
        ImGui::SameLine();
        if (ImGui::SmallButton("step 1"))
            queueBetweenSteps([this]() { worldApi->call("step", json11::Json::object{{"n", 1}}); });
        ImGui::SameLine();
        if (ImGui::SmallButton("step 10"))
            queueBetweenSteps([this]() { worldApi->call("step", json11::Json::object{{"n", 10}}); });
        if (DebugUiStyle::beginProperties("##simulation"))
        {
            DebugUiStyle::property(coop() ? "world (client)" : "world");
            ImGui::Text("%s", w.isPaused() ? "paused" : "running");
            DebugUiStyle::property("step");
            ImGui::Text("%llu  (%.0f steps/s)", (unsigned long long)w.stepIndex(), 1.0 / w.dt());
            DebugUiStyle::property("time scale");
            float scale = float(w.timeScale());
            if (ImGui::SliderFloat("##time scale", &scale, 0.1f, 4.0f, "%.2f", ImGuiSliderFlags_Logarithmic))
                queueBetweenSteps([this, scale]()
                                  { worldApi->call("time_scale", json11::Json::object{{"scale", double(scale)}}); });
            DebugUiStyle::property("dropped steps");
            ImGui::Text("%llu", (unsigned long long)w.droppedSteps());
            DebugUiStyle::property("entities");
            ImGui::Text("%u", w.registry().aliveCount());
            DebugUiStyle::endProperties();
        }
    }
    if (renderScene && ui->section("engine/Rendering", "Rendering"))
    {
        if (DebugUiStyle::beginProperties("##rendering"))
        {
            // The tone map and exposure of the frame (R-MECH-1; applied from the next frame)
            DebugUiStyle::property("tone map");
            int toneMap                = int(renderScene->parameters().toneMap);
            const char* toneMapNames[] = {"identity (clay)", "ACES"};
            if (ImGui::Combo("##tone map", &toneMap, toneMapNames, 2))
                renderScene->setToneMap(Vulkan::ToneMap(toneMap));
            // The exposure acts only on the ACES tone map (the range of --exposure)
            DebugUiStyle::property("exposure");
            float exposure = renderScene->parameters().exposure;
            ImGui::BeginDisabled(renderScene->parameters().toneMap == Vulkan::ToneMap::Identity);
            if (ImGui::SliderFloat("##exposure", &exposure, 1.0f / 1024, 1024, "%.3f", ImGuiSliderFlags_Logarithmic))
                renderScene->setExposure(exposure);
            ImGui::EndDisabled();
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                ImGui::SetTooltip("Acts only on the ACES tone map");
            DebugUiStyle::endProperties();
        }
        // What cannot be changed at runtime, in one line
        ImGui::TextDisabled("vsync %s, %dx MSAA, %d x %d", renderScene->vsync() ? "on" : "off",
                            renderScene->parameters().msaaSamples, int(renderScene->size()(0)),
                            int(renderScene->size()(1)));
    }
    // The terrain (S20): streaming per level, requests, uploads, memory, the stream time of complete points
    if (ui->section("engine/Terrain", "Terrain", false))
    {
        const TerrainSystem& terrain  = w.terrain();
        const TerrainSystem::Stats& t = terrain.stats();
        double gpuMb = 0, cpuMb = double(t.cpuBytes) / (1 << 20);
        if (renderScene)
        {
            gpuMb = double(renderScene->terrainAtlasStats().memory.gpuBytes()) / (1 << 20);
            cpuMb += double(renderScene->terrainAtlasStats().memory.cpuBytes()) / (1 << 20);
        }
        if (DebugUiStyle::beginProperties("##terrain"))
        {
            DebugUiStyle::property("requests / uploads");
            ImGui::Text("%d pending, %d tiles sent last frame", t.pending, t.uploadsLastUpdate);
            DebugUiStyle::property("upload bytes pending");
            ImGui::Text("%.0f", double(terrain.terrainUploadsPending()));
            DebugUiStyle::property("memory");
            ImGui::Text("CPU %.1f MB, GPU %.1f MB", cpuMb, gpuMb);
            DebugUiStyle::property("stream time");
            ImGui::Text("%.2f ms (last completeView %.2f ms)", streamMs, t.lastCompleteMs);
            DebugUiStyle::property("readiness at 256 m");
            ImGui::Text("%.3f", terrain.readiness(view.space, view.position, 256));
            // S21: the camera's space and a running load
            DebugUiStyle::property("camera space");
            ImGui::Text("%d '%s'", int(view.space),
                        view.space < w.spaceCount() ? w.spaceTemplate(view.space).name.c_str() : "?");
            DebugUiStyle::property("loading");
            if (loading)
                ImGui::Text("space %d, %.0f %%, %llu frames", int(loading->space), loading->progress * 100,
                            (unsigned long long)loading->frames);
            else
                ImGui::Text("no");
            DebugUiStyle::property("grass masks");
            ImGui::Text("%d desired, %d sent", t.masksDesired, t.masksSent);
            if (renderScene)
            {
                DebugUiStyle::property("patches ignored");
                ImGui::Text("%llu", (unsigned long long)renderScene->terrainAtlasStats().patchesIgnored);
            }
            DebugUiStyle::endProperties();
        }
        if (ImGui::BeginTable("##terrainLevels", 6, ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_RowBg))
        {
            for (const char* h : {"level", "desired", "truncated", "resident", "sent", "nodes"})
                ImGui::TableSetupColumn(h);
            ImGui::TableHeadersRow();
            const Vulkan::TerrainPass::Stats* pass = renderScene ? &renderScene->terrainPassStats() : nullptr;
            for (int k = 0; k < 8; ++k)
            {
                const auto& l   = t.levels[size_t(k)];
                const int nodes = pass ? pass->nodesPerLevel[size_t(k)] : 0;
                if (l.desired + l.resident + l.sent + nodes == 0) continue;
                ImGui::TableNextRow();
                for (int v : {k, l.desired, l.truncated, l.resident, l.sent, nodes})
                {
                    ImGui::TableNextColumn();
                    ImGui::Text("%d", v);
                }
            }
            ImGui::EndTable();
        }
    }
    if (ui->section("engine/Camera", "Camera and input", false))
    {
        if (DebugUiStyle::beginProperties("##camera"))
        {
            const Vec3 fwd = view.rotation * Vec3(0, 0, -1);
            DebugUiStyle::property("position");
            ImGui::Text("%.2f  %.2f  %.2f", view.position.x(), view.position.y(), view.position.z());
            DebugUiStyle::property("yaw / pitch");
            ImGui::Text("%.1f / %.1f deg", std::atan2(-fwd.x(), -fwd.z()) * 180 / 3.14159265358979323846,
                        std::asin(std::clamp(fwd.y(), -1.0, 1.0)) * 180 / 3.14159265358979323846);
            DebugUiStyle::property("field of view");
            ImGui::Text("%.1f deg", view.fovy * 180 / 3.14159265358979323846);
            DebugUiStyle::property("held by a tool");
            ImGui::Text("%s", cameraHeld ? "yes (set_camera release=true)" : "no");
            DebugUiStyle::property("local player");
            if (localPlayerEntity.valid())
                ImGui::Text("entity %u", localPlayerEntity.index);
            else
                ImGui::Text("none (free camera)");
            DebugUiStyle::property("move");
            const Vec2 m = inputMap.move();
            ImGui::Text("%.2f  %.2f", m.x(), m.y());
            DebugUiStyle::property("held buttons");
            ImGui::Text("0x%x", inputMap.heldBits());
            DebugUiStyle::endProperties();
        }
    }
}

void EngineApp::setupDebugUi()
{
    // The call log of every MCP tool call (agents and the console) into the engine log, category "mcp"
    callLog = std::make_unique<McpCallLog>();
    server->setCallObserver([this](const McpServer::CallRecord& r) { callLog->record(r); });
    server->addPollListener(
        [this]()
        {
            callLog->tick();
            return true;
        });
    // The state file only for an interactive window (screenshots, benchmarks and tests start from the defaults)
    const bool persist = appSetup.debugUiPersistence && !appSetup.headless && appOptions.screenshot.empty() &&
                         appOptions.benchmark.empty();
    ui = std::make_unique<DebugUi>(server.get(), persist ? std::string("debug_ui.json") : std::string());
    ui->setEnabled(appSetup.debugUi);
    ui->load();
    if (appOptions.showStatistics) ui->select(DebugTab::Performance);
    if (appOptions.debugTab >= 1) ui->select(appOptions.debugTab - 1);
    ui->setTabContent(DebugTab::Performance, [this]() { performanceTab(); });
    ui->setTabContent(DebugTab::Engine, [this]() { engineTab(); });
    ui->setSummary([this](float width) { barSummary(width); });
    ui->setPauseKey([this]() { togglePause(); });
}

}  // namespace Engine
}  // namespace Saiga
