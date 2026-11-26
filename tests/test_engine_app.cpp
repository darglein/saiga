/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The engine app (docs/engine_design.md §8.3; stage S13): command line, scenario runner, input map, the frame loop
// headless (with and without a Vulkan device), the app tools and the frame and GPU profiles.

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/imgui/imgui_internal.h"
#include "saiga/core/util/env.h"
#include "saiga/core/util/mcp/McpServer.h"
#include "saiga/engine/app/EngineApp.h"
#include "saiga/engine/app/EngineMain.h"
#include "saiga/engine/app/FrameStatistics.h"
#include "saiga/engine/app/LightingTools.h"
#include "saiga/engine/components/LightSource.h"
#include "saiga/engine/components/Parent.h"
#include "saiga/engine/components/Player.h"
#include "saiga/engine/components/PlayerInput.h"
#include "saiga/engine/components/RenderInstance.h"
#include "saiga/engine/components/WorldTime.h"
#include "saiga/engine/core/Log.h"
#include "saiga/engine/gameplay/DestroyOwnedBy.h"
#include "saiga/engine/lighting/EnvironmentWrites.h"
#include "saiga/engine/net/Fragmenter.h"
#include "saiga/engine/net/LoopbackTransport.h"
#include "saiga/engine/net/Replication.h"
#include "saiga/engine/sync/ExtractSystem.h"
#include "saiga/engine/sync/RecordingRenderSink.h"
#include "saiga/engine/sync/VulkanRenderSink.h"
#include "saiga/engine/terrain/TerrainSystem.h"
#include "saiga/engine/tools/ScenarioList.h"
#include "saiga/engine/tools/ScenarioWorld.h"
#include "saiga/vulkan/Instance.h"
#include "saiga/vulkan/scene/LightingDebug.h"
#include "saiga/vulkan/scene/PrototypeWorld.h"
#include "saiga/vulkan/scene/RenderScene.h"

#include "gtest/gtest.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <map>
#include <thread>

#include "lighting_test_parameters.h"
#include "mcp_test_helper.h"
#include "streaming_test_setup.h"
#include "terrain_test_hooks.h"

using namespace Saiga;
using namespace Saiga::Engine;
using Json = json11::Json;

namespace
{
const bool g_noEcho = (setLogEcho(false), true);

void actions(ActionList& a)
{
    a.add(1, "jump");
    a.add(2, "use");
    a.addHeld(0, "walk");
}

EngineAppSetup appSetup(bool render)
{
    EngineAppSetup s;
    s.name            = "test_app";
    s.headless        = true;
    s.noRender        = !render;
    s.width           = 64;
    s.height          = 48;
    s.mcpPort         = 0;
    s.world.actions   = actions;
    s.render.imgui    = false;
    s.render.lighting = testLightingParameters();
    s.validation      = render;  // a validation error fails the render tests (if the layer is installed)
    return s;
}

Json obj(const std::string& text)
{
    std::string err;
    Json j = Json::parse(text, err);
    EXPECT_TRUE(err.empty()) << err;
    return j;
}

Json callTool(McpServer& server, const std::string& name, const Json& args)
{
    return server.handleMessage(McpRequest("tools/call", Json::object{{"name", name}, {"arguments", args}}));
}
Json toolJson(const Json& r)
{
    std::string err;
    return Json::parse(r["result"]["content"][0]["text"].string_value(), err);
}

// setenv/unsetenv are POSIX (MSVC: _putenv_s, an empty value removes the variable)
void setEnv(const char* name, const char* value)
{
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value)
        setenv(name, value, 1);
    else
        unsetenv(name);
#endif
}

// Sets an environment variable for its scope and restores the previous value (or its absence)
struct ScopedEnv
{
    ScopedEnv(const char* name, const char* value) : name(name)
    {
        const auto old = GetEnvIfSet(name);
        had            = old.has_value();
        if (had) previous = *old;
        setEnv(name, value);
    }
    ~ScopedEnv() { setEnv(name.c_str(), had ? previous.c_str() : nullptr); }
    std::string name, previous;
    bool had = false;
};

// A new empty directory under the test temp directory (removed with the object)
struct TempDirectory
{
    TempDirectory()
    {
        // create_directory is false for an existing path (another process, a stale directory): then another name
        static int counter = 0;
        const std::filesystem::path base(::testing::TempDir());
        for (int attempt = 0; attempt < 1000; ++attempt)
        {
            path = base /
                   ("saiga_engine_app_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
                    "_" + std::to_string(counter++));
            std::error_code ec;
            if (std::filesystem::create_directory(path, ec)) return;
        }
        path.clear();  // never remove a directory this object did not create
        ADD_FAILURE() << "could not create a private temporary directory in " << base;
    }
    ~TempDirectory()
    {
        std::error_code ec;
        if (!path.empty()) std::filesystem::remove_all(path, ec);
    }
    std::filesystem::path path;
};

std::vector<const char*> argv(std::initializer_list<const char*> args)
{
    std::vector<const char*> v{"game"};
    v.insert(v.end(), args.begin(), args.end());
    return v;
}
}  // namespace

// ---------------------------------------------------------------------------------------------------------------------
// Command line and scenarios (no device)

TEST(AppCommandLine, OptionsGameFlagsAndErrors)
{
    EngineAppSetup s;
    s.name = "flight";
    AppOptions o;
    std::string error;
    auto a = argv({"--headless", "--width", "320", "--height", "200", "--no-vsync", "--validation", "--mcp-port", "0",
                   "--screenshot", "x.png", "--screenshot-frame", "5", "--demo", "--data", "/tmp/d"});
    ASSERT_TRUE(parseCommandLine(int(a.size()), a.data(), {"--demo"}, s, o, error)) << error;
    EXPECT_TRUE(s.headless);
    EXPECT_EQ(s.width, 320);
    EXPECT_EQ(s.height, 200);
    EXPECT_FALSE(s.vsync);
    EXPECT_TRUE(s.validation);
    EXPECT_EQ(s.mcpPort, 0);
    EXPECT_EQ(o.screenshot, "x.png");
    EXPECT_EQ(o.screenshotFrame, 5);
    EXPECT_TRUE(o.has("--demo"));
    EXPECT_EQ(s.world.dataDirectory, "/tmp/d");
    // Regression (CodeRabbit #83): the games read their environment file before the command line, so --data did not
    // select it; onDataDirectory runs after it with the final directory (also the default one)
    std::string seen;
    s.onDataDirectory = [&seen](EngineAppSetup& x) { seen = x.world.dataDirectory; };
    auto e            = argv({"--data", "/tmp/e"});
    ASSERT_TRUE(parseCommandLine(int(e.size()), e.data(), {}, s, o, error)) << error;
    EXPECT_EQ(seen, "/tmp/e");

    EngineAppSetup t;
    t.name = "flight";
    AppOptions p;
    auto b = argv({"--scenarios", "jump", "--no-render"});
    ASSERT_TRUE(parseCommandLine(int(b.size()), b.data(), {}, t, p, error)) << error;
    EXPECT_TRUE(p.runScenarios);
    EXPECT_EQ(p.scenarioFilter, "jump");
    EXPECT_TRUE(t.noRender && t.headless);
    // The default data directory: data/<name> of the source tree
    EXPECT_NE(t.world.dataDirectory.find("data/flight"), std::string::npos);

    // S17: the tone map and exposure of RenderScene
    EngineAppSetup v;
    AppOptions vo;
    auto c = argv({"--tonemap", "aces", "--exposure", "2.5"});
    ASSERT_TRUE(parseCommandLine(int(c.size()), c.data(), {}, v, vo, error)) << error;
    EXPECT_TRUE(v.render.toneMap == Vulkan::ToneMap::Aces);
    EXPECT_FLOAT_EQ(v.render.exposure, 2.5f);
    auto d = argv({"--exposure", "0.5", "--tonemap", "identity"});
    ASSERT_TRUE(parseCommandLine(int(d.size()), d.data(), {}, v, vo, error)) << error;
    EXPECT_TRUE(v.render.toneMap == Vulkan::ToneMap::Identity);
    EXPECT_FLOAT_EQ(v.render.exposure, 0.5f);

    for (auto bad :
         {argv({"--tonemap"}), argv({"--tonemap", "reinhard"}), argv({"--exposure", "0"}), argv({"--exposure", "-1"}),
          argv({"--exposure", "x"}), argv({"--exposure", "inf"}), argv({"--exposure", "nan"}), argv({"--exposure"})})
    {
        EngineAppSetup u;
        AppOptions q;
        EXPECT_FALSE(parseCommandLine(int(bad.size()), bad.data(), {}, u, q, error)) << bad[1];
    }

    for (auto bad : {argv({"--unknown"}), argv({"--width"}), argv({"--width", "0"}), argv({"--benchmark"}),
                     argv({"--benchmark", "flyover", "--headless"}), argv({"--benchmark", "nope"}), argv({"--demo"}),
                     argv({"--no-render", "--screenshot", "x.png"}), argv({"--data", "--headless"})})
    {
        EngineAppSetup u;
        u.benchmarks = {"flyover"};
        AppOptions q;
        EXPECT_FALSE(parseCommandLine(int(bad.size()), bad.data(), {}, u, q, error)) << bad[1];
    }
}

// S36: --time-of-day / --day-length (valued engine options; applied after the game's start)
TEST(AppCommandLine, TimeOfDayAndDayLength)
{
    EngineAppSetup s;
    AppOptions o;
    std::string error;
    auto a = argv({"--time-of-day", "18.5", "--day-length", "240"});
    ASSERT_TRUE(parseCommandLine(int(a.size()), a.data(), {}, s, o, error)) << error;
    ASSERT_TRUE(o.timeOfDay && o.dayLength);
    EXPECT_EQ(*o.timeOfDay, 18.5);
    EXPECT_EQ(*o.dayLength, 240);
    for (auto bad : {argv({"--time-of-day", "24"}), argv({"--time-of-day", "-1"}), argv({"--time-of-day", "x"}),
                     argv({"--time-of-day"}), argv({"--day-length", "0"}), argv({"--day-length", "10"}),
                     argv({"--day-length", "nan"}), argv({"--time-of-day", "6", "--scenarios"}),
                     argv({"--scenarios", "--day-length", "60"})})
    {
        EngineAppSetup u;
        AppOptions q;
        std::string args;
        for (size_t i = 1; i < bad.size(); ++i) args += std::string(i > 1 ? " " : "") + bad[i];
        EXPECT_FALSE(parseCommandLine(int(bad.size()), bad.data(), {}, u, q, error)) << args;
    }
    // A world without WorldTime stops with exit code 1 (before the frame loop)
    EngineAppSetup plain = appSetup(false);
    auto b               = argv({"--time-of-day", "6"});
    EXPECT_EQ(engineMain(int(b.size()), b.data(), plain, {}, nullptr), 1);
}

// S36: the app owns the environment settings; --time-of-day reaches the world
TEST(EngineAppNoRender, EnvironmentSettingsAndTimeOption)
{
    EngineAppSetup s                    = appSetup(false);
    s.world.environment.enabled         = true;
    s.environment.exposure.compensation = 2;
    EngineApp app(s);
    EXPECT_EQ(app.environmentSettings().exposure.compensation, 2);
    Vulkan::EnvironmentSettings e = app.environmentSettings();
    e.ambient.scale               = 3;
    app.setEnvironmentSettings(e);
    EXPECT_EQ(app.environmentSettings().ambient.scale, 3);
    EXPECT_FALSE(app.sendsEnvironment() && app.scene());
    const EnvironmentSample sample = app.sampleViewEnvironment();
    EXPECT_TRUE(sample.hasTime);
    EXPECT_FLOAT_EQ(sample.state.ev100, sample.curveEv100 - 2);
    EXPECT_EQ(sample.state.ambientScale, 3);
}

namespace
{
void passingScenarios(ScenarioList& l)
{
    l.add("steps", [](ScenarioWorld& w) { w.step(3); });
}
void failingScenarios(ScenarioList& l)
{
    l.add("fails", [](ScenarioWorld& w) { w.check(false, "on purpose"); });
}
}  // namespace

TEST(AppScenarios, ExitCodes)
{
    EngineAppSetup s;
    s.scenarios = passingScenarios;
    testing::internal::CaptureStdout();
    EXPECT_EQ(runGameScenarios(s, ""), 0);
    EXPECT_EQ(runGameScenarios(s, "nothing"), 1);
    s.scenarios = failingScenarios;
    EXPECT_EQ(runGameScenarios(s, ""), 1);
    const std::string out = testing::internal::GetCapturedStdout();
    EXPECT_NE(out.find("no scenarios matched"), std::string::npos) << out;
    EXPECT_NE(out.find("on purpose"), std::string::npos) << out;
    EXPECT_NE(out.find("3 passed, 0 failed"), std::string::npos) << out;
}

// ---------------------------------------------------------------------------------------------------------------------
// Input map

TEST(InputMapTest, KeysMouseHeldAndEdges)
{
    ActionList a;
    actions(a);
    InputMap m;
    std::string error;
    ASSERT_TRUE(m.parse(obj(R"({"bindings": {"jump": ["Space"], "walk": ["LeftShift", "Mouse2"]},
                                "axes": {"move_x": {"negative": ["A"], "positive": ["D"]},
                                         "move_z": {"negative": ["S"], "positive": ["W"]},
                                         "look_x": {"mouse": "x", "scale": 0.5}},
                                "move": ["move_x", "move_z"]})"),
                        "input.json", a, error))
        << error;
    InputState s;
    s.keys['D']       = true;
    s.keys['W']       = true;
    s.keys[32]        = true;  // Space
    s.mouseButtons[1] = true;
    s.mouseDelta      = Vec2(4, 0);
    m.update(s, false);
    EXPECT_EQ(m.axis("move_x"), 1);
    EXPECT_EQ(m.axis("look_x"), 2);
    EXPECT_NEAR(m.move().norm(), 1.0, 1e-12);  // the diagonal is shortened to length 1
    EXPECT_TRUE(m.pressed("jump"));
    EXPECT_EQ(m.pressedActions(), std::vector<uint32_t>{1});
    EXPECT_EQ(m.heldBits(), 1u);
    m.update(s, false);
    EXPECT_FALSE(m.pressed("jump"));  // an edge only in the first frame
    EXPECT_TRUE(m.held("jump"));
    // ImGui has the keyboard: nothing from the devices
    m.update(s, true);
    EXPECT_TRUE(m.neutral());
    // A virtual device (set_input)
    VirtualInput v;
    v.axes["move_z"] = 0.5;
    v.pressed        = {"use"};
    m.update(InputState(), false, v);
    EXPECT_EQ(m.move().y(), 0.5);
    EXPECT_EQ(m.pressedActions(), std::vector<uint32_t>{2});
}

// Regression (CodeRabbit, S13): setActions kept a pointer, so a temporary action list dangled
TEST(InputMapTest, SetActionsKeepsACopy)
{
    InputMap m;
    m.setActions(
        []()
        {
            ActionList a;
            actions(a);
            return a;
        }());
    m.setDefaults();
    VirtualInput v;
    v.held    = {"walk"};
    v.pressed = {"jump"};
    m.update(InputState(), false, v);
    EXPECT_EQ(m.heldBits(), 1u);
    EXPECT_EQ(m.pressedActions(), std::vector<uint32_t>{1});
    EXPECT_FALSE(m.describe()["actions"].array_items().empty()) << m.describe().dump();
}

TEST(InputMapTest, ErrorsNameTheFileAndKey)
{
    ActionList a;
    actions(a);
    InputMap m;
    std::string error;
    EXPECT_FALSE(m.parse(obj(R"({"bindings": {"fly": ["Space"]}})"), "in.json", a, error));
    EXPECT_NE(error.find("in.json: bindings.fly: not a registered"), std::string::npos) << error;
    EXPECT_FALSE(m.parse(obj(R"({"bindings": {"jump": ["Spacebar"]}})"), "in.json", a, error));
    EXPECT_NE(error.find("unknown key"), std::string::npos) << error;
    EXPECT_FALSE(m.parse(obj(R"({"axes": {"x": {"positive": ["A"], "speed": 1}}})"), "in.json", a, error));
    EXPECT_NE(error.find("axes.x.speed: unknown field"), std::string::npos) << error;
    EXPECT_FALSE(m.parse(obj(R"({"move": ["a", "b"]})"), "in.json", a, error));
    EXPECT_FALSE(m.parse(obj(R"({"keys": {}})"), "in.json", a, error));
}

// SAIGA_SCREENSHOT is ignored with --no-render (an argument is an error, see above)
TEST(AppCommandLine, ScreenshotVariableWithoutRendering)
{
    {
        ScopedEnv env("SAIGA_SCREENSHOT", "x.png");
        EngineAppSetup s;
        AppOptions o;
        std::string error;
        auto a = argv({"--no-render"});
        EXPECT_TRUE(parseCommandLine(int(a.size()), a.data(), {}, s, o, error)) << error;
        EXPECT_TRUE(o.screenshot.empty());
    }
    std::string error;
    // A known benchmark passes
    EngineAppSetup t;
    t.benchmarks = {"flyover"};
    AppOptions q;
    auto b = argv({"--benchmark", "flyover"});
    EXPECT_TRUE(parseCommandLine(int(b.size()), b.data(), {}, t, q, error)) << error;
}

// ---------------------------------------------------------------------------------------------------------------------
// The statistics window's frame statistics: the vsync wait is idle time, not work, and single long frames show

TEST(FrameStatistics, WindowTailAndStutter)
{
    FrameStatistics fs;
    // 200 frames at a 6 ms refresh with 0.5 ms work, then two frames that missed one refresh each
    for (int i = 0; i < 200; ++i) fs.addFrame(6, 0.5);
    fs.addFrame(12, 0.5);
    fs.addFrame(12, 0.5);
    for (int i = 0; i < 3; ++i) fs.addGpu(1, fs.now());
    FrameStatistics::Summary s = fs.summarize(10);
    EXPECT_EQ(s.interval.samples, 202);
    EXPECT_NEAR(s.seconds, 1.224, 1e-9);
    EXPECT_EQ(s.interval.median, 6);
    EXPECT_EQ(s.interval.max, 12);
    EXPECT_EQ(s.work.median, 0.5);
    EXPECT_EQ(s.idle.median, 5.5);  // the vsync wait: idle, not work
    EXPECT_EQ(s.gpu.samples, 3);
    // The slowest 1%: ceil(202 / 100) = 3 frames (12, 12, 6)
    EXPECT_EQ(s.interval.worstCount, 3);
    EXPECT_NEAR(s.interval.worst1Percent, 10, 1e-12);
    EXPECT_TRUE(s.interval.lowValid);
    // A missed refresh is an interval of exactly 2 refreshes: counted (a "> 2 x median" test missed it)
    EXPECT_EQ(s.missedRefreshes, 2);
    EXPECT_EQ(s.longFrames, 2);
    // The window: the last 30 ms are the two long frames and one before them
    s = fs.summarize(0.030);
    EXPECT_EQ(s.interval.samples, 3);
    EXPECT_FALSE(s.interval.lowValid);  // fewer than 100 frames
    EXPECT_EQ(s.gpu.samples, 3);        // the GPU frames came at the end
    // With the display's refresh period: a median at 2 refreshes (every frame misses one) still counts them all
    FrameStatistics slow;
    for (int i = 0; i < 100; ++i) slow.addFrame(12, 7);
    EXPECT_EQ(slow.summarize(10).missedRefreshes, 0);  // the median looks like the refresh
    EXPECT_EQ(slow.summarize(10, 6).missedRefreshes, 100);
    // The refresh period: measured (p10), the reported one only if it fits (a whole multiple)
    EXPECT_EQ(s.onTimeMedian, 6);
    EXPECT_EQ(FrameStatistics::refreshPeriod(s, 0), 6);
    EXPECT_EQ(FrameStatistics::refreshPeriod(s, 240), 6);  // another monitor's rate: 6 / 4.17 is no whole multiple
    EXPECT_NEAR(FrameStatistics::refreshPeriod(slow.summarize(10), 165), 1000.0 / 165, 1e-9);  // 12 ms = 2 refreshes
    // Regression (review): a relative tolerance accepted 240 Hz for frames of 2 refreshes at 165 Hz (12.1 ms is
    // close to 3 x 4.17 ms)
    EXPECT_EQ(FrameStatistics::refreshPeriod(slow.summarize(10), 240), 12);
    // The shown frames with vsync: CPU-side jitter around the refresh is no stutter, a missed refresh is
    FrameStatistics jitter;
    for (int i = 0; i < 200; ++i) jitter.addFrame(i % 2 == 0 ? 5.2 : 6.8, 0.5);
    FrameStatistics::Summary j = jitter.summarize(10, 6);
    EXPECT_EQ(j.shown.max, 6);
    EXPECT_EQ(j.missedRefreshes, 0);
    EXPECT_TRUE(j.alternating);  // the CPU-side intervals alternate
    // Regression (review): each frame rounded on its own made a late frame and its catch-up frame (9.1 + 3.0 ms
    // = 2 refreshes of 6.06 ms) one missed refresh; the queue absorbed it
    EXPECT_EQ(FrameStatistics::shownRefreshes({9.1, 3.0}, 6.06), (std::vector<int>{1, 1}));
    EXPECT_EQ(FrameStatistics::shownRefreshes({12.1, 6.06}, 6.06), (std::vector<int>{2, 1}));  // a real miss
    EXPECT_EQ(FrameStatistics::shownRefreshes({5.2, 6.8, 5.2, 6.8}, 6.0), (std::vector<int>{1, 1, 1, 1}));
    // Regression (CodeRabbit): after an absorbed late frame the grid position kept the extra refresh, so a real miss
    // right after it was lost
    EXPECT_EQ(FrameStatistics::shownRefreshes({9.1, 3.0, 12.12}, 6.06), (std::vector<int>{1, 1, 2}));
    FrameStatistics hitch;
    for (int i = 0; i < 50; ++i) hitch.addFrame(6.06, 0.5);
    hitch.addFrame(9.1, 0.5);
    hitch.addFrame(3.0, 0.5);
    EXPECT_EQ(hitch.summarize(10, 6.06).missedRefreshes, 0);
    EXPECT_NEAR(hitch.summarize(10, 6.06).shown.max, 6.06, 1e-9);
    // The session's period: a window where every frame misses keeps the session's period; a display change to a
    // matching reported rate (144 -> 72 Hz) is taken
    double session = 0;
    EXPECT_NEAR(FrameStatistics::sessionRefreshPeriod(6.06, 240, session), 6.06, 1e-9);
    EXPECT_NEAR(FrameStatistics::sessionRefreshPeriod(12.12, 240, session), 6.06, 1e-9);
    session = 1000.0 / 144;
    EXPECT_NEAR(FrameStatistics::sessionRefreshPeriod(1000.0 / 72, 72, session), 1000.0 / 72, 1e-9);
    EXPECT_NEAR(session, 1000.0 / 72, 1e-9);
    // The work is clamped to the interval
    fs.addFrame(1, 5);
    EXPECT_EQ(fs.summarize(0.0005).work.max, 1);
}

TEST(FrameStatistics, PacingAndTrim)
{
    // Alternating 4 and 8 ms: mean and median look fine, the frame pacing does not
    FrameStatistics fs(1.0);
    for (int i = 0; i < 400; ++i) fs.addFrame(i % 2 == 0 ? 4 : 8, 1);
    FrameStatistics::Summary s = fs.summarize(10);
    EXPECT_EQ(s.pacingP99, 4);
    EXPECT_TRUE(s.alternating);
    // Regression (CodeRabbit): isolated spikes (8 ms every 20 frames of 2 ms) flip the sign of every change above
    // 0.1 ms, but they are single stutters, not alternation
    FrameStatistics spikes;
    for (int i = 0; i < 600; ++i) spikes.addFrame(i % 20 == 0 ? 8 : 2, 1);
    EXPECT_FALSE(spikes.summarize(100).alternating);
    // Random jitter (about 2 of 3 changes flip their sign) is no alternation
    FrameStatistics noise;
    uint32_t x = 12345;
    for (int i = 0; i < 400; ++i)
    {
        x = x * 1664525u + 1013904223u;
        noise.addFrame(4.0 + 2.0 * double(x >> 8) / double(1u << 24), 1);
    }
    EXPECT_FALSE(noise.summarize(10).alternating);
    EXPECT_EQ(s.shortMs, 4);
    EXPECT_EQ(s.longMs, 8);
    // Only the last second is kept
    EXPECT_NEAR(s.seconds, 1.0, 0.01);
    EXPECT_LE(fs.frames(), 168u);
    EXPECT_EQ(fs.recentIntervals(4), (std::vector<float>{4, 8, 4, 8}));
    // Suspended rendering: the clock advances, the old frames age out, no long frame is added
    fs.advance(2000);
    EXPECT_EQ(fs.summarize(1).interval.samples, 0);
    fs.addFrame(4, 1, {}, 2);
    EXPECT_EQ(fs.summarize(1).steps, 2);
    fs.clear();
    EXPECT_EQ(fs.summarize(10).interval.samples, 0);
}

// S25: --save <dir> and its refusals with --scenarios and --benchmark
TEST(AppCommandLine, SaveDirectory)
{
    EngineAppSetup s;
    AppOptions o;
    std::string error;
    auto a = argv({"--save", "/tmp/s25_save"});
    ASSERT_TRUE(parseCommandLine(int(a.size()), a.data(), {}, s, o, error)) << error;
    EXPECT_EQ(s.saveDirectory, "/tmp/s25_save");
    for (auto bad : {argv({"--save"}), argv({"--save", "d", "--scenarios"}), argv({"--save", "d", "--benchmark", "x"})})
    {
        EngineAppSetup u;
        u.benchmarks = {"x"};
        AppOptions q;
        EXPECT_FALSE(parseCommandLine(int(bad.size()), bad.data(), {}, u, q, error)) << bad[1];
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// The app without a device (servers, agents, CI without Vulkan)

// S25: a headless app with --save (no file syncs: the protocol is covered by the memory file system): an edit through
// edit_terrain, quit commits; a second app on the directory has the edit
TEST(EngineAppNoRender, SaveDirectoryQuitCommits)
{
    const std::string dir =
        ::testing::TempDir() + "saiga_s25_app_" + std::to_string(std::hash<std::string>{}(::testing::TempDir()));
    std::filesystem::remove_all(dir);
    EngineAppSetup s = appSetup(false);
    s.saveDirectory  = dir;
    s.saveSync       = false;
    double edited    = 0;
    {
        EngineApp app(s);
        app.openSaveDirectory();
        EXPECT_TRUE(app.world().hasSaveDirectory());
        app.runFrames(2, 1.0 / 60);
        Poller poller(app.mcp());
        EXPECT_FALSE(
            callTool(app.mcp(), "edit_terrain",
                     obj(R"({"op": "add", "center": [10, 0, 10], "radius": 3, "strength": 1})"))["result"]["isError"]
                .bool_value());
        edited = app.world().terrainHeight(0, 10, 10);
        EXPECT_NE(edited, 0);
        EXPECT_FALSE(callTool(app.mcp(), "quit", Json::object{})["result"]["isError"].bool_value());
    }
    {
        EngineApp app(s);
        app.openSaveDirectory();
        EXPECT_EQ(app.world().terrainHeight(0, 10, 10), edited);
        EXPECT_EQ(app.world().saveStatus().committed, 2u);
    }
    std::filesystem::remove_all(dir);
}

TEST(EngineAppNoRender, AdvancesAndAnswersTools)
{
    EngineApp app(appSetup(false));
    EXPECT_EQ(app.scene(), nullptr);
    app.runFrames(60, 1.0 / 60);
    EXPECT_EQ(app.world().stepIndex(), 60u);
    EXPECT_EQ(app.frameCount(), 60u);
    // The frame profile is in get_profile
    {
        Poller poller(app.mcp());
        Json p = toolJson(callTool(app.mcp(), "get_profile", Json::object{}));
        EXPECT_TRUE(p["frame"].is_object()) << p.dump();
        EXPECT_GE(p["frame"]["samples"].number_value(), 60);
        Json st = toolJson(callTool(app.mcp(), "app_status", Json::object{}));
        EXPECT_TRUE(st["headless"].bool_value());
        EXPECT_FALSE(st["rendering"].bool_value());
        EXPECT_TRUE(st["local_player"].is_null());
        // set_input needs a local player
        EXPECT_TRUE(
            callTool(app.mcp(), "set_input", obj(R"({"axes": {"move_x": 1}})"))["result"]["isError"].bool_value());
        // queue_action by name (the registered actions)
        Json d = toolJson(callTool(app.mcp(), "describe_world", Json::object{}));
        EXPECT_EQ(d["actions"].array_items().size(), 2u);
        EXPECT_FALSE(callTool(app.mcp(), "quit", Json::object{})["result"]["isError"].bool_value());
    }
    EXPECT_TRUE(app.quitting());
    // A quitting app runs no more frames
    app.runFrames(5, 1.0 / 60);
    EXPECT_EQ(app.frameCount(), 60u);
}

// S22: with a fixed frame time of the world's dt every frame() runs exactly one step, also when the frames take
// longer or shorter than dt (the terraform benchmark's stepping inside run())
TEST(EngineAppNoRender, FixedFrameTimeStepsOncePerFrame)
{
    EngineApp app(appSetup(false));
    app.setFixedFrameTime(app.world().dt());
    for (int i = 0; i < 12; ++i)
    {
        // A slow frame (3 dt), a fast one (no sleep)
        if (i % 2 == 0) std::this_thread::sleep_for(std::chrono::duration<double>(3 * app.world().dt()));
        const uint64_t before = app.world().stepIndex();
        app.frame();
        EXPECT_EQ(app.world().stepIndex(), before + 1) << "frame " << i;
    }
    // runFrames with another dt: still one step per frame
    app.runFrames(5, 0.1);
    EXPECT_EQ(app.world().stepIndex(), 17u);
    // Back to the measured time
    app.setFixedFrameTime(0);
    EXPECT_EQ(app.fixedFrameTime(), 0.0);
    app.runFrames(3, 0);
    EXPECT_EQ(app.world().stepIndex(), 17u);
}

// The local player gets the input; neutral input does not overwrite what an agent set
TEST(EngineAppNoRender, InputToTheLocalPlayer)
{
    EngineApp app(appSetup(false));
    World& w       = app.world();
    const Entity p = w.create();
    w.add<Player>(p);
    w.add<PlayerInput>(p);
    app.setLocalPlayer(p);
    // An agent's value survives neutral input
    w.get<PlayerInput>(p).move = Vec3(0.25, 0, 0);
    app.runFrames(3, 1.0 / 60);
    EXPECT_EQ(w.read<PlayerInput>(p).move.x(), 0.25);
    // A virtual device drives it (move relative to the camera; actions queued once)
    VirtualInput v;
    v.held    = {"walk"};
    v.pressed = {"jump"};
    app.setVirtualInput(v, 2);
    // Held and pressed names count (the default bindings have no keys for them)
    app.runFrames(1, 1.0 / 60);
    EXPECT_EQ(w.read<PlayerInput>(p).heldButtons, 1u);
    // Queued in the frame, delivered by its step
    ASSERT_EQ(w.events.read<ActionEvent>().size(), 1u);
    EXPECT_EQ(w.events.read<ActionEvent>()[0].action, 1u);
    app.runFrames(1, 1.0 / 60);
    EXPECT_TRUE(w.events.read<ActionEvent>().empty());  // pressed once
    // Released right after its second step (before another step could read it)
    EXPECT_EQ(w.read<PlayerInput>(p).heldButtons, 0u);
    // Regression: the release reached the player only through the next active input, the held button stayed down. It is
    // written once; later neutral frames leave an agent's value alone again.
    app.runFrames(1, 1.0 / 60);
    EXPECT_EQ(w.read<PlayerInput>(p).heldButtons, 0u);
    w.get<PlayerInput>(p).heldButtons = 2;
    app.runFrames(2, 1.0 / 60);
    EXPECT_EQ(w.read<PlayerInput>(p).heldButtons, 2u);
    // set_input through the tool: the default axes move the player; unknown names are rejected
    {
        Poller poller(app.mcp());
        Json ok = callTool(app.mcp(), "set_input", obj(R"({"axes": {"move_x": 1}, "steps": 0})"));
        EXPECT_FALSE(ok["result"]["isError"].bool_value()) << ok.dump();
        EXPECT_TRUE(
            callTool(app.mcp(), "set_input", obj(R"({"axes": {"mvoe_x": 1}})"))["result"]["isError"].bool_value());
        EXPECT_TRUE(callTool(app.mcp(), "set_input", obj(R"({"held": ["fly"]})"))["result"]["isError"].bool_value());
        Json st = toolJson(callTool(app.mcp(), "app_status", Json::object{}));
        EXPECT_EQ(st["input"]["actions"].array_items().size(), 2u) << st.dump();
        EXPECT_EQ(st["input"]["held_buttons"].array_items().size(), 1u) << st.dump();
        EXPECT_TRUE(st["input"]["axes"]["move_x"].is_object()) << st.dump();
    }
    app.runFrames(1, 1.0 / 60);
    EXPECT_GT(w.read<PlayerInput>(p).move.norm(), 0.5);
    // After a load the local player is still the same entity
    const auto blob = w.save();
    ASSERT_TRUE(w.tryLoad(blob).ok());
    EXPECT_EQ(app.localPlayer(), p);
}

// set_input lasts world steps, not frames: a paused world (or a frame shorter than a step) keeps it until a step read
// it (a move set while paused was released before the next step)
TEST(EngineAppNoRender, VirtualInputCountsSteps)
{
    EngineApp app(appSetup(false));
    World& w       = app.world();
    const Entity p = w.create();
    w.add<Player>(p);
    w.add<PlayerInput>(p);
    app.setLocalPlayer(p);
    w.setPaused(true);
    VirtualInput v;
    v.held = {"walk"};
    app.setVirtualInput(v, 1);
    app.runFrames(3, 1.0 / 60);
    EXPECT_EQ(w.stepIndex(), 0u);
    EXPECT_EQ(w.read<PlayerInput>(p).heldButtons, 1u);
    w.setPaused(false);
    app.runFrames(1, 1.0 / 240);  // shorter than a step: none runs
    EXPECT_EQ(w.read<PlayerInput>(p).heldButtons, 1u);
    app.runFrames(4, 1.0 / 240);  // the step runs with the input, the next frame releases it
    EXPECT_GE(w.stepIndex(), 1u);
    app.runFrames(1, 1.0 / 60);
    EXPECT_EQ(w.read<PlayerInput>(p).heldButtons, 0u);
}

// Regression (CodeRabbit, S13): steps run between frames (world().step(), the step tool) did not count against
// set_input, so the input stayed active on a paused world after its steps
TEST(EngineAppNoRender, VirtualInputCountsStepsBetweenFrames)
{
    EngineApp app(appSetup(false));
    World& w       = app.world();
    const Entity p = w.create();
    w.add<Player>(p);
    w.add<PlayerInput>(p);
    app.setLocalPlayer(p);
    w.setPaused(true);
    VirtualInput v;
    v.held = {"walk"};
    app.setVirtualInput(v, 1);
    app.runFrames(1, 1.0 / 60);
    EXPECT_EQ(w.read<PlayerInput>(p).heldButtons, 1u);
    w.step();  // the one step reads the input
    app.runFrames(2, 1.0 / 60);
    EXPECT_EQ(w.read<PlayerInput>(p).heldButtons, 0u);  // released before another step could read it
    // Within one world().step(n): released right after the step it covers (the second step reads neutral input)
    std::vector<uint32_t> seen;
    const int id = w.addStepListener(
        [&]()
        {
            seen.push_back(w.read<PlayerInput>(p).heldButtons);
            return true;
        });
    app.setVirtualInput(v, 1);
    app.runFrames(1, 1.0 / 60);
    EXPECT_EQ(w.read<PlayerInput>(p).heldButtons, 1u);
    w.step(3);
    EXPECT_EQ(seen, (std::vector<uint32_t>{0, 0, 0}));  // after each step (the release ran after the first)
    // A multi-step advance: the same
    seen.clear();
    w.setPaused(false);
    app.setVirtualInput(v, 1);
    app.runFrames(1, 0.0);  // applies the input, no step
    EXPECT_EQ(w.read<PlayerInput>(p).heldButtons, 1u);
    w.advance(3.0 / 60 + 1e-6);
    EXPECT_EQ(seen, (std::vector<uint32_t>{0, 0, 0}));
    w.removeStepListener(id);
}

// Regression (CodeRabbit, S13): a pending release went to the new local player, the previous one kept its input
TEST(EngineAppNoRender, ChangingTheLocalPlayerReleasesThePreviousOne)
{
    EngineApp app(appSetup(false));
    World& w       = app.world();
    const Entity a = w.create();
    const Entity b = w.create();
    for (Entity e : {a, b})
    {
        w.add<Player>(e);
        w.add<PlayerInput>(e);
    }
    app.setLocalPlayer(a);
    VirtualInput v;
    v.held = {"walk"};
    app.setVirtualInput(v, 0);
    app.runFrames(1, 1.0 / 60);
    EXPECT_EQ(w.read<PlayerInput>(a).heldButtons, 1u);
    w.get<PlayerInput>(b).heldButtons = 2;  // an agent's value on b
    app.setLocalPlayer(b);
    EXPECT_EQ(w.read<PlayerInput>(a).heldButtons, 0u);
    app.setVirtualInput(VirtualInput(), 1);
    app.runFrames(1, 1.0 / 60);
    EXPECT_EQ(w.read<PlayerInput>(b).heldButtons, 2u);  // no release of a's input on b
}

// Regression (CodeRabbit, S13): set_input was written to the player only by the next frame, so a step run before it
// (world().step(), the step tool) read the old input and then expired the request
TEST(EngineAppNoRender, VirtualInputAppliesBeforeAnExplicitStep)
{
    EngineAppSetup s = appSetup(false);
    int jumps        = 0;
    s.applyInput     = [&](const InputMap& in, World&, Entity) { jumps += in.pressed("jump") ? 1 : 0; };
    EngineApp app(s);
    World& w       = app.world();
    const Entity p = w.create();
    w.add<Player>(p);
    w.add<PlayerInput>(p);
    app.setLocalPlayer(p);
    w.setPaused(true);
    VirtualInput v;
    v.held    = {"walk"};
    v.pressed = {"jump"};
    app.setVirtualInput(v, 1);
    EXPECT_EQ(w.read<PlayerInput>(p).heldButtons, 1u);  // before any frame
    EXPECT_EQ(jumps, 1);
    w.step();  // reads it; released after the step
    EXPECT_EQ(w.read<PlayerInput>(p).heldButtons, 0u);
    app.runFrames(2, 1.0 / 60);
    EXPECT_EQ(jumps, 1);  // the pressed action once
}

// Regression (CodeRabbit, S13): a change of the local player released only PlayerInput on the previous one; the game's
// own controls (applyInput) of the previous player stayed active
TEST(EngineAppNoRender, ChangingTheLocalPlayerReleasesTheGameInput)
{
    EngineAppSetup s = appSetup(false);
    std::vector<std::pair<Entity, bool>> calls;  // (player, input neutral)
    s.applyInput = [&](const InputMap& in, World&, Entity e) { calls.push_back({e, in.neutral()}); };
    EngineApp app(s);
    World& w       = app.world();
    const Entity a = w.create();
    const Entity b = w.create();
    for (Entity e : {a, b})
    {
        w.add<Player>(e);
        w.add<PlayerInput>(e);
    }
    app.setLocalPlayer(a);
    VirtualInput v;
    v.held = {"walk"};
    app.setVirtualInput(v, 0);
    ASSERT_FALSE(calls.empty());
    EXPECT_EQ(calls.back(), std::make_pair(a, false));
    calls.clear();
    app.setLocalPlayer(b);
    ASSERT_EQ(calls.size(), 1u);
    EXPECT_EQ(calls[0], std::make_pair(a, true));  // a neutral update of a, nothing for b
}

// The local player's handle is dead after a load: the first player of peer 0 in the loaded world
TEST(EngineAppNoRender, LocalPlayerFallbackAfterALoad)
{
    EngineApp app(appSetup(false));
    World& w       = app.world();
    const Entity b = w.create();
    w.add<Player>(b);
    const auto blob = w.save();
    const Entity a  = w.create();
    w.add<Player>(a);
    app.setLocalPlayer(a);
    ASSERT_TRUE(w.tryLoad(blob).ok());
    EXPECT_FALSE(w.alive(a));
    EXPECT_EQ(app.localPlayer(), b);
}

// ---------------------------------------------------------------------------------------------------------------------
// S14: camera and scenario tools (no device)

namespace
{
// A tool call while a Poller runs poll() (and optionally frames) on its own thread; the Poller is gone when it returns,
// so the test thread may read the app afterwards
Json callApp(EngineApp& app, const std::string& name, const Json& args, bool frames = false)
{
    Poller poller(app.mcp(),
                  frames ? std::function<void()>([&app]() { app.runFrames(1, 1.0 / 60); }) : std::function<void()>());
    return callTool(app.mcp(), name, args);
}
bool isError(const Json& r)
{
    return r["result"]["isError"].bool_value();
}
std::string errorText(const Json& r)
{
    return r["result"]["content"][0]["text"].string_value();
}

std::atomic_bool g_pollInStep{false};
EngineApp* g_app = nullptr;

void testScenarios(ScenarioList& l)
{
    l.add("walks",
          [](ScenarioWorld& w)
          {
              const Entity p = w.spawnPlayer(Vec3::Zero());
              w.move(p, Vec3(1, 0, 0));
              w.step(10);
              w.move(p, Vec3::Zero());
              w.step(1);
              w.check(w.world().read<Transform>(p).position().x() > 0.5, "walked");
          });
    l.add("fails",
          [](ScenarioWorld& w)
          {
              w.step(2);
              w.check(false, "always fails");
          });
    l.add("moving_at_end",
          [](ScenarioWorld& w)
          {
              const Entity p = w.spawnPlayer(Vec3::Zero());
              w.move(p, Vec3(1, 0, 0));
              w.step(5);
          });
    l.add("throws",
          [](ScenarioWorld& w)
          {
              w.step(1);
              throw std::runtime_error("boom");
          });
    l.add("slow",
          [](ScenarioWorld& w)
          {
              w.addStepListener([]() { std::this_thread::sleep_for(std::chrono::milliseconds(20)); });
              w.step(100000);
          });
    l.add("slow_short",
          [](ScenarioWorld& w)
          {
              w.addStepListener([]() { std::this_thread::sleep_for(std::chrono::milliseconds(30)); });
              w.step(10);
          });
    // One step, then longer than run_scenario's answer deadline (0.8 s timeout with the margin 0.5: 0.4 s) but shorter
    // than the timeout (0.2 s to either side)
    l.add("late_finish",
          [](ScenarioWorld& w)
          {
              w.step(1);
              std::this_thread::sleep_for(std::chrono::milliseconds(600));
          });
    l.add("reload",
          [](ScenarioWorld& w)
          {
              std::string error;
              w.check(!w.world().reloadData(error), "World::reloadData is refused");
              w.check(!w.call("reload_data", Json::object{}).ok, "the reload_data operation is refused");
          });
}

void pollInStepSystem(World&, double)
{
    if (g_pollInStep && g_app) g_app->mcp().poll();
}
void testSystems(SystemList& s)
{
    s.add(Phase::Gameplay, "poll_in_step", pollInStepSystem);
}

EngineAppSetup scenarioSetup()
{
    EngineAppSetup s = appSetup(false);
    s.scenarios      = testScenarios;
    s.world.systems  = testSystems;
    return s;
}
}  // namespace

TEST(EngineAppNoRender, FrameToolsNeedADevice)
{
    EngineApp app(appSetup(false));
    EXPECT_TRUE(isError(callApp(app, "screenshot", Json::object{})));
    const Json r = callApp(app, "measure_performance", Json::object{{"frames", 5}});
    EXPECT_TRUE(isError(r));
    EXPECT_NE(errorText(r).find("renders no frames"), std::string::npos) << errorText(r);
    EXPECT_TRUE(isError(callApp(app, "measure_performance", Json::object{{"frames", 5}, {"steps", 5}})));
}

TEST(EngineAppNoRender, SetAndMoveCamera)
{
    EngineApp app(appSetup(false));
    Json c = toolJson(callApp(app, "set_camera", obj(R"({"position": [10, 5, 20], "yaw_deg": 90, "pitch_deg": 0})")));
    EXPECT_NEAR(c["position"][0].number_value(), 10, 1e-3) << c.dump();
    EXPECT_NEAR(c["forward"][0].number_value(), 1, 1e-3) << c.dump();  // yaw 90: +x
    EXPECT_TRUE(c["held"].bool_value());
    // look_at wins over yaw/pitch
    c = toolJson(callApp(app, "set_camera", obj(R"({"look_at": [10, 5, 0], "yaw_deg": 90})")));
    EXPECT_NEAR(c["forward"][2].number_value(), -1, 1e-3) << c.dump();
    // Errors: non-finite or out-of-range values, unknown arguments, release with others
    EXPECT_TRUE(isError(callApp(app, "set_camera", obj(R"({"position": [1e7, 0, 0]})"))));
    EXPECT_TRUE(isError(callApp(app, "set_camera", obj(R"({"position": [1, 2]})"))));
    EXPECT_TRUE(isError(callApp(app, "set_camera", obj(R"({"fovy_deg": 50})"))));
    EXPECT_TRUE(isError(callApp(app, "set_camera", obj(R"({"release": true, "yaw_deg": 3})"))));
    // move_camera: along the ground plane, the pitch clamped
    c = toolJson(callApp(app, "move_camera", obj(R"({"forward": 4, "pitch_deg": 200})")));
    EXPECT_NEAR(c["position"][2].number_value(), 16, 1e-2) << c.dump();  // from z = 20, facing -z
    // Clamped to 89. Regression (CodeRabbit, S14): the view kept |forward.y| <= 0.999, so 89 became about 87.4
    EXPECT_NEAR(c["pitch_deg"].number_value(), 89, 0.05) << c.dump();
    c = toolJson(callApp(app, "move_camera", obj(R"({"pitch_deg": -89})")));  // back to level
    EXPECT_NEAR(c["pitch_deg"].number_value(), 0, 0.05) << c.dump();
    c = toolJson(callApp(app, "move_camera", obj(R"({"pitch_deg": 88})")));
    EXPECT_NEAR(c["pitch_deg"].number_value(), 88, 0.05) << c.dump();
    // Straight down at a target: (almost) vertical, finite, the heading kept (-z)
    c = toolJson(callApp(app, "set_camera", obj(R"({"position": [0, 10, 0], "look_at": [0, 0, 0]})")));
    EXPECT_LE(c["pitch_deg"].number_value(), -89.85) << c.dump();
    EXPECT_NEAR(c["yaw_deg"].number_value(), 0, 0.05) << c.dump();
    EXPECT_TRUE(std::isfinite(c["forward"][1].number_value())) << c.dump();
    // The hold in app_status, then released
    EXPECT_TRUE(toolJson(callApp(app, "app_status", Json::object{}))["camera_held"].bool_value());
    c = toolJson(callApp(app, "set_camera", obj(R"({"release": true})")));
    EXPECT_FALSE(app.cameraHeldByTool());
}

// Relative moves of the input use the heading of the (held) camera
TEST(EngineAppNoRender, InputMovesRelativeToAHeldCamera)
{
    EngineApp app(appSetup(false));
    World& w       = app.world();
    const Entity p = w.create();
    w.add<Player>(p);
    w.add<PlayerInput>(p);
    app.setLocalPlayer(p);
    callApp(app, "set_camera", obj(R"({"yaw_deg": 90, "pitch_deg": 0})"));
    VirtualInput v;
    v.axes["move_z"] = 1;
    app.setVirtualInput(v, 0);
    const Vec3 m = w.read<PlayerInput>(p).move;
    EXPECT_NEAR(m.x(), 1, 1e-3) << m.transpose();
    EXPECT_NEAR(m.z(), 0, 1e-3) << m.transpose();
}

TEST(EngineAppNoRender, ListAndRunScenarios)
{
    EngineApp app(scenarioSetup());
    Json l = toolJson(callApp(app, "list_scenarios", obj(R"({"filter": "walk"})")));
    ASSERT_EQ(l["scenarios"].array_items().size(), 1u) << l.dump();
    EXPECT_EQ(l["scenarios"][0]["name"].string_value(), "walks");
    // A passing run in two variants; no snapshot without keep / load
    Json r = toolJson(callApp(app, "run_scenario", obj(R"({"name": "walks", "variants": ["Plain", "Repeat"]})")));
    ASSERT_EQ(r["results"].array_items().size(), 2u) << r.dump();
    EXPECT_TRUE(r["results"][0]["passed"].bool_value()) << r.dump();
    EXPECT_EQ(r["results"][1]["variant"].string_value(), "Repeat");
    EXPECT_FALSE(app.api().hasSnapshot("scenario_walks"));
    // A failing run has its report
    r = toolJson(callApp(app, "run_scenario", obj(R"({"name": "fails"})")));
    EXPECT_FALSE(r["results"][0]["passed"].bool_value());
    EXPECT_NE(r["results"][0]["report"].string_value().find("always fails"), std::string::npos) << r.dump();
    // Errors
    EXPECT_TRUE(isError(callApp(app, "run_scenario", obj(R"({"name": "nope"})"))));
    EXPECT_TRUE(isError(callApp(app, "run_scenario", obj(R"({"name": "walks", "variants": ["Fast"]})"))));
    EXPECT_TRUE(
        isError(callApp(app, "run_scenario", obj(R"({"name": "walks", "variants": ["Repeat"], "load": true})"))));
    // The tool-run worlds refuse to reload the shared data
    r = toolJson(callApp(app, "run_scenario", obj(R"({"name": "reload"})")));
    EXPECT_TRUE(r["results"][0]["passed"].bool_value()) << r.dump();
    // keep stores the end state; a throwing scenario stores nothing
    r = toolJson(callApp(app, "run_scenario", obj(R"({"name": "walks", "keep": true})")));
    EXPECT_EQ(r["snapshot"].string_value(), "scenario_walks") << r.dump();
    EXPECT_TRUE(app.api().hasSnapshot("scenario_walks"));
    r = toolJson(callApp(app, "run_scenario", obj(R"({"name": "throws", "keep": true})")));
    EXPECT_FALSE(app.api().hasSnapshot("scenario_throws")) << r.dump();
    EXPECT_FALSE(r["snapshot_error"].string_value().empty()) << r.dump();
}

TEST(EngineAppNoRender, RunScenarioLoadsTheEndState)
{
    EngineApp app(scenarioSetup());
    World& w            = app.world();
    const Entity before = w.create();
    w.step(3);
    const uint64_t gameHash = w.hash();
    Json r                  = toolJson(callApp(app, "run_scenario", obj(R"({"name": "walks", "load": true})")));
    ASSERT_TRUE(r["loaded"].is_object()) << r.dump();
    // The live world has the scenario's end state (neutral input at its end: the hash equals the snapshot's), paused
    EXPECT_TRUE(w.isPaused());
    EXPECT_EQ(r["loaded"]["hash"].string_value(), r["snapshot_hash"].string_value()) << r.dump();
    EXPECT_TRUE(app.liveWorldFromScenario());
    // A second load keeps the first before_scenario (the game's state)
    r = toolJson(callApp(app, "run_scenario", obj(R"({"name": "walks", "load": true})")));
    ASSERT_TRUE(r["loaded"].is_object()) << r.dump();
    ApiResult back = app.api().call("load_snapshot", obj(R"({"name": "before_scenario"})"));
    ASSERT_TRUE(back.ok) << back.error;
    EXPECT_EQ(w.hash(), gameHash);
    EXPECT_TRUE(w.alive(before));
    EXPECT_FALSE(app.liveWorldFromScenario());
    // A scenario that ends with input held: the load neutralizes the local player's input, so the loaded hash differs
    r = toolJson(callApp(app, "run_scenario", obj(R"({"name": "moving_at_end", "load": true})")));
    ASSERT_TRUE(r["loaded"].is_object()) << r.dump();
    EXPECT_NE(r["loaded"]["hash"].string_value(), r["snapshot_hash"].string_value()) << r.dump();
    // Deleting before_scenario makes the next load save it again
    ASSERT_TRUE(app.api().call("delete_snapshot", obj(R"({"name": "before_scenario"})")).ok);
    callApp(app, "run_scenario", obj(R"({"name": "walks", "load": true})"));
    EXPECT_TRUE(app.api().hasSnapshot("before_scenario"));
}

TEST(EngineAppNoRender, RunScenarioSnapshotSlotsAndDeadline)
{
    // The answer deadline after 0.3 s, 0.3 s before the timeout
    EngineAppSetup s          = scenarioSetup();
    s.mcpToolTimeoutSeconds   = 0.6;
    s.mcpAnswerMarginFraction = 0.5;
    EngineApp app(s);
    // The deadline stops a long scenario: no snapshot even with keep
    Json r = toolJson(callApp(app, "run_scenario", obj(R"({"name": "slow", "keep": true})")));
    ASSERT_EQ(r["results"].array_items().size(), 1u) << r.dump();
    EXPECT_FALSE(r["results"][0]["passed"].bool_value());
    EXPECT_EQ(r["results"][0]["failures"][0].string_value(), "aborted: stopped: tool time budget") << r.dump();
    EXPECT_FALSE(app.api().hasSnapshot("scenario_slow"));
    // Regression (S14 review): a Repeat stopped by the deadline reported a false determinism failure ("the runs took
    // N and 1 steps"); now only the abort, and the remaining variants are skipped
    r = toolJson(callApp(app, "run_scenario", obj(R"({"name": "slow", "variants": ["Repeat", "SaveLoad"]})")));
    ASSERT_EQ(r["results"].array_items().size(), 2u) << r.dump();
    ASSERT_EQ(r["results"][0]["failures"].array_items().size(), 1u) << r.dump();
    EXPECT_EQ(r["results"][0]["failures"][0].string_value(), "aborted: stopped: tool time budget") << r.dump();
    EXPECT_TRUE(r["results"][1]["skipped"].bool_value()) << r.dump();
    // Full snapshot slots are checked before the run
    const auto blob = app.world().save();
    for (size_t i = 0; app.api().snapshotCount() < WorldApi::kMaxSnapshotCount; ++i)
        ASSERT_TRUE(app.api().putSnapshot("s" + std::to_string(i), blob, 0, 0, false).ok);
    r = callApp(app, "run_scenario", obj(R"({"name": "walks", "keep": true})"));
    EXPECT_TRUE(isError(r));
    EXPECT_NE(errorText(r).find("full"), std::string::npos) << errorText(r);
}

// Regression (CodeRabbit, S14): a Plain run that finished after the deadline (no step saw the clock) was still stored
// and, with load, replaced the live world although the client may already have had its timeout
TEST(EngineAppNoRender, RunScenarioFinishingAfterTheDeadlineIsNotLoaded)
{
    EngineAppSetup s          = scenarioSetup();
    s.mcpToolTimeoutSeconds   = 0.8;
    s.mcpAnswerMarginFraction = 0.5;
    EngineApp app(s);
    app.world().step(3);
    const uint64_t liveHash = app.world().hash();
    const Json r            = toolJson(callApp(app, "run_scenario", obj(R"({"name": "late_finish", "load": true})")));
    EXPECT_NE(r["snapshot_error"].string_value().find("after the tool's deadline"), std::string::npos) << r.dump();
    EXPECT_FALSE(r["loaded"].is_object()) << r.dump();
    EXPECT_FALSE(app.api().hasSnapshot("scenario_late_finish"));
    EXPECT_EQ(app.world().hash(), liveHash);
    EXPECT_FALSE(app.world().isPaused());
}

// run_scenario is refused inside a step (a system polls MCP during the step; that poll is the only poll site)
TEST(EngineAppNoRender, RunScenarioRefusedInsideAStep)
{
    EngineApp app(scenarioSetup());
    g_app = &app;
    std::atomic_bool done{false};
    Json result;
    std::thread caller(
        [&]()
        {
            result = callTool(app.mcp(), "run_scenario", obj(R"({"name": "walks"})"));
            done   = true;
        });
    g_pollInStep = true;
    for (int i = 0; i < 2000 && !done; ++i)
    {
        app.world().step();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    g_pollInStep = false;
    caller.join();
    g_app = nullptr;
    EXPECT_TRUE(isError(result));
    EXPECT_NE(errorText(result).find("between steps"), std::string::npos) << errorText(result);
}

// Regression (S14): the frame after a blocking tool call (run_scenario) had the blocked time as its dt: a burst of
// steps and dropped steps
TEST(EngineAppNoRender, FrameClockRestartsAfterABlockingTool)
{
    EngineApp app(scenarioSetup());
    app.frame();  // starts the frame clock
    {
        // The call blocks about 300 ms inside a poll (between frames)
        Poller poller(app.mcp());
        const Json r = toolJson(callTool(app.mcp(), "run_scenario", obj(R"({"name": "slow_short"})")));
        EXPECT_TRUE(r["results"][0]["passed"].bool_value()) << r.dump();
    }
    app.frame();  // the first frame after the call
    Profiler& f = app.frameProfile();
    // Without the restart the blocked ~300 ms come out as a burst of about 18 steps; a few steps of jitter (the
    // poller's shutdown, Windows' coarse timer: 3 steps on the MSVC CI runner) are not the bug
    EXPECT_LE(f.counterStats(f.counter("steps_per_frame"), 1).max, 4);
    EXPECT_EQ(f.counterStats(f.counter("dropped_steps"), 1).max, 0);
}

// Device input is transient: a load neutralizes the local player's restored held input
TEST(EngineAppNoRender, LoadNeutralizesTheLocalPlayersInput)
{
    EngineApp app(appSetup(false));
    World& w       = app.world();
    const Entity p = w.create();
    w.add<Player>(p);
    w.add<PlayerInput>(p);
    w.get<PlayerInput>(p).move        = Vec3(1, 0, 0);
    w.get<PlayerInput>(p).heldButtons = 1;
    const auto blob                   = w.save();
    app.setLocalPlayer(p);
    ASSERT_TRUE(w.tryLoad(blob).ok());
    EXPECT_EQ(w.read<PlayerInput>(p).move, Vec3::Zero());
    EXPECT_EQ(w.read<PlayerInput>(p).heldButtons, 0u);
}

// ---------------------------------------------------------------------------------------------------------------------
// The app with a headless Vulkan device (lavapipe in CI; skipped without a driver)

class EngineAppRender : public ::testing::Test
{
   protected:
    void SetUp() override
    {
        if (!Vulkan::Instance::suitableDeviceAvailable()) GTEST_SKIP() << "No Vulkan 1.3 device available";
    }
};

TEST_F(EngineAppRender, FramesStepsScreenshotsAndProfiles)
{
    EngineAppSetup s     = appSetup(true);
    s.render.msaaSamples = 4;
    s.render.clearColor  = vec4(1, 0, 0, 1);
    // S19b: beforeRender once per rendered frame with the world's alpha (transfer and render are gone)
    int beforeRender           = 0;
    s.renderHooks.beforeRender = [&](EngineApp& a, double alpha)
    {
        EXPECT_EQ(alpha, a.world().alpha());
        beforeRender++;
    };
    EngineApp app(s);
    ASSERT_NE(app.scene(), nullptr);
    int rendered = 0;
    app.onFrameRendered(
        [&]()
        {
            rendered++;
            return rendered == 10;  // removed after 10
        });
    // Regression (CodeRabbit, S13): a callback was called as a copy, so its own state never advanced (a mutable counter
    // that removes itself after 10 calls ran forever); one that adds a callback while it runs
    int stateful = 0, added = 0;
    app.onFrameRendered(
        [&stateful, n = 0]() mutable
        {
            stateful++;
            return ++n == 10;
        });
    app.onFrameRendered(
        [&]()
        {
            app.onFrameRendered(
                [&]()
                {
                    added++;
                    return true;
                });
            return true;
        });
    app.runFrames(60, 1.0 / 60);
    EXPECT_EQ(stateful, 10);
    EXPECT_EQ(added, 1);
    EXPECT_EQ(app.world().stepIndex(), 60u);
    EXPECT_EQ(app.renderedFrames(), 60u);
    EXPECT_EQ(beforeRender, 60);
    EXPECT_EQ(rendered, 10);
    // Rendering disabled: the world keeps stepping
    app.setRenderingEnabled(false);
    app.runFrames(30, 1.0 / 60);
    EXPECT_EQ(app.world().stepIndex(), 90u);
    EXPECT_EQ(app.renderedFrames(), 60u);
    app.setRenderingEnabled(true);
    // A screenshot of the clear color at the rendered size
    TemplatedImage<ucvec4> shot;
    app.scene()->requestScreenshot([&](const TemplatedImage<ucvec4>& img) { shot = img; });
    app.runFrames(1, 1.0 / 60);
    ASSERT_EQ(shot.width, 64);
    ASSERT_EQ(shot.height, 48);
    EXPECT_EQ(int(shot(10, 10)(0)), 255);
    EXPECT_EQ(int(shot(10, 10)(1)), 0);
    // GPU times arrive a few frames later (if the device has timestamps)
    app.runFrames(5, 1.0 / 60);
    {
        Poller poller(app.mcp());
        Json p = toolJson(callTool(app.mcp(), "get_profile", Json::object{}));
        EXPECT_TRUE(p["gpu"].is_object()) << p.dump();
        EXPECT_GT(p["gpu"]["samples"].number_value(), 0) << p.dump();
        Json st = toolJson(callTool(app.mcp(), "app_status", Json::object{}));
        EXPECT_TRUE(st["rendering"].bool_value());
        EXPECT_EQ(st["size"][0].number_value(), 64);
    }
    // The frame's wall time splits into the CPU work and the render's waits (frame slot, image, present), so the
    // statistics do not show waiting for vsync as CPU time; the scopes are in get_profile
    {
        Profiler& fp                    = app.frameProfile();
        const std::vector<double> frame = fp.scopeHistory(fp.scope("frame"));
        const std::vector<double> work  = fp.scopeHistory(fp.scope("work"));
        const std::vector<double> wait  = fp.scopeHistory(fp.scope("frame/render/wait"));
        // S20: the terrain stream time of a complete point (the first frame) is not work either
        const std::vector<double> stream = fp.scopeHistory(fp.scope("frame/render/terrain_stream"));
        ASSERT_EQ(frame.size(), work.size());
        ASSERT_EQ(frame.size(), wait.size());
        ASSERT_EQ(frame.size(), stream.size());
        ASSERT_FALSE(frame.empty());
        for (size_t i = 0; i < frame.size(); ++i)
        {
            EXPECT_GE(wait[i], 0) << i;
            EXPECT_LE(wait[i], frame[i]) << i;
            EXPECT_NEAR(work[i] + wait[i] + stream[i], frame[i], 1e-9) << i;
        }
        Poller poller(app.mcp());
        Json p     = toolJson(callTool(app.mcp(), "get_profile", Json::object{}));
        bool found = false;
        for (const auto& sc : p["frame"]["scopes"].array_items()) found |= sc["scope"].string_value() == "work";
        EXPECT_TRUE(found) << p.dump();
    }
    // quit from a frame callback ends run()
    app.onFrameRendered(
        [&]()
        {
            app.quit(3);
            return true;
        });
    EXPECT_EQ(app.run(), 3);
}

// Regression (review): frames that were never presented (rendering off, a minimized window: paced to the step rate)
// entered the frame statistics, so after a restore the window showed missed refreshes and a low 1% low for a while
TEST_F(EngineAppRender, FrameStatisticsOnlyPresentedFrames)
{
    EngineApp app(appSetup(true));
    // The first second is left out
    app.runFrames(50, 1.0 / 60);
    EXPECT_EQ(app.frameStatistics().frames(), 0u);
    app.runFrames(30, 1.0 / 60);
    const size_t presented = app.frameStatistics().frames();
    EXPECT_GT(presented, 0u);
    app.setRenderingEnabled(false);
    app.runFrames(30, 1.0 / 60);
    // The last presented frame ends with the first frame without rendering
    EXPECT_LE(app.frameStatistics().frames(), presented + 1);
    const size_t before = app.frameStatistics().frames();
    app.setRenderingEnabled(true);
    app.runFrames(10, 1.0 / 60);
    EXPECT_EQ(app.frameStatistics().frames(), before + 9);
    // Their GPU times at their frames (a few frames late)
    EXPECT_GT(app.frameStatistics().gpuSamples().size(), 0u);
}

// --screenshot: saved after the given frame, then the app quits
TEST_F(EngineAppRender, ScreenshotOption)
{
    TempDirectory dir;
    ASSERT_FALSE(dir.path.empty());  // never write into the working directory
    AppOptions o;
    o.screenshot      = (dir.path / "shot.png").string();
    o.screenshotFrame = 3;
    EngineApp app(appSetup(true), o);
    EXPECT_EQ(app.run(), 0);
    EXPECT_EQ(app.renderedFrames(), 3u);
    TemplatedImage<ucvec4> img;
    EXPECT_TRUE(img.load(o.screenshot));
    // Regression: a screenshot that could not be saved exited with 0
    AppOptions bad;
    bad.screenshot      = (dir.path / "no_such_directory" / "shot.png").string();
    bad.screenshotFrame = 1;
    EngineApp app2(appSetup(true), bad);
    EXPECT_EQ(app2.run(), 1);
}

// Regression (CodeRabbit, S13): a callback added during the dispatch ran in the same frame (a chain that adds its
// successor never ended the frame), and a throwing callback left an empty slot (std::bad_function_call later)
TEST_F(EngineAppRender, FrameCallbacksAddedDuringDispatchRunNextFrame)
{
    EngineApp app(appSetup(true));
    std::vector<uint64_t> calls;  // the rendered frame of each call
    std::function<bool()> chain = [&]()
    {
        calls.push_back(app.renderedFrames());
        if (calls.size() < 3) app.onFrameRendered(chain);
        return true;
    };
    app.onFrameRendered(chain);
    app.runFrames(4, 1.0 / 60);
    EXPECT_EQ(calls, (std::vector<uint64_t>{1, 2, 3}));
    // A stateful callback keeps its state
    int count = 0;
    app.onFrameRendered(
        [&count, n = 0]() mutable
        {
            count = ++n;
            return n == 3;
        });
    app.runFrames(5, 1.0 / 60);
    EXPECT_EQ(count, 3);
    // A throwing callback is removed; the frames go on
    app.onFrameRendered([]() -> bool { throw std::runtime_error("callback failed"); });
    EXPECT_THROW(app.runFrames(1, 1.0 / 60), std::runtime_error);
    EXPECT_NO_THROW(app.runFrames(2, 1.0 / 60));
}

// A throwing init hook: shutdown runs for the GPU state it created (the constructor never finishes, no destructor runs)
TEST_F(EngineAppRender, ShutdownAfterAThrowingInit)
{
    EngineAppSetup s       = appSetup(true);
    bool shutdown          = false;
    s.renderHooks.init     = [](EngineApp&) { throw std::runtime_error("init failed"); };
    s.renderHooks.shutdown = [&](EngineApp&) { shutdown = true; };
    EXPECT_THROW(EngineApp app(s), std::runtime_error);
    EXPECT_TRUE(shutdown);
}

// S14: screenshot and the frames form of measure_performance through handleMessage
namespace
{
std::string base64Decode(const std::string& in)
{
    auto value = [](char c) -> int
    {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        return c == '+' ? 62 : c == '/' ? 63 : -1;
    };
    std::string out;
    // Unsigned: the accumulator shifts past 31 bits (only the low bits are read)
    uint32_t buffer = 0;
    int bits        = 0;
    for (char c : in)
    {
        const int v = value(c);
        if (v < 0) continue;
        buffer = (buffer << 6) | uint32_t(v);
        bits += 6;
        if (bits >= 8)
        {
            bits -= 8;
            out.push_back(char((buffer >> bits) & 0xFF));
        }
    }
    return out;
}
}  // namespace

TEST_F(EngineAppRender, ScreenshotTool)
{
    EngineAppSetup s = appSetup(true);
    s.width          = 256;
    s.height         = 144;
    s.render.imgui   = true;
    EngineApp app(s);
    app.scene()->setImguiVisible(true);
    auto image = [](const Json& r, int expectedWidth, const std::string& mime)
    {
        ASSERT_FALSE(r["result"]["isError"].bool_value()) << r.dump().substr(0, 400);
        const Json& c = r["result"]["content"];
        EXPECT_EQ(c[0]["mimeType"].string_value(), mime);
        const std::string data = base64Decode(c[0]["data"].string_value());
        TemplatedImage<ucvec3> img;
        ASSERT_TRUE(
            img.loadFromMemory(ArrayView<const char>(data.data(), data.size()), mime == "image/png" ? "png" : "jpg"));
        EXPECT_EQ(img.width, expectedWidth);
        std::string err;
        const Json text = Json::parse(c[1]["text"].string_value(), err);
        EXPECT_TRUE(text["camera"].is_object()) << c[1]["text"].string_value();
        EXPECT_GT(text["frame"].number_value(), 0);
    };
    image(callApp(app, "screenshot", Json::object{}, true), 256, "image/jpeg");
    image(callApp(app, "screenshot", obj(R"({"format": "png", "max_width": 128})"), true), 128, "image/png");
    // The GUI visibility is restored after the frame
    EXPECT_TRUE(app.scene()->imguiVisible());
    // The previous (hidden) state comes back after a gui: true screenshot
    app.scene()->setImguiVisible(false);
    image(callApp(app, "screenshot", obj(R"({"gui": true})"), true), 256, "image/jpeg");
    EXPECT_FALSE(app.scene()->imguiVisible());
    // Argument errors
    EXPECT_TRUE(isError(callApp(app, "screenshot", obj(R"({"max_width": 63})"), true)));
    EXPECT_TRUE(isError(callApp(app, "screenshot", obj(R"({"max_width": 8193})"), true)));
    EXPECT_TRUE(isError(callApp(app, "screenshot", obj(R"({"format": "bmp"})"), true)));
}

// Regression (S14): with no frame rendered the screenshot answers with an error at the deadline; a later frame does
// not answer a second time and the GUI flag is restored
TEST_F(EngineAppRender, ScreenshotDeadlineWithoutFrames)
{
    // The answer deadline after 0.3 s, 0.3 s before the timeout
    EngineAppSetup s          = appSetup(true);
    s.render.imgui            = true;
    s.mcpToolTimeoutSeconds   = 0.6;
    s.mcpAnswerMarginFraction = 0.5;
    EngineApp app(s);
    app.scene()->setImguiVisible(true);
    app.setRenderingEnabled(false);
    const Json r = callApp(app, "screenshot", Json::object{}, true);
    EXPECT_TRUE(isError(r));
    EXPECT_NE(errorText(r).find("deadline"), std::string::npos) << errorText(r);
    EXPECT_TRUE(app.scene()->imguiVisible());
    // Regression (CodeRabbit, S14): each expired request kept its frame callback (with its responder) while nothing was
    // rendered; now the requests share one readback and the deadline releases them
    EXPECT_TRUE(isError(callApp(app, "screenshot", obj(R"({"gui": true})"), true)));
    EXPECT_EQ(app.frameCallbackCount(), 1u);
    EXPECT_FALSE(app.scene()->imguiOverridden().has_value());
    app.setRenderingEnabled(true);
    app.runFrames(5, 1.0 / 60);  // the abandoned readback is a no-op
    EXPECT_TRUE(app.scene()->imguiVisible());
    EXPECT_EQ(app.frameCallbackCount(), 0u);
    EXPECT_GE(app.renderedFrames(), 5u);
}

// Regression (CodeRabbit, S14): the screenshot hid the GUI through the flag the G key toggles, so a toggle while it
// waited for its frame was undone afterwards; now the GUI option applies to the captured frame only
TEST_F(EngineAppRender, ScreenshotKeepsAGuiToggleMeanwhile)
{
    EngineAppSetup s = appSetup(true);
    s.render.imgui   = true;
    EngineApp app(s);
    app.scene()->setImguiVisible(true);
    Json result;
    std::thread caller([&]() { result = callTool(app.mcp(), "screenshot", Json::object{}); });
    // Until the request waits for its frame (the GUI override is set)
    for (int i = 0; i < 5000 && !app.scene()->imguiOverridden().has_value(); ++i)
    {
        app.mcp().poll();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_TRUE(app.scene()->imguiOverridden().has_value());
    EXPECT_FALSE(*app.scene()->imguiOverridden());
    app.scene()->setImguiVisible(false);  // the user presses G
    app.runFrames(1, 1.0 / 60);
    caller.join();
    EXPECT_FALSE(isError(result)) << result.dump().substr(0, 400);
    EXPECT_FALSE(app.scene()->imguiVisible());
    EXPECT_FALSE(app.scene()->imguiOverridden().has_value());
}

// Regression (CodeRabbit, S20): the screenshot's GUI override hid the GUI in every frame while the request waited (a
// capture now waits for the terrain); it applies only to a frame that is read back
TEST_F(EngineAppRender, ImguiOverrideOnlyInReadBackFrames)
{
    EngineAppSetup s = appSetup(true);
    s.render.imgui   = true;
    EngineApp app(s);
    app.scene()->setImguiVisible(true);
    app.scene()->setImguiOverride(false);
    app.runFrames(1, 1.0 / 60);
    EXPECT_TRUE(app.scene()->lastDrewImgui());
    TemplatedImage<ucvec4> shot;
    app.scene()->requestScreenshot([&](const TemplatedImage<ucvec4>& img) { shot = img; });
    app.runFrames(1, 1.0 / 60);
    EXPECT_GT(shot.width, 0);
    EXPECT_FALSE(app.scene()->lastDrewImgui());
    app.runFrames(1, 1.0 / 60);
    EXPECT_TRUE(app.scene()->lastDrewImgui());
    app.scene()->setImguiOverride(std::nullopt);
    EXPECT_TRUE(app.scene()->imguiVisible());
}

namespace
{
// measure_performance(frames: n) started between frames, run until it answers
Json measureDirect(EngineApp& app, int n, std::chrono::seconds budget = std::chrono::seconds(60))
{
    Json result;
    app.measureFrames(n, std::chrono::steady_clock::now() + budget,
                      [&](const McpServer::ToolResult& r)
                      {
                          std::string err;
                          result = Json::parse(r.content.empty() ? "" : r.content[0]["text"].string_value(), err);
                      });
    for (int i = 0; i < n + 100 && result.is_null(); ++i) app.runFrames(1, 1.0 / 60);
    return result;
}
}  // namespace

// S17: the gpu profile has RenderScene's sections in frame order (S19a: transfer, cull, depth and its sub-section,
// scene and its sub-sections, resolve, post, imgui); the tone map of the setup reaches RenderScene. S19b: the game's
// own passes (passNames / passTimes) are gone
TEST_F(EngineAppRender, GpuSections)
{
    EngineAppSetup s  = appSetup(true);
    s.render.toneMap  = Vulkan::ToneMap::Aces;
    s.render.exposure = 2;
    EngineApp app(s);
    ASSERT_NE(app.scene(), nullptr);
    app.runFrames(12, 1.0 / 60);
    EXPECT_TRUE(app.scene()->toneMap() == Vulkan::ToneMap::Aces);
    EXPECT_FLOAT_EQ(app.scene()->exposure(), 2.0f);
    EXPECT_TRUE(app.scene()->sceneWritesLinear());
    // S36a1c: every app sends the environment: the pre-exposure is the EV100's exposure
    EXPECT_FLOAT_EQ(app.scene()->preExposure(), app.scene()->frameExposure());

    std::vector<std::string> scopes;
    const Json report = app.gpuProfile().report();
    for (const auto& sc : report["scopes"].array_items()) scopes.push_back(sc["scope"].string_value());
    const std::vector<std::string> expected = {"gpu",
                                               "gpu/transfer",
                                               "gpu/environment",
                                               "gpu/cull",
                                               "gpu/lights",
                                               "gpu/shadow",
                                               "gpu/shadow.cascades",
                                               "gpu/shadow.horizon",
                                               "gpu/shadow.atlas",
                                               "gpu/depth",
                                               "gpu/depth.terrain",
                                               "gpu/scene",
                                               "gpu/scene.terrain",
                                               "gpu/scene.instances",
                                               "gpu/scene.grass",
                                               "gpu/scene.sky",
                                               "gpu/scene.clouds",
                                               "gpu/scene.debug",
                                               "gpu/resolve",
                                               "gpu/lighting.preview",
                                               "gpu/post",
                                               "gpu/imgui"};
    EXPECT_EQ(scopes, expected);
    // A completed frame's sample by name (the benchmark's gpu_frame_ms)
    std::vector<std::pair<std::string, double>> sample;
    bool found = false;
    for (uint64_t f = app.scene()->frames(); f > 0 && !found; --f) found = app.gpuFrameSample(f, sample);
    ASSERT_TRUE(found);
    std::map<std::string, double> byName(sample.begin(), sample.end());
    EXPECT_EQ(byName.size(), expected.size());
    EXPECT_GE(byName["gpu"], byName["gpu/scene"]);
    EXPECT_FALSE(app.gpuFrameSample(app.scene()->frames() + 100, sample));
}

TEST_F(EngineAppRender, MeasureFramesWhilePaused)
{
    EngineApp app(appSetup(true));
    app.world().setPaused(true);
    app.runFrames(5, 1.0 / 60);
    const Json r = toolJson(callApp(app, "measure_performance", obj(R"({"frames": 6})"), true));
    EXPECT_EQ(r["frame"]["samples"].number_value(), 6) << r.dump();
    EXPECT_EQ(r["steps"]["ran"].number_value(), 0) << r.dump();
    EXPECT_FALSE(r["stopped_early"].bool_value());
    // The GPU samples belong to the rendered frames of the range (lavapipe has timestamps): measured from a known frame
    // (between frames: the next loop is the first), the endpoints lie in that frame's successors
    const uint64_t start = app.scene()->frames();
    const Json d         = measureDirect(app, 6);
    EXPECT_EQ(d["frame"]["samples"].number_value(), 6) << d.dump();
    if (d["gpu"]["samples"].number_value() > 0)
    {
        EXPECT_LE(d["gpu"]["samples"].number_value(), 6) << d.dump();
        EXPECT_GE(d["gpu"]["frames"][0].number_value(), double(start + 1)) << start << d.dump();
        EXPECT_LE(d["gpu"]["frames"][0].number_value(), d["gpu"]["frames"][1].number_value()) << d.dump();
        EXPECT_LE(d["gpu"]["frames"][1].number_value(), double(start + 6)) << start << d.dump();
    }
    else
        EXPECT_FALSE(d["gpu"]["frames"].is_array()) << d.dump();  // no timestamps: no endpoints
    // Running: the step profile has the steps of those frames
    app.world().setPaused(false);
    const Json running = toolJson(callApp(app, "measure_performance", obj(R"({"frames": 4})"), true));
    EXPECT_EQ(running["frame"]["samples"].number_value(), 4) << running.dump();
    EXPECT_GE(running["steps"]["ran"].number_value(), 3) << running.dump();
}

// The frames measurement started inside a frame (a frame callback) and between frames measures the same number of loops
// and exactly their steps (one step per loop at dt = 1/60); a reset or a load during it answers with an error
TEST_F(EngineAppRender, MeasureFramesStartAndInvalidation)
{
    EngineApp app(appSetup(true));
    app.runFrames(3, 1.0 / 60);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    auto measure        = [&](bool insideFrame)
    {
        Json result;
        auto respond = [&](const McpServer::ToolResult& r)
        {
            std::string err;
            result = Json::parse(r.content.empty() ? "" : r.content[0]["text"].string_value(), err);
        };
        if (insideFrame)
            app.onFrameRendered(
                [&]()
                {
                    app.measureFrames(3, deadline, respond);
                    return true;
                });
        else
            app.measureFrames(3, deadline, respond);
        for (int i = 0; i < 30 && result.is_null(); ++i) app.runFrames(1, 1.0 / 60);
        return result;
    };
    for (bool inside : {false, true})
    {
        const Json r = measure(inside);
        EXPECT_EQ(r["frame"]["samples"].number_value(), 3) << inside << r.dump();
        EXPECT_EQ(r["steps"]["ran"].number_value(), 3) << inside << r.dump();
    }
    // Invalidation: a profile reset and a load during the measurement
    for (int which = 0; which < 2; ++which)
    {
        bool error = false;
        app.measureFrames(50, deadline, [&](const McpServer::ToolResult& r) { error = r.isError; });
        app.runFrames(2, 1.0 / 60);
        if (which == 0)
            ASSERT_TRUE(app.api().call("reset_profile", Json::object{}).ok);
        else
            ASSERT_TRUE(app.world().tryLoad(app.world().save()).ok());
        app.runFrames(2, 1.0 / 60);
        EXPECT_TRUE(error) << which;
    }
}

// Regression (CodeRabbit, S14): the GPU log held 600 samples, so the first sample after a 600-frame range evicted the
// range's first one (599 reported)
TEST_F(EngineAppRender, MeasureSixHundredFrames)
{
    EngineApp app(appSetup(true));
    app.world().setPaused(true);
    app.runFrames(3, 1.0 / 60);
    const uint64_t start = app.scene()->frames();
    const Json r         = measureDirect(app, 600);
    EXPECT_EQ(r["frame"]["samples"].number_value(), 600) << r.dump();
    if (r["gpu"]["samples"].number_value() > 0)
    {
        EXPECT_EQ(r["gpu"]["samples"].number_value(), 600) << r.dump();
        EXPECT_EQ(r["gpu"]["frames"][0].number_value(), double(start + 1)) << r.dump();
        EXPECT_EQ(r["gpu"]["frames"][1].number_value(), double(start + 600)) << r.dump();
    }
}

// Regression (CodeRabbit, S14): the CPU and step reports were taken at the next poll's listener, after that poll's
// tool calls (a step there was counted); and a deadline that ended the wait for the GPU samples did not say
// stopped_early
TEST_F(EngineAppRender, MeasureFramesEndsWithItsLastLoop)
{
    EngineApp app(appSetup(true));
    app.world().setPaused(true);
    app.runFrames(3, 1.0 / 60);
    Json result;
    auto respond = [&](const McpServer::ToolResult& r)
    {
        std::string err;
        result = Json::parse(r.content.empty() ? "" : r.content[0]["text"].string_value(), err);
    };
    // Generous for the 3 frames and the step below (a few ms each on a software renderer; 400 ms on a loaded machine)
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(400);
    app.measureFrames(3, deadline, respond);
    app.runFrames(3, 1.0 / 60);
    ASSERT_TRUE(result.is_null());  // waits for the GPU samples of its frames
    // A step by a tool call after the measured loops (as in the next loop's poll)
    ASSERT_TRUE(app.api().call("step", Json::object{{"n", 1}}).ok);
    std::this_thread::sleep_until(deadline + std::chrono::milliseconds(10));
    app.mcp().poll();  // the deadline ends the wait (no frame rendered meanwhile)
    ASSERT_TRUE(result.is_object());
    EXPECT_EQ(result["frame"]["samples"].number_value(), 3) << result.dump();
    EXPECT_EQ(result["steps"]["ran"].number_value(), 0) << result.dump();
    EXPECT_TRUE(result["stopped_early"].bool_value()) << result.dump();
    // A normal completion has no stopped_early
    const Json normal = measureDirect(app, 3);
    EXPECT_EQ(normal["frame"]["samples"].number_value(), 3) << normal.dump();
    EXPECT_FALSE(normal["stopped_early"].is_bool()) << normal.dump();
}

// ---------------------------------------------------------------------------------------------------------------------
// S16b: --coop (a hidden host world and the rendered client world over a loopback pair)

TEST(AppCommandLine, Coop)
{
    std::string error;
    auto parse = [&](std::initializer_list<const char*> args, AppOptions& o, EngineAppSetup& s)
    {
        auto a = argv(args);
        return parseCommandLine(int(a.size()), a.data(), {"--demo"}, s, o, error);
    };
    {
        EngineAppSetup s;
        AppOptions o;
        ASSERT_TRUE(parse({"--coop"}, o, s)) << error;
        EXPECT_TRUE(o.coop);
        EXPECT_EQ(o.coopDelay, 2);
    }
    {
        EngineAppSetup s;
        AppOptions o;
        ASSERT_TRUE(parse({"--coop", "0", "--headless"}, o, s)) << error;
        EXPECT_EQ(o.coopDelay, 0);
        EXPECT_TRUE(s.headless);
    }
    {
        // An argument that is not an integer is not consumed
        EngineAppSetup s;
        AppOptions o;
        ASSERT_TRUE(parse({"--coop", "--demo"}, o, s)) << error;
        EXPECT_EQ(o.coopDelay, 2);
        EXPECT_TRUE(o.has("--demo"));
        ASSERT_TRUE(parse({"--coop", "30"}, o, s)) << error;
        EXPECT_EQ(o.coopDelay, 30);
    }
    {
        // checkOptions sees the parsed options (a game clamps the delay for its --demo) and may reject them
        EngineAppSetup s;
        s.checkOptions = [](AppOptions& o, std::string& e)
        {
            if (o.coopDelay > 8 && o.has("--demo")) o.coopDelay = 8;
            if (o.coopDelay == 7) e = "seven";
            return o.coopDelay != 7;
        };
        AppOptions o;
        ASSERT_TRUE(parse({"--coop", "20", "--demo"}, o, s)) << error;
        EXPECT_EQ(o.coopDelay, 8);
        AppOptions p;
        EXPECT_FALSE(parse({"--coop", "7"}, p, s));
        EXPECT_EQ(error, "seven");
    }
    for (auto bad : {argv({"--coop", "31"}), argv({"--coop", "-1"}), argv({"--coop", "--scenarios"}),
                     argv({"--scenarios", "x", "--coop"}), argv({"--coop", "--benchmark", "flyover"})})
    {
        EngineAppSetup u;
        u.benchmarks = {"flyover"};
        AppOptions q;
        std::string args;
        for (size_t i = 1; i < bad.size(); ++i) args += std::string(i > 1 ? " " : "") + bad[i];
        EXPECT_FALSE(parseCommandLine(int(bad.size()), bad.data(), {}, u, q, error)) << args;
    }
}

namespace
{
struct CoopLog
{
    std::vector<uint8_t> joined, left;
    std::string order;  // "H" / "C" per host / client step
};
CoopLog g_coop;

// The test game's join policy: a player entity owned by the joined peer
void joinPolicy(EngineApp&, World& host, uint8_t peer)
{
    g_coop.joined.push_back(peer);
    const Entity p = host.create();
    Player pl;
    pl.peer        = peer;
    pl.characterId = 100 + peer;
    host.add<Player>(p, pl);
    host.add<PlayerInput>(p);
    host.setOwner(p, peer);
}

EngineAppSetup coopSetup()
{
    g_coop           = CoopLog();
    EngineAppSetup s = scenarioSetup();
    s.onPeerJoined   = joinPolicy;
    s.onPeerLeft     = [](EngineApp&, World&, uint8_t peer) { g_coop.left.push_back(peer); };
    return s;
}
AppOptions coopOptions(int delay)
{
    AppOptions o;
    o.coop      = true;
    o.coopDelay = delay;
    return o;
}
}  // namespace

TEST(EngineAppCoop, InterleavedStepsAndTheClientsPlayer)
{
    EngineApp app(coopSetup(), coopOptions(2));
    ASSERT_NE(app.hostWorld(), nullptr);
    World& host   = *app.hostWorld();
    World& client = app.world();
    EXPECT_TRUE(app.coop());
    EXPECT_EQ(&app.startWorld(), &host);
    EXPECT_TRUE(host.isHost());
    EXPECT_FALSE(client.isHost());
    EXPECT_EQ(client.localPeer(), 1);
    host.addStepListener(
        [&]()
        {
            g_coop.order += 'H';
            return true;
        });
    std::vector<uint64_t> applied;  // the client's applied snapshot sequence after each of its steps
    client.addStepListener(
        [&]()
        {
            g_coop.order += 'C';
            applied.push_back(client.replication().snapshotSequence());
            return true;
        });

    // S26: the join is seen in the host step that admits the client's JoinRequest (one round trip at delay 2: host
    // step 4); the player arrives 1 + delay steps later and becomes the local player
    app.runFrames(1, 1.0 / 60);
    EXPECT_TRUE(g_coop.joined.empty());
    int frames = 1;
    for (; frames < 16 && g_coop.joined.empty(); ++frames) app.runFrames(1, 1.0 / 60);
    ASSERT_EQ(g_coop.joined, std::vector<uint8_t>{1});
    EXPECT_EQ(host.stepIndex(), 4u);
    EXPECT_FALSE(app.localPlayer().valid());
    for (int k = 0; k < 16 && !app.localPlayer().valid(); ++k) app.runFrames(1, 1.0 / 60);
    const Entity mine = app.localPlayer();
    ASSERT_TRUE(mine.valid());
    EXPECT_EQ(client.read<Player>(mine).peer, 1);
    EXPECT_EQ(client.read<Player>(mine).characterId, 101u);
    EXPECT_EQ(host.stepIndex(), client.stepIndex());

    // A low frame rate: several steps per frame, interleaved host / client (the client never gets two host steps in
    // one round); one accumulator, so both worlds have the same step count and the client's alpha follows the host
    g_coop.order.clear();
    applied.clear();
    app.runFrames(5, 3.5 / 60);
    EXPECT_EQ(host.stepIndex(), client.stepIndex());
    ASSERT_GE(g_coop.order.size(), 30u);
    for (size_t i = 0; i < g_coop.order.size(); ++i) EXPECT_EQ(g_coop.order[i], i % 2 ? 'C' : 'H') << i;
    // The client applies every host step's snapshot (none skipped by frame batching)
    ASSERT_GE(applied.size(), 15u);
    for (size_t i = 1; i < applied.size(); ++i) EXPECT_EQ(applied[i], applied[i - 1] + 1) << i;
    EXPECT_DOUBLE_EQ(client.alpha(), host.alpha());
    // The input reaches the client's player (owner-authoritative) and the host through its owner state
    app.setVirtualInput(VirtualInput{{{"move_x", 1.0}}, {}, {}}, 0);
    app.runFrames(8, 1.0 / 60);
    EXPECT_GT(client.read<PlayerInput>(mine).move.norm(), 0.5);
    Entity onHost;
    host.viewAll<const Player>().each(
        [&](Entity e, const Player& p)
        {
            if (p.peer == 1) onHost = e;
        });
    ASSERT_TRUE(onHost.valid());
    EXPECT_GT(host.read<PlayerInput>(onHost).move.norm(), 0.5);
    app.setVirtualInput(VirtualInput(), 1);

    // app_status reports both worlds
    {
        const Json st = toolJson(callApp(app, "app_status", Json::object{}));
        EXPECT_EQ(st["coop"]["host_step"].number_value(), double(host.stepIndex())) << st.dump();
        EXPECT_EQ(st["coop"]["client_step"].number_value(), double(client.stepIndex()));
        EXPECT_EQ(st["coop"]["delay_steps"].number_value(), 2);
        EXPECT_EQ(st["local_player"]["index"].number_value(), double(mine.index));
    }

    // A Resync (a host load): the replicas are destroyed and arrive again; the local player follows the new replica
    ASSERT_TRUE(host.tryLoad(host.save()).ok());
    app.runFrames(8, 1.0 / 60);
    const Entity again = app.localPlayer();
    ASSERT_TRUE(again.valid());
    EXPECT_NE(again, mine);
    EXPECT_FALSE(client.alive(mine));
    EXPECT_EQ(client.read<Player>(again).peer, 1);
    // The join policy ran once
    EXPECT_EQ(g_coop.joined.size(), 1u);
    EXPECT_TRUE(g_coop.left.empty());
}

#ifdef SAIGA_ASSERTS
TEST(EngineAppCoopDeathTest, SetLocalPlayerOfAnotherPeer)
{
    auto other = []()
    {
        EngineApp app(coopSetup(), coopOptions(0));
        World& client  = app.world();
        const Entity e = client.create();
        Player p;
        p.peer = 0;
        client.add<Player>(e, p);
        app.setLocalPlayer(e);
    };
    EXPECT_DEATH(other(), "the client's own player");
}
#endif

TEST(EngineAppCoop, ToolsActOnBothWorlds)
{
    EngineApp app(coopSetup(), coopOptions(2));
    World& host   = *app.hostWorld();
    World& client = app.world();
    app.runFrames(10, 1.0 / 60);

    // pause, step and time_scale on both worlds
    EXPECT_FALSE(isError(callApp(app, "pause", Json::object{})));
    EXPECT_TRUE(host.isPaused());
    EXPECT_TRUE(client.isPaused());
    app.runFrames(5, 1.0 / 60);
    EXPECT_EQ(host.stepIndex(), 10u);
    const Json st = toolJson(callApp(app, "step", obj(R"({"n": 3})")));
    EXPECT_EQ(st["step"].number_value(), 13) << st.dump();
    EXPECT_EQ(host.stepIndex(), 13u);
    EXPECT_EQ(client.stepIndex(), 13u);
    // measure_performance(steps) while paused: both step, the host's profile under host
    const Json m = toolJson(callApp(app, "measure_performance", obj(R"({"steps": 4})")));
    EXPECT_EQ(m["samples"].number_value(), 4) << m.dump();
    EXPECT_EQ(m["host"]["samples"].number_value(), 4) << m.dump();
    EXPECT_EQ(host.stepIndex(), 17u);
    EXPECT_EQ(client.stepIndex(), 17u);
    EXPECT_FALSE(isError(callApp(app, "time_scale", obj(R"({"scale": 0.5})"))));
    EXPECT_EQ(host.timeScale(), 0.5);
    EXPECT_EQ(client.timeScale(), 0.5);
    EXPECT_FALSE(isError(callApp(app, "resume", Json::object{})));
    EXPECT_FALSE(host.isPaused());
    EXPECT_FALSE(client.isPaused());
    // The time scale applies to the shared accumulator: 10 frames of one step at 0.5 are 5 steps
    app.runFrames(10, 1.0 / 60);
    EXPECT_EQ(host.stepIndex(), 22u);
    EXPECT_EQ(client.stepIndex(), 22u);
    callApp(app, "time_scale", obj(R"({"scale": 1})"));

    // world_status and get_profile have the host's numbers
    const Json ws = toolJson(callApp(app, "world_status", Json::object{}));
    EXPECT_EQ(ws["host"]["step"].number_value(), double(host.stepIndex())) << ws.dump();
    EXPECT_TRUE(ws["host"]["hash"].is_string());
    const Json gp = toolJson(callApp(app, "get_profile", obj(R"({"steps": 5})")));
    EXPECT_EQ(gp["host"]["samples"].number_value(), 5) << gp.dump();

    // The snapshot tools are refused
    for (const char* tool : {"save_snapshot", "load_snapshot", "export_snapshot", "import_snapshot"})
    {
        Json args    = std::string(tool) == "export_snapshot" || std::string(tool) == "import_snapshot"
                           ? obj(R"({"name": "a", "file": "f"})")
                           : obj(R"({"name": "a"})");
        const Json r = callApp(app, tool, args);
        EXPECT_TRUE(isError(r)) << tool;
        EXPECT_NE(errorText(r).find("--coop"), std::string::npos) << tool << ": " << errorText(r);
    }
    EXPECT_TRUE(isError(callApp(app, "list_snapshots", Json::object{})));

    // run_scenario runs in its own worlds (the host's data), also the host + client variants; keep and load refused
    Json r = toolJson(callApp(app, "run_scenario", obj(R"({"name": "walks", "variants": ["Plain", "HostClient"]})")));
    ASSERT_EQ(r["results"].array_items().size(), 2u) << r.dump();
    EXPECT_TRUE(r["results"][0]["passed"].bool_value()) << r.dump();
    EXPECT_TRUE(r["results"][1]["passed"].bool_value()) << r.dump();
    EXPECT_EQ(r["results"][1]["variant"].string_value(), "HostClient");
    const Json kept = callApp(app, "run_scenario", obj(R"({"name": "walks", "keep": true})"));
    EXPECT_TRUE(isError(kept));
    EXPECT_NE(errorText(kept).find("--coop"), std::string::npos) << errorText(kept);
    // reload_data reloads the shared data and reconfigures through the host (the client resyncs)
    const Json rd = callApp(app, "reload_data", Json::object{});
    EXPECT_FALSE(isError(rd)) << errorText(rd);
    EXPECT_EQ(host.loadCount(), 1u);
}

// Regression (local review, S16b): in co-op reset_profile left the host's step profile (reported by get_profile and
// measure_performance) untouched; reload_data replaced the client's state through the host's load, but the client's
// cursors and world_generation did not show a reload; a running measure_performance did not notice the host's load
TEST(EngineAppCoop, ResetProfileAndReloadReachTheHost)
{
    EngineApp app(coopSetup(), coopOptions(2));
    World& host = *app.hostWorld();
    app.runFrames(10, 1.0 / 60);
    EXPECT_FALSE(isError(callApp(app, "pause", Json::object{})));

    // reset_profile: the host's profile is empty afterwards
    ASSERT_GT(host.profiler().report(0)["samples"].number_value(), 0);
    EXPECT_FALSE(isError(callApp(app, "reset_profile", Json::object{})));
    const Json gp = toolJson(callApp(app, "get_profile", Json::object{}));
    EXPECT_EQ(gp["samples"].number_value(), 0) << gp.dump();
    EXPECT_EQ(gp["host"]["samples"].number_value(), 0) << gp.dump();

    // reload_data: a new world generation, a cursor from before answers reloaded
    const Json before = toolJson(callApp(app, "changes_since", obj(R"({"cursor": "now"})")));
    ASSERT_TRUE(before["cursor"].is_string()) << before.dump();
    const double generation = toolJson(callApp(app, "world_status", Json::object{}))["world_generation"].number_value();
    const Json rd           = toolJson(callApp(app, "reload_data", Json::object{}));
    EXPECT_EQ(rd["world_generation"].number_value(), generation + 1) << rd.dump();
    const Json after = toolJson(callApp(app, "changes_since", Json::object{{"cursor", before["cursor"]}}));
    EXPECT_TRUE(after["reloaded"].bool_value()) << after.dump();
    EXPECT_FALSE(isError(callApp(app, "resume", Json::object{})));

    // A host load during a running measure_performance(steps) ends it with an error
    int frames = 0;
    Poller poller(app.mcp(),
                  [&]()
                  {
                      app.runFrames(1, 1.0 / 60);
                      if (++frames == 5)
                      {
                          ASSERT_TRUE(host.tryLoad(host.save()).ok());
                      }
                  });
    const Json m = callTool(app.mcp(), "measure_performance", obj(R"({"steps": 60})"));
    EXPECT_TRUE(isError(m)) << m.dump();
    EXPECT_NE(errorText(m).find("loaded"), std::string::npos) << errorText(m);
}

// World::stepsForAdvance takes the same steps out of the accumulator as advance runs (time scale, the cap, the dropped
// backlog, pause), without stepping
TEST(EngineAppCoop, StepsForAdvanceMatchesAdvance)
{
    World a{WorldSetup()}, b{WorldSetup()};
    a.setTimeScale(0.75);
    b.setTimeScale(0.75);
    uint64_t taken = 0;
    for (double dt : {1.0 / 60, 1.0 / 30, 0.5, 0.0, 1.0 / 144, 0.2, std::numeric_limits<double>::infinity(), 1.0 / 60})
    {
        a.advance(dt);
        const int n = b.stepsForAdvance(dt);
        EXPECT_EQ(b.stepIndex(), taken) << "no step runs";
        b.step(n);
        taken += uint64_t(n);
        EXPECT_EQ(a.stepIndex(), b.stepIndex()) << dt;
        EXPECT_EQ(a.droppedSteps(), b.droppedSteps()) << dt;
        EXPECT_DOUBLE_EQ(a.alpha(), b.alpha()) << dt;
    }
    b.setPaused(true);
    EXPECT_EQ(b.stepsForAdvance(1.0), 0);
}

// Regression (local review, S16b): stepsForAdvance lacked advance's guards (inside a step, inside a step listener)
#ifdef SAIGA_ASSERTS
void stepsForAdvanceInSystem(World& w, double)
{
    w.stepsForAdvance(1.0);
}

TEST(EngineAppCoopDeathTest, StepsForAdvanceGuards)
{
    auto inListener = []()
    {
        World w{WorldSetup()};
        w.addStepListener(
            [&]()
            {
                w.stepsForAdvance(1.0);
                return true;
            });
        w.step();
    };
    EXPECT_DEATH(inListener(), "stepsForAdvance inside a step listener");
    auto inStep = []()
    {
        WorldSetup s;
        s.systems = [](SystemList& l) { l.add(Phase::Gameplay, "advances", stepsForAdvanceInSystem); };
        World w(s);
        w.step();
    };
    EXPECT_DEATH(inStep(), "stepsForAdvance inside a step");
}
#endif

// The leave path at world level (demo-independent): destroying the client world and then its transport end makes the
// peer appear in the host's peersLeft(); destroyOwnedBy removes its entities and keeps the host's
TEST(EngineCoopLeave, PeersLeftAndDestroyOwnedBy)
{
    auto transports = LoopbackTransport::createNetwork(2, LoopbackOptions());
    WorldSetup hs;
    hs.transport = transports[0].get();
    World host(hs);
    WorldSetup cs;
    cs.transport = transports[1].get();
    cs.role      = NetRole::Client;
    cs.localPeer = 1;
    auto client  = std::make_unique<World>(cs);
    host.step();
    client->step();
    // S26: admitted in the host step that handles the JoinRequest
    host.step();
    ASSERT_EQ(host.peersJoined(), std::vector<uint8_t>{1});
    client->step();
    // The peer's player with a child, and a host-owned vehicle
    const Entity player = host.create();
    Player pl;
    pl.peer = 1;
    host.add<Player>(player, pl);
    host.setOwner(player, 1);
    const Entity child = host.create();
    host.add<Parent>(child, Parent{player});
    const Entity vehicle = host.create();
    for (int i = 0; i < 4; ++i)
    {
        host.step();
        client->step();
    }
    client.reset();
    transports[1].reset();
    bool left = false;
    for (int i = 0; i < 4 && !left; ++i)
    {
        host.step();
        left = host.peersLeft() == std::vector<uint8_t>{1};
    }
    EXPECT_TRUE(left);
    EXPECT_EQ(destroyOwnedBy(host, 1), 1u);
    EXPECT_FALSE(host.alive(player));
    EXPECT_FALSE(host.alive(child));
    EXPECT_TRUE(host.alive(vehicle));
    EXPECT_EQ(destroyOwnedBy(host, 1), 0u);
}

// The app's leave wiring (local review): in --coop the client lives as long as the host, so a leave happens only when
// the link breaks; a bad fragment from the client (the loopback test hook) disconnects both ends, and the host's step
// listener calls EngineAppSetup::onPeerLeft once (here the leave policy destroys the peer's entities)
TEST(EngineAppCoop, OnPeerLeftAfterABrokenLink)
{
    EngineAppSetup s = coopSetup();
    s.onPeerLeft     = [](EngineApp&, World& host, uint8_t peer)
    {
        g_coop.left.push_back(peer);
        destroyOwnedBy(host, peer);
    };
    EngineApp app(s, coopOptions(2));
    World& host = *app.hostWorld();
    app.runFrames(10, 1.0 / 60);
    ASSERT_TRUE(app.localPlayer().valid());
    Entity onHost;
    host.viewAll<const Player>().each(
        [&](Entity e, const Player& p)
        {
            if (p.peer == 1) onHost = e;
        });
    ASSERT_TRUE(onHost.valid());
    EXPECT_TRUE(g_coop.left.empty());
    std::vector<uint8_t> bad(kFragmentHeader + 4, 0);
    bad[4] = 9;  // index 9 of count 0
    static_cast<LoopbackTransport*>(host.transport())->injectFragment(1, 0, Channel::Control, bad);
    app.runFrames(10, 1.0 / 60);
    EXPECT_EQ(g_coop.left, std::vector<uint8_t>{1});
    EXPECT_FALSE(host.alive(onHost));
    EXPECT_TRUE(host.connectedPeers().empty());
}

// ---------------------------------------------------------------------------------------------------------------------
// S19b: the view camera, beforeRender and the render sink

namespace
{
std::shared_ptr<const Vulkan::MeshLods> testTriangle()
{
    static std::shared_ptr<const Vulkan::MeshLods> lods = []()
    {
        auto l = std::make_shared<Vulkan::MeshLods>();
        Vulkan::MeshLod lod;
        lod.vertices.resize(3);
        lod.vertices[1].position = vec3(1, 0, 0);
        lod.vertices[2].position = vec3(0, 1, 0);
        lod.indices              = {0, 1, 2};
        l->lods.push_back(lod);
        l->sphereRadius = 1;
        return std::shared_ptr<const Vulkan::MeshLods>(l);
    }();
    return lods;
}
void triangleAssets(World& w)
{
    w.registerMesh("test/triangle", testTriangle(), w.registerMaterial("test/flat", Vulkan::MaterialData()));
}
}  // namespace

// The single camera: the game's setViewCamera (double), the tools' hold (the game's call returns false), the release
// callback; camera() is derived from it
TEST(EngineAppNoRender, ViewCameraHoldAndRelease)
{
    EngineAppSetup s   = appSetup(false);
    int released       = 0;
    s.onCameraReleased = [&](EngineApp&) { released++; };
    EngineApp app(s);
    const Quat r = cameraLookRotation(Vec3(1, 0, 0));
    EXPECT_TRUE(app.setViewCamera(0, Vec3(1e5 + 0.25, 3, -7), r, 1.0, true));
    EXPECT_EQ(app.viewCamera().position, Vec3(1e5 + 0.25, 3, -7));
    EXPECT_EQ(app.viewCamera().fovy, 1.0);
    EXPECT_TRUE(app.viewCamera().snap);
    EXPECT_NEAR(app.camera().getPosition()(0), 1e5, 1);
    // The derived camera looks along +x
    const vec3 forward = -make_vec3(app.camera().model.col(2)).normalized();
    EXPECT_NEAR(forward(0), 1, 1e-5);
    Json c = toolJson(callApp(app, "app_status", Json::object{}))["camera"];
    EXPECT_NEAR(c["yaw_deg"].number_value(), 90, 1e-3) << c.dump();
    // set_camera holds the camera: the game's calls are ignored (no branch in the game)
    callApp(app, "set_camera", obj(R"({"position": [999999.37, 5, 20], "yaw_deg": 0})"));
    EXPECT_TRUE(app.cameraHeldByTool());
    EXPECT_FALSE(app.setViewCamera(0, Vec3::Zero(), Quat::Identity(), 1.5));
    // Double: the tool's position is kept exactly also near the 1e6 m limit (the float camera had 6 cm steps there)
    EXPECT_EQ(app.viewCamera().position, Vec3(999999.37, 5, 20));
    EXPECT_EQ(app.viewCamera().fovy, 1.0);  // the tools keep the field of view
    // A tool write without a hold (game_set_camera, the walk mode) is not blocked and does not hold
    app.setViewCameraByTool(0, Vec3(1, 2, 3), Quat::Identity(), 0.5, false);
    EXPECT_EQ(app.viewCamera().position, Vec3(1, 2, 3));
    EXPECT_TRUE(app.cameraHeldByTool());
    // The release edge calls the game's camera reset once
    callApp(app, "set_camera", obj(R"({"release": true})"));
    callApp(app, "set_camera", obj(R"({"release": true})"));
    EXPECT_EQ(released, 1);
    EXPECT_TRUE(app.setViewCamera(0, Vec3(4, 5, 6), Quat::Identity(), 1.2));
    EXPECT_EQ(app.viewCamera().position, Vec3(4, 5, 6));
    // move_camera on the double camera
    c = toolJson(callApp(app, "move_camera", obj(R"({"forward": 4})")));
    EXPECT_NEAR(c["position"][2].number_value(), 2, 1e-9) << c.dump();  // identity looks along -z
    callApp(app, "set_camera", obj(R"({"release": true})"));
    EXPECT_EQ(released, 2);
}

// beforeRender runs once per rendered frame after the extract's catch-up: an entity spawned while paused is tracked
// there; the world's sink is the app's VulkanRenderSink, the World's meshes reach the scene
TEST_F(EngineAppRender, BeforeRenderAfterCatchUp)
{
    EngineAppSetup s     = appSetup(true);
    s.world.renderAssets = triangleAssets;
    Entity spawned;
    int calls = 0, tracked = 0;
    s.renderHooks.beforeRender = [&](EngineApp& a, double)
    {
        calls++;
        WorldPose p;
        if (spawned.valid() && a.world().extract().interpolatedPose(spawned, 0, p)) tracked++;
    };
    EngineApp app(s);
    ASSERT_NE(app.vulkanSink(), nullptr);
    EXPECT_EQ(app.world().renderSink(), app.vulkanSink());
    EXPECT_EQ(app.scene()->findMesh("test/triangle"), app.world().findMesh("test/triangle"));
    app.runFrames(2, 1.0 / 60);
    app.world().setPaused(true);
    spawned = app.world().create();
    app.world().add<RenderInstance>(spawned, RenderInstance("test/triangle"));
    app.runFrames(3, 1.0 / 60);
    EXPECT_EQ(calls, 5);
    EXPECT_EQ(tracked, 3);
    EXPECT_EQ(app.world().extract().instanceCount(), 1u);
}

// --coop: only the client world (the rendered one) has a sink; the host's entities reach it through replication
TEST_F(EngineAppRender, CoopClientHasTheSink)
{
    EngineAppSetup s       = coopSetup();
    const EngineAppSetup r = appSetup(true);
    s.noRender             = false;
    s.validation           = r.validation;
    s.render               = r.render;
    s.world.renderAssets   = triangleAssets;
    EngineApp app(s, coopOptions(1));
    ASSERT_NE(app.hostWorld(), nullptr);
    EXPECT_EQ(app.hostWorld()->renderSink(), nullptr);
    EXPECT_EQ(app.world().renderSink(), app.vulkanSink());
    const Entity e = app.hostWorld()->create();
    app.hostWorld()->add<RenderInstance>(e, RenderInstance("test/triangle"));
    app.runFrames(12, 1.0 / 60);
    EXPECT_EQ(app.world().extract().instanceCount(), 1u);
    EXPECT_EQ(app.hostWorld()->extract().trackedCount(), 0u);
}

// ---------------------------------------------------------------------------------------------------------------------
// The debug UI (S19c)

// set_ui_tab selects tabs by name (also without a device: the selection is kept); app_status names the tab
TEST(EngineAppNoRender, SetUiTab)
{
    EngineApp app(appSetup(false));
    Poller poller(app.mcp());
    Json r = toolJson(callTool(app.mcp(), "set_ui_tab", Json::object{}));
    EXPECT_EQ(r["tab"], Json("Game"));
    ASSERT_EQ(r["tabs"].array_items().size(), 5u);
    EXPECT_EQ(r["tabs"][1]["key"], Json("F2"));
    EXPECT_EQ(r["tabs"][4]["key"], Json("F5"));
    EXPECT_FALSE(r["visible"].bool_value());
    r = toolJson(callTool(app.mcp(), "set_ui_tab", obj(R"({"tab": "console"})")));
    EXPECT_EQ(r["tab"], Json("Console"));
    EXPECT_TRUE(app.debugUi().isSelected(DebugTab::Console));
    EXPECT_EQ(toolJson(callTool(app.mcp(), "app_status", Json::object{}))["ui_tab"], Json("Console"));
    // S36: the Lighting tab (F5)
    r = toolJson(callTool(app.mcp(), "set_ui_tab", obj(R"({"tab": "lighting"})")));
    EXPECT_EQ(r["tab"], Json("Lighting"));
    EXPECT_TRUE(app.debugUi().isSelected(DebugTab::Lighting));
    Json bad = callTool(app.mcp(), "set_ui_tab", obj(R"({"tab": "nope"})"));
    EXPECT_TRUE(bad["result"]["isError"].bool_value());
    EXPECT_NE(bad["result"]["content"][0]["text"].string_value().find("Game, Performance, Engine, Console, Lighting"),
              std::string::npos);
    EXPECT_TRUE(callTool(app.mcp(), "set_ui_tab", obj(R"({"tab": 3})"))["result"]["isError"].bool_value());
    EXPECT_TRUE(callTool(app.mcp(), "set_ui_tab", obj(R"({"page": "game"})"))["result"]["isError"].bool_value());
    // A game tab
    app.debugUi().addTab("AI", {});
    EXPECT_EQ(toolJson(callTool(app.mcp(), "set_ui_tab", obj(R"({"tab": "ai"})")))["tab"], Json("AI"));
}

TEST(EngineAppNoRender, SetUiTabWithTheUiOff)
{
    EngineAppSetup s = appSetup(false);
    s.debugUi        = false;
    EngineApp app(s);
    Poller poller(app.mcp());
    EXPECT_TRUE(callTool(app.mcp(), "set_ui_tab", Json::object{})["result"]["isError"].bool_value());
    EXPECT_TRUE(toolJson(callTool(app.mcp(), "app_status", Json::object{}))["ui_tab"].is_null());
}

// queueBetweenSteps runs after the poll and before the frame's steps (GUI actions never run inside a frame)
TEST(EngineAppNoRender, ActionsRunBetweenSteps)
{
    EngineApp app(appSetup(false));
    app.runFrames(3, 1.0 / 60);
    const uint64_t before = app.world().stepIndex();
    uint64_t seen         = 0;
    app.queueBetweenSteps(
        [&]()
        {
            seen = app.world().stepIndex();
            app.world().setPaused(true);
        });
    // Not yet: only in the next frame
    EXPECT_FALSE(app.world().isPaused());
    app.runFrames(1, 1.0 / 60);
    EXPECT_EQ(seen, before);
    // Paused before the frame's advance: the frame ran no step
    EXPECT_EQ(app.world().stepIndex(), before);
}

// Every MCP tool call is logged as an "mcp" line (agents and the console); the logs tool leaves them out
TEST(EngineAppNoRender, McpCallsInTheLog)
{
    EngineApp app(appSetup(false));
    const uint64_t before = engineLog().lastSequence();
    {
        Poller poller(app.mcp(), [&]() {});
        callTool(app.mcp(), "quit", Json::object{});
        // The record of the answer is delivered by the next poll
        for (int i = 0; i < 1000; ++i)
        {
            bool found = false;
            for (const auto& l : engineLog().since(before)) found = found || l.category == "mcp";
            if (found) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    std::vector<std::string> lines;
    for (const auto& l : engineLog().since(before))
        if (l.category == "mcp") lines.push_back(l.message);
    ASSERT_EQ(lines.size(), 1u);
    EXPECT_EQ(lines[0].rfind("quit -> ok", 0), 0u) << lines[0];
    // The console's calls are marked
    app.debugUi().console().submit("world_status");
    app.mcp().poll();
    bool console = false;
    for (const auto& l : engineLog().since(before))
        console = console || (l.category == "mcp" && l.message.rfind("(console) world_status -> ok", 0) == 0);
    EXPECT_TRUE(console);
    EXPECT_EQ(app.debugUi().console().entries().back().kind, DebugConsole::Entry::Kind::Output);
}

// Each tab draws in a headless frame with ImGui (no validation error); a click on a tab button selects it
TEST_F(EngineAppRender, DebugUiTabsRender)
{
    EngineAppSetup s = appSetup(true);
    s.render.imgui   = true;
    s.width          = 800;
    s.height         = 600;
    EngineApp app(s);
    app.debugUi().addSection(DebugTab::Engine, "Game section", []() { ImGui::TextUnformatted("game"); });
    for (int tab = 0; tab < app.debugUi().tabCount(); ++tab)
    {
        app.debugUi().select(tab);
        app.runFrames(3, 1.0 / 60);
    }
    ASSERT_TRUE(ImGui::GetCurrentContext());
    // The tab buttons are on the bar (y in [0, barHeight)); "F3 Engine" is the third. One input event per frame
    app.debugUi().select(DebugTab::Game);
    app.runFrames(2, 1.0 / 60);
    const float y = app.debugUi().barHeight() / 2;
    float x       = 6;
    for (int i = 0; i < 2; ++i)
        x += ImGui::CalcTextSize(("F" + std::to_string(i + 1) + " " + app.debugUi().tabName(i)).c_str()).x + 2 * 6 + 2;
    x += 10;
    ImGuiIO& io = ImGui::GetIO();
    io.AddMousePosEvent(x, y);
    app.runFrames(1, 1.0 / 60);
    io.AddMouseButtonEvent(ImGuiMouseButton_Left, true);
    app.runFrames(1, 1.0 / 60);
    io.AddMouseButtonEvent(ImGuiMouseButton_Left, false);
    app.runFrames(1, 1.0 / 60);
    EXPECT_TRUE(app.debugUi().isSelected(DebugTab::Engine));
}

// The bar's pause button (every tab): a click pauses the world, a click on the PAUSED button resumes it
TEST_F(EngineAppRender, BarPauseButton)
{
    EngineAppSetup s = appSetup(true);
    s.render.imgui   = true;
    s.width          = 800;
    s.height         = 600;
    EngineApp app(s);
    app.debugUi().select(DebugTab::Engine);
    app.runFrames(2, 1.0 / 60);
    ImGuiIO& io = ImGui::GetIO();
    for (bool paused : {true, false})
    {
        ImVec2 at;
        ASSERT_TRUE(app.debugUi().itemCenter("bar/pause", at));
        io.AddMousePosEvent(at.x, at.y);
        app.runFrames(1, 1.0 / 60);
        io.AddMouseButtonEvent(ImGuiMouseButton_Left, true);
        app.runFrames(1, 1.0 / 60);
        io.AddMouseButtonEvent(ImGuiMouseButton_Left, false);
        // The click queues the tool between steps: applied in the next frame
        app.runFrames(2, 1.0 / 60);
        EXPECT_EQ(app.world().isPaused(), paused);
    }
}

// Regression: the panel was moved in front of all windows every frame, also in front of the popups opened inside it
// (combo lists): the Performance tab's window combo showed no list and could not be changed
TEST_F(EngineAppRender, DebugUiPopupsStayInFront)
{
    EngineAppSetup s = appSetup(true);
    s.render.imgui   = true;
    s.width          = 800;
    s.height         = 600;
    EngineApp app(s);
    bool open = true;
    app.debugUi().addSection(
        DebugTab::Performance, "Popup",
        [&]()
        {
            if (open) ImGui::OpenPopup("test popup");
            open = false;
            if (ImGui::BeginPopup("test popup"))
            {
                ImGui::TextUnformatted("in front");
                ImGui::EndPopup();
            }
        },
        true);
    app.debugUi().select(DebugTab::Performance);
    app.runFrames(4, 1.0 / 60);
    ImGuiContext& g = *ImGui::GetCurrentContext();
    ASSERT_EQ(g.OpenPopupStack.Size, 1);
    ImGuiWindow* popup = g.OpenPopupStack[0].Window;
    ImGuiWindow* panel = ImGui::FindWindowByName("##debug panel");
    ASSERT_NE(popup, nullptr);
    ASSERT_NE(panel, nullptr);
    EXPECT_GT(g.Windows.index_from_ptr(g.Windows.find(popup)), g.Windows.index_from_ptr(g.Windows.find(panel)));
}

// ---------------------------------------------------------------------------------------------------------------------
// Terrain (S20): the app's streaming, the complete points, the stream time

namespace
{
EngineAppSetup terrainAppSetup(TerrainHeightHook hook)
{
    EngineAppSetup s = appSetup(true);
    s.world.config.spaceTemplates.push_back(TerrainTestHooks::testSpaceTemplate(32, 2, hook, 0));
    s.world.generators = TerrainTestHooks::registerTestHookGenerators;
    // No cascades and no horizon map (the clay studio sun is a shadow light): these tests bound the wall clock of
    // frames after camera jumps of kilometres, where a refit with the terrain casters and the horizon map's restart
    // cost a loaded lavapipe more than the bounds (CI: a 124 ms frame against StreamTimePausesTheClock's 100 ms)
    s.render.lighting.sunShadows     = false;
    s.render.lighting.horizonShadows = false;
    return s;
}

// The terrain streams and uploads, but its passes (terrain, water, grass, sky) draw nothing: for the tests of
// streaming, readiness, deadlines and loads, whose assertions read no terrain pixels (test speed: a CDLOD frame is
// about 100k triangles, 50 ms in lavapipe's single-threaded vertex stage)
void noTerrainDraws(EngineApp& app)
{
    Vulkan::PrototypeSettings p = app.scene()->prototypeSettings();
    p.terrain                   = false;
    p.water                     = false;
    p.grass                     = false;
    p.sky                       = false;
    app.scene()->setPrototypeSettings(p);
}

// A tool call on another thread while this (the app's main) thread runs frames, until it answers
Json callWhileRunning(EngineApp& app, const std::string& name, const Json& args, double* seconds = nullptr,
                      int maxFrames = 20000)
{
    std::atomic_bool done{false};
    Json result;
    const auto t0 = std::chrono::steady_clock::now();
    std::thread caller(
        [&]()
        {
            result = callTool(app.mcp(), name, args);
            done   = true;
        });
    for (int i = 0; i < maxFrames && !done; ++i)
    {
        app.runFrames(1, 1.0 / 60);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    caller.join();
    if (seconds) *seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return result;
}
}  // namespace

// The terrain block of app_status; the first frame is a complete point
TEST_F(EngineAppRender, TerrainStatus)
{
    EngineApp app(terrainAppSetup(TerrainTestHooks::integerTestHook));
    noTerrainDraws(app);
    app.setViewCamera(0, Vec3(1000, 80, 1000), Quat::Identity(), 1.0);
    app.runFrames(3, 1.0 / 60);
    EXPECT_GT(app.lastTerrainStreamMs() + 1, 0.0);
    EXPECT_EQ(app.world().terrain().readiness(0, Vec3(1000, 80, 1000), 256), 1.0);
    std::atomic_bool done{false};
    Json status;
    std::thread caller(
        [&]()
        {
            status = toolJson(callTool(app.mcp(), "app_status", Json::object{}));
            done   = true;
        });
    while (!done) app.runFrames(1, 1.0 / 60);
    caller.join();
    const Json t = status["terrain"];
    ASSERT_TRUE(t.is_object()) << status.dump();
    EXPECT_GT(t["resident"].number_value(), 10);
    EXPECT_GE(t["pending"].number_value(), 0);
    EXPECT_EQ(t["uploads_pending_bytes"].number_value(), 0);
    EXPECT_GT(t["cpu_mb"].number_value(), 1);
    EXPECT_GT(t["gpu_mb"].number_value(), 1);
    EXPECT_EQ(t["readiness"].number_value(), 1.0);
}

// An MCP screenshot after a set_camera far away: a frame with every desired tile sent and no pending uploads
TEST_F(EngineAppRender, ScreenshotCompletesTheView)
{
    EngineApp app(terrainAppSetup(TerrainTestHooks::integerTestHook));
    noTerrainDraws(app);
    app.runFrames(2, 1.0 / 60);
    const Json c =
        callWhileRunning(app, "set_camera", obj(R"({"position": [6500, 120, 7000], "yaw_deg": 30, "pitch_deg": -30})"));
    ASSERT_FALSE(isError(c)) << c.dump();
    const Json r = callWhileRunning(app, "screenshot", Json::object{});
    ASSERT_FALSE(isError(r)) << r.dump().substr(0, 300);
    const Vec3 cam = app.viewCamera().position;
    EXPECT_EQ(app.world().terrain().readiness(0, cam, 1e9), 1.0);
    EXPECT_EQ(app.world().terrain().terrainUploadsPending(), 0u);
}

// With the workers held (armed after the first frame: that one is a complete point with a fatal timeout) and a 0.6 s
// tool timeout, a screenshot of an unstreamed area answers "terrain not complete before the deadline" within the 0.3 s
// budget (0.3 s before the timeout), exactly once, and the app keeps running
TEST_F(EngineAppRender, ScreenshotTerrainDeadline)
{
    EngineAppSetup s          = terrainAppSetup(TerrainTestHooks::latchedHook);
    s.mcpToolTimeoutSeconds   = 0.6;
    s.mcpAnswerMarginFraction = 0.5;
    EngineApp app(s);
    noTerrainDraws(app);
    app.setViewCamera(0, Vec3(500, 80, 500), Quat::Identity(), 1.0);
    app.runFrames(2, 1.0 / 60);
    TerrainTestHooks::armLatch();
    struct Open
    {
        ~Open() { TerrainTestHooks::openLatch(); }
    } open;
    ASSERT_FALSE(isError(callWhileRunning(app, "set_camera", obj(R"({"position": [7000, 120, 7500]})"))));
    double seconds = 0;
    const Json r   = callWhileRunning(app, "screenshot", Json::object{}, &seconds);
    ASSERT_TRUE(isError(r)) << r.dump().substr(0, 300);
    EXPECT_NE(errorText(r).find("terrain not complete before the deadline"), std::string::npos) << errorText(r);
    // The answer is the tool's own (a call past the 0.6 s timeout would answer with the timeout error); the wall-clock
    // bound only catches a hang (generous for a loaded machine)
    EXPECT_LT(seconds, 3.0);
    // Exactly once: no second answer (the deadline listener) and the app keeps running
    const uint64_t frames = app.renderedFrames();
    app.runFrames(5, 1.0 / 60);
    EXPECT_EQ(app.renderedFrames(), frames + 5);
    TerrainTestHooks::openLatch();
    // Streams the new view: a timeout that a loaded machine meets (the deadline was tested above)
    app.mcp().toolTimeoutSeconds = 30;
    const Json ok                = callWhileRunning(app, "screenshot", Json::object{});
    EXPECT_FALSE(isError(ok)) << ok.dump().substr(0, 300);
}

// setCompleteViewEachFrame: every frame completes; the frame that generated tiles at the origin (100 ms each) has its
// stream time, which is neither CPU work, nor a long frame, nor a backlog of the simulation clock
TEST_F(EngineAppRender, StreamTimePausesTheClock)
{
    TerrainTestHooks::setSleepAtOrigin(true);
    struct Reset
    {
        ~Reset() { TerrainTestHooks::setSleepAtOrigin(false); }
    } reset;
    // A short statistics warm-up (the frames after it count; 1 s of real time by default)
    EngineAppSetup setup          = terrainAppSetup(TerrainTestHooks::latchedHook);
    setup.statisticsWarmupSeconds = 0.1;
    EngineApp app(setup);
    noTerrainDraws(app);
    app.setCompleteViewEachFrame(true);
    // Few tiles: the frames' upload copies stay short also on a software renderer (their GPU time is a real wait)
    TerrainViewSettings view = app.world().terrain().viewSettings();
    view.residencyRange0     = 48;
    view.grassRadius         = 0;
    app.world().terrain().setViewSettings(view);
    // The first view far from the origin: the pyramid (it contains the origin) sleeps 100 ms
    app.setViewCamera(0, Vec3(5000, 80, 5000), Quat::Identity(), 1.0);
    app.frame();
    EXPECT_GE(app.lastTerrainStreamMs(), 100.0);
    // Past the statistics' warm-up
    const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
    while (std::chrono::steady_clock::now() < end)
    {
        app.frame();
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    EXPECT_LT(app.lastTerrainStreamMs(), 100.0);
    // To the origin: the tiles containing it sleep
    app.setViewCamera(0, Vec3(40, 80, 40), Quat::Identity(), 1.0);
    app.frame();
    // The samples from here on: the stream frame's interval and the later ones
    const size_t samplesBefore = app.frameStatistics().frameSamples().size();
    EXPECT_GE(app.lastTerrainStreamMs(), 100.0);
    const uint64_t steps = app.world().stepIndex();
    app.frame();
    // The next frame's dt is its own interval (a few ms): no catch-up steps for the 100+ ms (6 or more steps at 60 Hz;
    // up to 3 allow a frame of up to 50 ms on a loaded machine)
    EXPECT_LE(app.world().stepIndex() - steps, 3u);
    for (int i = 0; i < 20; ++i)
    {
        app.frame();
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    // No interval from the move on contains the 100+ ms (the first frames of a software renderer are slow for other
    // reasons; scheduling noise may make single frames "long", the stream must not)
    const auto& samples = app.frameStatistics().frameSamples();
    ASSERT_GT(samples.size(), samplesBefore + 10);
    double longest = 0;
    for (size_t i = samplesBefore; i < samples.size(); ++i) longest = std::max(longest, samples[i].interval);
    EXPECT_LT(longest, 100.0);
}

// ---------------------------------------------------------------------------------------------------------------------
// Spaces (S21): the loading screen of a view space switch, set_camera(space)

namespace
{
int32_t otherIntegerHook(uint64_t seed, int64_t x, int64_t z)
{
    return int32_t(int64_t(seed % 300) * 16 + x * 3 - z * 5 + 4096);
}

// Space 0: open world (integerTestHook, 2 m, 16 chunks), 1: the enclosed cellar, 2: the latched hook (2 m, 32 chunks
// = 8 km), 3: an unlatched integer hook (2 m, 32 chunks), 4: flat
EngineAppSetup fiveSpaceSetup()
{
    EngineAppSetup s = appSetup(true);
    SpaceTemplate a  = TerrainTestHooks::testSpaceTemplate(16, 2, TerrainTestHooks::integerTestHook, 0);
    a.name           = "overworld";
    SpaceTemplate c  = TerrainTestHooks::testSpaceTemplate(32, 2, TerrainTestHooks::latchedHook, 0);
    c.name           = "latched";
    SpaceTemplate d  = TerrainTestHooks::testSpaceTemplate(32, 2, otherIntegerHook, 0);
    d.name           = "islands";
    SpaceTemplate flat;
    flat.name                     = "flat";
    s.world.config.spaceTemplates = {a, SpaceTemplate::enclosed("cellar"), c, d, flat};
    s.world.generators            = TerrainTestHooks::registerTestHookGenerators;
    return s;
}

// Opens the latch when it goes out of scope (before the app it is declared after: the pool can join)
struct LatchGuard
{
    ~LatchGuard() { TerrainTestHooks::openLatch(); }
};
}  // namespace

// End rule 1 and 3 through S20's forwarding of the sink's pending uploads (no device)
TEST(LoadingRules, EndsOnlyWithoutPendingUploads)
{
    WorldSetup ws;
    ws.profiling = false;
    World w(ws);
    RecordingRenderSink sink;
    w.setRenderSink(&sink);
    sink.uploadsPendingForTests = 4096;
    SpaceLoading l{0, 0.5, 3};
    EXPECT_FALSE(loadEnds(l, 1.0, w.terrain().terrainUploadsPending(), 600));
    sink.uploadsPendingForTests = 0;
    EXPECT_TRUE(loadEnds(l, 1.0, w.terrain().terrainUploadsPending(), 600));
    EXPECT_FALSE(loadEnds(l, 0.99, w.terrain().terrainUploadsPending(), 600));
    l.frames = 600;
    EXPECT_TRUE(loadEnds(l, 0.5, 4096, 600));
    w.setRenderSink(nullptr);
}

// set_camera(space): validated against the view world's spaces, with the hold; no load without rendering
TEST(EngineAppNoRender, SetCameraSpace)
{
    EngineAppSetup s              = appSetup(false);
    s.world.config.spaceTemplates = {SpaceTemplate(), SpaceTemplate::enclosed("cellar")};
    EngineApp app(s);
    Json c = callApp(app, "set_camera", obj(R"({"space": 1, "position": [640, 2, 640]})"));
    ASSERT_FALSE(isError(c)) << c.dump();
    EXPECT_EQ(toolJson(c)["space"].int_value(), 1);
    EXPECT_EQ(app.viewCamera().space, 1);
    EXPECT_TRUE(app.cameraHeldByTool());
    // Without space: the current one
    c = callApp(app, "set_camera", obj(R"({"position": [641, 2, 640]})"));
    EXPECT_EQ(app.viewCamera().space, 1);
    const Json e = callApp(app, "set_camera", obj(R"({"space": 2})"));
    EXPECT_TRUE(isError(e));
    EXPECT_NE(errorText(e).find("2 spaces"), std::string::npos) << errorText(e);
    EXPECT_TRUE(isError(callApp(app, "set_camera", obj(R"({"space": 0.5})"))));
    // --no-render: no load while the view space changes
    app.runFrames(3, 1.0 / 60);
    EXPECT_EQ(app.spaceLoading(), nullptr);
    EXPECT_TRUE(toolJson(callApp(app, "app_status", Json::object{}))["space_loading"].is_null());
}

// The sequence of §12.4 step 2 on five spaces (one app): no load for the enclosed and the flat space; a load in the
// latched space that stays at 0 while the workers are held (app_status reports it, the game's hook draws it once per
// frame); a hop out through the enclosed space and a new load that ends at end rule 1 once the latch opens; a load in
// the unlatched space 3 cut off by an MCP screenshot (a complete point); back to space 0 without a pyramid generation
TEST_F(EngineAppRender, SpaceLoadingSequence)
{
    EngineAppSetup s = fiveSpaceSetup();
    s.render.imgui   = true;
    int hookCalls    = 0;
    s.loadingScreen  = [&](EngineApp&, const SpaceLoading&) { hookCalls++; };
    EngineApp app(s);
    noTerrainDraws(app);
    app.scene()->setImguiVisible(true);
    const Vec3 cam0(500, 80, 500);
    app.setViewCamera(0, cam0, Quat::Identity(), 1.0);
    app.runFrames(1, 1.0 / 60);  // a complete point in space 0 (never calls the latched hook)
    const auto* pyramid0 = app.world().terrain().pyramid(0);
    ASSERT_NE(pyramid0, nullptr);
    TerrainTestHooks::armLatch();
    LatchGuard open;
    auto frame = [&]() { app.runFrames(1, 1.0 / 60); };

    // 1. The enclosed space 1 and the flat space 4: no load
    app.setViewCameraByTool(1, Vec3(640, 2, 640), Quat::Identity(), 1.0, true);
    for (int i = 0; i < 3; ++i)
    {
        frame();
        EXPECT_EQ(app.spaceLoading(), nullptr);
    }
    app.setViewCameraByTool(4, Vec3(10, 2, 10), Quat::Identity(), 1.0, true);
    for (int i = 0; i < 3; ++i)
    {
        frame();
        EXPECT_EQ(app.spaceLoading(), nullptr);
    }
    EXPECT_EQ(hookCalls, 0);

    // 2. The latched space 2: a load at 0 while the pyramid job is held
    const Vec3 cam2(1000, 300, 1000);
    app.setViewCameraByTool(2, cam2, Quat::Identity(), 1.0, true);
    frame();
    ASSERT_NE(app.spaceLoading(), nullptr);
    EXPECT_EQ(app.spaceLoading()->space, 2);
    EXPECT_EQ(app.spaceLoading()->progress, 0.0);
    const int callsAtStart = hookCalls;
    EXPECT_EQ(callsAtStart, 1);
    for (int i = 0; i < 10; ++i)
    {
        frame();
        ASSERT_NE(app.spaceLoading(), nullptr);
        EXPECT_EQ(app.spaceLoading()->progress, 0.0);
    }
    EXPECT_EQ(hookCalls, callsAtStart + 10);
    const Json status = callWhileRunning(app, "app_status", Json::object{});
    const Json sl     = toolJson(status)["space_loading"];
    ASSERT_TRUE(sl.is_object()) << status.dump().substr(0, 400);
    EXPECT_EQ(sl["space"].int_value(), 2);
    EXPECT_EQ(sl["name"].string_value(), "latched");
    EXPECT_EQ(sl["progress"].number_value(), 0.0);
    EXPECT_GT(sl["frames"].number_value(), 10);

    // 4. Out through the enclosed space (no load, no pool work), then a new load in space 2 at a fixed camera; the
    // latch opens: the progress never decreases and the load ends at end rule 1
    app.setViewCameraByTool(1, Vec3(640, 2, 640), Quat::Identity(), 1.0, true);
    frame();
    EXPECT_EQ(app.spaceLoading(), nullptr);
    app.setViewCameraByTool(2, cam2, Quat::Identity(), 1.0, true);
    frame();
    ASSERT_NE(app.spaceLoading(), nullptr);
    TerrainTestHooks::openLatch();
    double last     = 0;
    uint64_t frames = 0;
    for (int i = 0; i < 600 && app.spaceLoading(); ++i)
    {
        EXPECT_GE(app.spaceLoading()->progress, last);
        last   = app.spaceLoading()->progress;
        frames = app.spaceLoading()->frames;
        frame();
    }
    EXPECT_EQ(app.spaceLoading(), nullptr);
    // End rule 1, not the frame limit: frames is the count before the last frame, which ended the load (PR review)
    EXPECT_LT(frames + 1, 600u);
    EXPECT_EQ(app.world().spaceReadiness(2, cam2, s.loadingRadius), 1.0);

    // 5. The unlatched space 3: a load (the pyramid is generated on the pool), cut off by an MCP screenshot
    app.setViewCameraByTool(3, Vec3(2000, 300, 2000), Quat::Identity(), 1.0, true);
    frame();
    ASSERT_NE(app.spaceLoading(), nullptr);
    const Json shot = callWhileRunning(app, "screenshot", Json::object{});
    EXPECT_FALSE(isError(shot)) << shot.dump().substr(0, 300);
    EXPECT_EQ(app.spaceLoading(), nullptr);

    // 6. Back to space 0: its tiles were dropped at the first switch (D5), its pyramid is cached
    const uint64_t pyramids = app.world().terrain().stats().pyramidsGenerated;
    app.setViewCameraByTool(0, cam0, Quat::Identity(), 1.0, true);
    frame();
    ASSERT_NE(app.spaceLoading(), nullptr);
    for (int i = 0; i < 600 && app.spaceLoading(); ++i) frame();
    EXPECT_EQ(app.spaceLoading(), nullptr);
    EXPECT_EQ(app.world().terrain().pyramid(0), pyramid0);
    EXPECT_EQ(app.world().terrain().stats().pyramidsGenerated, pyramids);
}

// A moving camera in the latched space (loadingMaxFrames 30, the screen off): the progress never decreases and the load
// ends after 30 frames with a log line; nothing is drawn while spaceLoading reports the load
TEST_F(EngineAppRender, SpaceLoadingMovingCamera)
{
    EngineAppSetup s    = fiveSpaceSetup();
    s.render.imgui      = true;
    s.loadingMaxFrames  = 30;
    s.showLoadingScreen = false;
    int hookCalls       = 0;
    s.loadingScreen     = [&](EngineApp&, const SpaceLoading&) { hookCalls++; };
    EngineApp app(s);
    noTerrainDraws(app);
    app.scene()->setImguiVisible(true);
    app.setViewCamera(0, Vec3(500, 80, 500), Quat::Identity(), 1.0);
    app.runFrames(1, 1.0 / 60);
    TerrainTestHooks::armLatch();
    LatchGuard open;
    const uint64_t before = engineLog().lastSequence();
    Vec3 cam(1000, 300, 1000);
    app.setViewCameraByTool(2, cam, Quat::Identity(), 1.0, true);
    app.runFrames(1, 1.0 / 60);
    ASSERT_NE(app.spaceLoading(), nullptr);
    double last = 0;
    int frames  = 1;
    while (app.spaceLoading() && frames < 100)
    {
        EXPECT_GE(app.spaceLoading()->progress, last);
        last = app.spaceLoading()->progress;
        cam.x() += 100;  // 3 km in 30 frames, inside the 8 km space
        app.setViewCameraByTool(2, cam, Quat::Identity(), 1.0, true);
        app.runFrames(1, 1.0 / 60);
        frames++;
    }
    EXPECT_EQ(frames, 30);
    EXPECT_EQ(app.viewCamera().space, 2);
    EXPECT_EQ(hookCalls, 0);
    bool logged = false;
    for (const auto& l : engineLog().since(before))
        logged = logged || l.message.find("loading of space 'latched' ended after 30 frames") != std::string::npos;
    EXPECT_TRUE(logged);
}

// --coop: the rendered client world's camera decides
TEST_F(EngineAppRender, SpaceLoadingCoopClientCamera)
{
    EngineAppSetup s       = coopSetup();
    const EngineAppSetup r = appSetup(true);
    s.noRender             = false;
    s.validation           = r.validation;
    s.render               = r.render;
    s.render.imgui         = true;
    // The latched hook holds the load over the screenshot frame (PR review: with a fast hook the load could end
    // before the screen was drawn)
    SpaceTemplate hills           = TerrainTestHooks::testSpaceTemplate(16, 2, TerrainTestHooks::latchedHook, 0);
    hills.name                    = "hills";
    s.world.config.spaceTemplates = {SpaceTemplate(), hills};
    s.world.generators            = TerrainTestHooks::registerTestHookGenerators;
    EngineApp app(s, coopOptions(1));
    noTerrainDraws(app);
    ASSERT_NE(app.hostWorld(), nullptr);
    app.scene()->setImguiVisible(true);
    app.runFrames(2, 1.0 / 60);
    TerrainTestHooks::armLatch();
    LatchGuard open;
    app.setViewCameraByTool(1, Vec3(1000, 80, 1000), Quat::Identity(), 1.0, true);
    app.runFrames(1, 1.0 / 60);
    ASSERT_NE(app.spaceLoading(), nullptr);
    // The engine's default screen over the whole viewport (a frame of the scene directly: an app capture would be a
    // complete point, which ends the load)
    TemplatedImage<ucvec4> shot;
    app.scene()->requestScreenshot([&](const TemplatedImage<ucvec4>& img) { shot = img; });
    app.runFrames(1, 1.0 / 60);
    app.scene()->waitIdle();
    ASSERT_GT(shot.w, 0);
    for (const ivec2 p : {ivec2(1, shot.h - 2), ivec2(shot.w - 2, shot.h - 2)})  // below the debug UI's bar
    {
        const ucvec4 c = shot(p.y(), p.x());
        EXPECT_NEAR(int(c(0)), 12, 3);
        EXPECT_NEAR(int(c(1)), 13, 3);
        EXPECT_NEAR(int(c(2)), 16, 3);
    }
    TerrainTestHooks::openLatch();
    // The load's own frame count (it ran frames before this loop), before the frame that ended it. Regression
    // (CodeRabbit, S21): the loop counted only its own frames, so an end at the frame limit passed the check
    uint64_t frames = 0;
    for (int i = 0; i < 600 && app.spaceLoading(); ++i)
    {
        frames = app.spaceLoading()->frames;
        app.runFrames(1, 1.0 / 60);
    }
    EXPECT_EQ(app.spaceLoading(), nullptr);
    EXPECT_LT(frames + 1, s.loadingMaxFrames);  // end rule 1, not the frame limit
    // The client world streamed it, the hidden host world nothing
    EXPECT_NE(app.world().terrain().pyramid(1), nullptr);
    EXPECT_EQ(app.hostWorld()->terrain().pyramid(1), nullptr);
}

// ---------------------------------------------------------------------------------------------------------------------
// Streaming (S24): the complete point of the residency, the loading screen of an arrival, the Engine tab's section

namespace
{
EngineAppSetup streamingAppSetup()
{
    EngineAppSetup s = appSetup(true);
    s.world          = StreamingTest::streamingTestSetup(StreamingTest::sharedStreamingCache());
    // As terrainAppSetup: no cascades and no horizon map (the frames after a camera jump of kilometres)
    s.render.lighting.sunShadows     = false;
    s.render.lighting.horizonShadows = false;
    return s;
}

// Holds the residency's jobs on the pool's workers (TerrainTestHooks' latch: the main thread never blocks)
void holdResidencyJobs(World& w)
{
    w.residency().jobHookForTests = [](const ChunkId&) { TerrainTestHooks::latchedHook(0, 0, 0); };
}
}  // namespace

namespace
{
// Every chunk within the required distance of (1150, 1400) is ready
void expectRequiredReady(World& w, SpaceId s)
{
    const SpaceTemplate& t = w.spaceTemplate(s);
    int checked            = 0;
    for (int z = 0; z < t.sizeChunks; ++z)
        for (int x = 0; x < t.sizeChunks; ++x)
        {
            const double dx = std::max({x * 256.0 - 1150, 0.0, 1150 - (x + 1) * 256.0});
            const double dz = std::max({z * 256.0 - 1400, 0.0, 1400 - (z + 1) * 256.0});
            if (std::sqrt(dx * dx + dz * dz) > t.requiredDistance()) continue;
            ChunkId c;
            c.space = s;
            c.x     = x;
            c.z     = z;
            EXPECT_TRUE(w.residency().ready(c)) << x << ", " << z;
            checked++;
        }
    EXPECT_GT(checked, 0);
}
}  // namespace

// The first frame's complete point leaves every required chunk of the streaming centres resident (the asynchronous
// residency, World::completeResidency before completeView); then (one app for both: an EngineApp with a RenderScene
// costs 0.3 s on lavapipe) the loading screen of the local player's arrival
// starts at the request frame (the target space, progress below 0.5 while the island is held), continues after the
// move and ends with the player's chunk ready
TEST_F(EngineAppRender, StreamingCompletePointAndLoadingAtRequest)
{
    EngineApp app(streamingAppSetup());
    noTerrainDraws(app);
    World& w = app.world();
    // test_small: a pyramid of 129^2 samples (the overworld's 1025^2 cost 0.4 s; test speed)
    const SpaceId s = StreamingTest::spaceNamed(w, "test_small");
    const Entity p  = StreamingTest::addPlayer(w, Vec3(1150, 0, 1400), s);
    app.setLocalPlayer(p);
    app.setViewCamera(s, Vec3(1150, w.terrainHeight(s, 1150, 1400) + 20, 1400), Quat::Identity(), 1.0);
    ASSERT_FALSE(w.streamingSettings().synchronous);
    app.runFrames(1, 1.0 / 60);
    expectRequiredReady(w, s);
    ASSERT_EQ(app.spaceLoading(), nullptr);
    holdResidencyJobs(w);
    TerrainTestHooks::ScopedLatch latch;
    // Beyond the kept distance (192 m + the island's 64 m): its island has chunks that are not resident (a near target:
    // few new chunks to generate, test speed)
    const Vec3 target(1600, 0, 1400);
    ASSERT_TRUE(w.requestArrival(p, s, target));
    app.runFrames(1, 1.0 / 60);
    ASSERT_NE(app.spaceLoading(), nullptr);
    EXPECT_TRUE(app.spaceLoading()->arrival);
    EXPECT_EQ(app.spaceLoading()->space, s);
    EXPECT_LT(app.spaceLoading()->progress, 0.5);
    app.runFrames(3, 1.0 / 60);
    ASSERT_NE(app.spaceLoading(), nullptr);
    EXPECT_TRUE(w.has<PendingTransition>(p));
    // The game's camera follows the player (set before every frame, as games do in beforeRender)
    auto follow = [&]()
    {
        const Vec3& at = w.read<Transform>(p).position();
        app.setViewCamera(s, at + Vec3(0, w.terrainHeight(s, at.x(), at.z()) - at.y() + 20, 0), Quat::Identity(), 1.0);
    };
    w.residency().jobHookForTests = nullptr;
    TerrainTestHooks::openLatch();
    double last = 0;
    for (int i = 0; i < 600 && app.spaceLoading(); ++i)
    {
        EXPECT_GE(app.spaceLoading()->progress, last);
        last = app.spaceLoading()->progress;
        follow();
        app.runFrames(1, 1.0 / 60);
    }
    EXPECT_EQ(app.spaceLoading(), nullptr);
    EXPECT_FALSE(w.has<PendingTransition>(p));
    EXPECT_NEAR(w.read<Transform>(p).position().x(), target.x(), 1e-6);
    EXPECT_TRUE(w.chunkReady(s, target.x(), target.z()));
}

// The Engine tab's Streaming section renders (a streaming world and one without streaming)
// (a streaming world; the section of a world without streaming is one disabled line: test speed, one app)
TEST_F(EngineAppRender, DebugUiStreamingSectionRenders)
{
    EngineAppSetup s = streamingAppSetup();
    s.render.imgui   = true;
    s.width          = 800;
    s.height         = 600;
    EngineApp app(s);
    noTerrainDraws(app);
    const SpaceId small = StreamingTest::spaceNamed(app.world(), "test_small");
    StreamingTest::addPlayer(app.world(), Vec3(1150, 0, 1400), small);
    app.setViewCamera(small, Vec3(1150, 300, 1400), Quat::Identity(), 1.0);
    Json::object state = app.debugUi().state().object_items();
    state["sections"]  = Json::object{{"engine/Streaming", true}};
    app.debugUi().setState(state);
    app.debugUi().select(DebugTab::Engine);
    app.runFrames(3, 1.0 / 60);
    ASSERT_TRUE(ImGui::GetCurrentContext());
    EXPECT_NE(ImGui::FindWindowByName("##debug panel"), nullptr);
}
