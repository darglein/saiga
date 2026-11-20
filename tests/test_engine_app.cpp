/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The engine app (docs/engine_design.md §8.3; stage S13): command line, scenario runner, input map, the frame loop
// headless (with and without a Vulkan device), the app tools and the frame and GPU profiles.

#include "saiga/core/util/mcp/McpServer.h"
#include "saiga/engine/app/EngineApp.h"
#include "saiga/engine/app/EngineMain.h"
#include "saiga/engine/components/Player.h"
#include "saiga/engine/components/PlayerInput.h"
#include "saiga/engine/core/Log.h"
#include "saiga/engine/tools/ScenarioList.h"
#include "saiga/engine/tools/ScenarioWorld.h"
#include "saiga/vulkan/Instance.h"
#include "saiga/vulkan/scene/RenderScene.h"

#include "gtest/gtest.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>

#include "mcp_test_helper.h"

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
    s.name          = "test_app";
    s.headless      = true;
    s.noRender      = !render;
    s.width         = 64;
    s.height        = 48;
    s.mcpPort       = 0;
    s.world.actions = actions;
    s.render.imgui  = false;
    s.validation    = render;  // a validation error fails the render tests (if the layer is installed)
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
        const char* old = std::getenv(name);
        had             = old != nullptr;
        if (had) previous = old;
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
        static int counter = 0;
        path               = std::filesystem::path(::testing::TempDir()) /
               ("saiga_engine_app_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
                "_" + std::to_string(counter++));
        std::filesystem::create_directories(path);
    }
    ~TempDirectory()
    {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
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
// The app without a device (servers, agents, CI without Vulkan)

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
    int transfers = 0, renders = 0;
    s.renderHooks.transfer = [&](vk::CommandBuffer) { transfers++; };
    s.renderHooks.render   = [&](vk::CommandBuffer) { renders++; };
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
    EXPECT_EQ(transfers, 60);
    EXPECT_EQ(renders, 60);
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
    // quit from a frame callback ends run()
    app.onFrameRendered(
        [&]()
        {
            app.quit(3);
            return true;
        });
    EXPECT_EQ(app.run(), 3);
}

// --screenshot: saved after the given frame, then the app quits
TEST_F(EngineAppRender, ScreenshotOption)
{
    TempDirectory dir;
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
