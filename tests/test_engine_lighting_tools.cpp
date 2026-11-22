/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// S36: the lighting's MCP tools (get_lighting, set_time_of_day, set_weather, set_lighting, app_status.lighting) through
// handleMessage, on one headless app per suite (with a device: skipped without a Vulkan 1.3 driver; without a device;
// --coop), and the console commands.

#include "saiga/core/util/mcp/McpServer.h"
#include "saiga/engine/app/DebugUi.h"
#include "saiga/engine/app/EngineApp.h"
#include "saiga/engine/components/Weather.h"
#include "saiga/engine/components/WorldTime.h"
#include "saiga/engine/core/Log.h"
#include "saiga/engine/lighting/Systems.h"
#include "saiga/vulkan/Instance.h"
#include "saiga/vulkan/scene/RenderScene.h"

#include "gtest/gtest.h"

#include <cmath>
#include <limits>
#include <memory>

#include "mcp_test_helper.h"

using namespace Saiga;
using namespace Saiga::Engine;
using Json = json11::Json;

namespace
{
const bool g_noEcho = (setLogEcho(false), true);

EngineAppSetup lightingSetup(bool render)
{
    EngineAppSetup s;
    s.name                      = "test_lighting";
    s.headless                  = true;
    s.noRender                  = !render;
    s.width                     = 64;
    s.height                    = 48;
    s.mcpPort                   = 0;
    s.render.imgui              = false;
    s.validation                = render;
    s.world.environment.enabled = true;
    s.world.systems             = addEnvironmentSystems;
    // Two spaces: the open world and an enclosed one
    s.world.config.spaceTemplates = {SpaceTemplate(), SpaceTemplate::enclosed("cellar")};
    return s;
}

Json obj(const std::string& text)
{
    std::string err;
    Json j = Json::parse(text, err);
    EXPECT_TRUE(err.empty()) << err;
    return j;
}
Json call(EngineApp& app, const std::string& name, const Json& args)
{
    Poller poller(app.mcp());
    return app.mcp().handleMessage(McpRequest("tools/call", Json::object{{"name", name}, {"arguments", args}}));
}
bool isError(const Json& r)
{
    return r["result"]["isError"].bool_value();
}
std::string text(const Json& r)
{
    return r["result"]["content"][0]["text"].string_value();
}
Json result(const Json& r)
{
    std::string err;
    Json j = Json::parse(text(r), err);
    EXPECT_TRUE(err.empty()) << err << ": " << text(r);
    return j;
}
}  // namespace

// ---------------------------------------------------------------------------------------------------------------------
// With a device (one app for the suite)

class LightingTools : public ::testing::Test
{
   public:
    static void SetUpTestSuite()
    {
        if (!Vulkan::Instance::suitableDeviceAvailable()) return;
        app = std::make_unique<EngineApp>(lightingSetup(true));
        app->runFrames(2, 1.0 / 60);
    }
    static void TearDownTestSuite() { app.reset(); }

   protected:
    void SetUp() override
    {
        if (!app) GTEST_SKIP() << "No Vulkan 1.3 device available";
        // The state of the previous case: the default settings, noon, the view in space 0
        app->setEnvironmentSettings(Vulkan::EnvironmentSettings());
        app->scene()->setLightingParameters(Vulkan::LightingParameters());
        app->scene()->setToneMap(Vulkan::ToneMap::Identity);
        app->setViewCameraByTool(0, Vec3(0, 2, 6), Quat::Identity(), 1, false);
        ASSERT_FALSE(isError(call(*app, "set_time_of_day", obj(R"({"hours": 12, "day": 0, "rate": 0})"))));
    }
    static inline std::unique_ptr<EngineApp> app;
};

TEST_F(LightingTools, GetLightingKeysAndSections)
{
    const Json all = result(call(*app, "get_lighting", Json::object{}));
    for (const char* key : {"time", "weather", "sun", "moon", "shadow_light", "tone_map", "exposure", "ambient",
                            "clouds", "settings", "parameters"})
        EXPECT_FALSE(all[key].is_null()) << key << " " << all.dump();
    EXPECT_EQ(all["time"]["hours"].number_value(), 12);
    EXPECT_NEAR(all["sun"]["elevation_deg"].number_value(), 45, 0.1);
    EXPECT_EQ(all["shadow_light"], Json("sun"));
    EXPECT_EQ(all["tone_map"]["value"], Json("identity"));
    EXPECT_FALSE(all["tone_map"]["note"].string_value().empty());
    EXPECT_EQ(all["settings"]["exposure_mode"], Json("time_of_day"));
    EXPECT_EQ(all["parameters"]["environment_resolution"].number_value(), 128);
    // sections keeps only the named keys
    const Json some = result(call(*app, "get_lighting", obj(R"({"sections": ["sun", "time"]})")));
    EXPECT_EQ(some.object_items().size(), 2u);
    EXPECT_FALSE(some["sun"].is_null());
    const Json bad = call(*app, "get_lighting", obj(R"({"sections": ["sunlight"]})"));
    EXPECT_TRUE(isError(bad));
    EXPECT_NE(text(bad).find("sunlight"), std::string::npos);
    EXPECT_TRUE(isError(call(*app, "get_lighting", obj(R"({"section": ["sun"]})"))));
}

TEST_F(LightingTools, SetTimeOfDay)
{
    const Json r = result(call(*app, "set_time_of_day", obj(R"({"hours": 6.5, "rate": 60})")));
    EXPECT_EQ(r["values"]["hours"].number_value(), 6.5);
    EXPECT_EQ(r["values"]["rate"].number_value(), 60);
    EXPECT_EQ(r["previous"]["hours"].number_value(), 12);
    EXPECT_EQ(app->world().readGlobal<WorldTime>().timeOfDay, 6.5);
    // Errors name the argument; nothing changes
    for (const char* args : {R"({"hours": 24})", R"({"hours": -1})", R"({"rate": 4000})", R"({"day": 1.5})",
                             R"({"hours": "six"})", R"({"minutes": 3})"})
    {
        const Json e = call(*app, "set_time_of_day", obj(args));
        EXPECT_TRUE(isError(e)) << args;
    }
    EXPECT_NE(text(call(*app, "set_time_of_day", obj(R"({"hours": 24})"))).find("hours"), std::string::npos);
    EXPECT_NE(text(call(*app, "set_time_of_day", obj(R"({"minutes": 3})"))).find("minutes"), std::string::npos);
    Json::object nan{{"hours", std::numeric_limits<double>::quiet_NaN()}};
    EXPECT_TRUE(isError(call(*app, "set_time_of_day", nan)));
    EXPECT_EQ(app->world().readGlobal<WorldTime>().timeOfDay, 6.5);
    // The next frame's sun follows (rate 0: exactly at 6:00, on the horizon)
    ASSERT_FALSE(isError(call(*app, "set_time_of_day", obj(R"({"hours": 6, "rate": 0})"))));
    app->runFrames(1, 1.0 / 60);
    const Json l = result(call(*app, "get_lighting", obj(R"({"sections": ["sun"]})")));
    EXPECT_NEAR(l["sun"]["elevation_deg"].number_value(), 0, 0.5);
}

TEST_F(LightingTools, SetWeather)
{
    const Json r = result(call(*app, "set_weather", obj(R"({"cloudiness": 0.8, "wind": [4, -2]})")));
    EXPECT_EQ(r["space"].number_value(), 0);
    EXPECT_EQ(r["values"]["cloudiness"].number_value(), 0.8);
    EXPECT_EQ(r["values"]["wind"][1].number_value(), -2);
    EXPECT_EQ(r["previous"]["cloudiness"].number_value(), Weather().cloudiness);
    EXPECT_EQ(app->world().readSpaceState<Weather>(0).cloudiness, 0.8);
    // Per space
    ASSERT_FALSE(isError(call(*app, "set_weather", obj(R"({"space": 1, "fog_density": 0.01})"))));
    EXPECT_EQ(app->world().readSpaceState<Weather>(1).fogDensity, 0.01);
    EXPECT_EQ(app->world().readSpaceState<Weather>(0).fogDensity, Weather().fogDensity);
    for (const char* args : {R"({"cloudiness": 2})", R"({"fog_density": 0.5})", R"({"wind": [1]})",
                             R"({"space": 9, "rain": 0.5})", R"({"storm": 1})"})
        EXPECT_TRUE(isError(call(*app, "set_weather", obj(args)))) << args;
    EXPECT_NE(text(call(*app, "set_weather", obj(R"({"space": 9, "rain": 0.5})"))).find("space"), std::string::npos);
    EXPECT_NE(text(call(*app, "set_weather", obj(R"({"cloudiness": 2})"))).find("cloudiness"), std::string::npos);
    // get_lighting reports the view space's weather
    app->runFrames(1, 1.0 / 60);
    const Json l = result(call(*app, "get_lighting", obj(R"({"sections": ["weather"]})")));
    EXPECT_EQ(l["weather"]["values"]["cloudiness"].number_value(), 0.8);
}

// Every key round-trips through previous; invalid values are errors that name every key and change nothing
TEST_F(LightingTools, SetLightingRoundTrip)
{
    const Json set = obj(R"({"tone_map": "aces", "exposure_mode": "fixed", "ev100": 12.5,
        "exposure_compensation": 1, "ambient_scale": 2, "min_ambient": [0.01, 0.02, 0.03],
        "sun_override": {"azimuth_deg": 90, "elevation_deg": 20}, "cloud_altitude": 1800, "cloud_radius": 30000,
        "cloud_shadow_fade": 5000, "star_intensity": 2, "ground_albedo": 0.3, "clouds": false,
        "environment_resolution": 64, "environment_update_degrees": 1, "force_full_updates": true})");
    const Json r   = result(call(*app, "set_lighting", set));
    EXPECT_EQ(r["values"].object_items().size(), set.object_items().size());
    const Json l = result(call(*app, "get_lighting", obj(R"({"sections": ["settings", "parameters", "tone_map"]})")));
    for (auto& kv : r["values"].object_items())
    {
        if (kv.first == "tone_map") continue;
        const Json& got = l["settings"][kv.first].is_null() ? l["parameters"][kv.first] : l["settings"][kv.first];
        EXPECT_EQ(got, kv.second) << kv.first;
    }
    EXPECT_EQ(l["tone_map"]["value"], Json("aces"));
    EXPECT_EQ(app->environmentSettings().exposure.ev100, 12.5f);
    EXPECT_EQ(app->scene()->lightingParameters().environmentResolution, 64);
    EXPECT_TRUE(app->environmentSettings().sunOverride.has_value());
    // The sun override moves the sun of the next frame; the next frame's EV100 is the fixed value minus the
    // compensation
    app->runFrames(1, 1.0 / 60);
    const Json after = result(call(*app, "get_lighting", obj(R"({"sections": ["sun", "exposure"]})")));
    EXPECT_NEAR(after["sun"]["elevation_deg"].number_value(), 20, 1e-3);
    EXPECT_NEAR(after["exposure"]["ev100"].number_value(), 11.5, 1e-4);
    EXPECT_EQ(after["exposure"]["source"], Json("fixed"));
    // previous restores everything (an undo payload)
    ASSERT_FALSE(isError(call(*app, "set_lighting", r["previous"])));
    const Json back = result(call(*app, "get_lighting", obj(R"({"sections": ["settings", "parameters"]})")));
    EXPECT_TRUE(back["settings"]["sun_override"].is_null());
    EXPECT_EQ(back["settings"]["ev100"].number_value(), 15);
    EXPECT_EQ(back["parameters"]["environment_resolution"].number_value(), 128);
    EXPECT_EQ(app->scene()->parameters().toneMap, Vulkan::ToneMap::Identity);

    // All or nothing: one valid and two invalid keys change nothing and name both invalid ones
    const Json bad =
        call(*app, "set_lighting", obj(R"({"ev100": 3, "environment_resolution": 100, "ground_albedo": 2})"));
    EXPECT_TRUE(isError(bad));
    EXPECT_NE(text(bad).find("environment_resolution"), std::string::npos) << text(bad);
    EXPECT_NE(text(bad).find("ground_albedo"), std::string::npos) << text(bad);
    EXPECT_EQ(app->environmentSettings().exposure.ev100, 15.0f);
    for (const char* args : {R"({"tone_map": "reinhard"})", R"({"exposure_mode": "auto"})",
                             R"({"min_ambient": [1, 2]})", R"({"sun_override": {"azimuth": 3}})",
                             R"({"environment_resolution": 1024})", R"({"clouds": 1})", R"({"cascade_count": 4})"})
        EXPECT_TRUE(isError(call(*app, "set_lighting", obj(args)))) << args;
    // tone_map identity is clay: get_lighting says so
    ASSERT_FALSE(isError(call(*app, "set_lighting", obj(R"({"tone_map": "identity"})"))));
    const Json clay = result(call(*app, "get_lighting", obj(R"({"sections": ["tone_map"]})")));
    EXPECT_EQ(clay["tone_map"]["value"], Json("identity"));
}

TEST_F(LightingTools, AppStatusLighting)
{
    ASSERT_FALSE(isError(call(*app, "set_lighting", obj(R"({"tone_map": "aces"})"))));
    ASSERT_FALSE(isError(call(*app, "set_time_of_day", obj(R"({"hours": 9.5})"))));
    app->runFrames(1, 1.0 / 60);
    const Json st = result(call(*app, "app_status", Json::object{}));
    ASSERT_TRUE(st["lighting"].is_object()) << st.dump();
    EXPECT_NEAR(st["lighting"]["time_of_day"].number_value(), 9.5, 1e-9);
    EXPECT_EQ(st["lighting"]["tone_map"], Json("aces"));
    EXPECT_EQ(st["lighting"]["debug_view"], Json("none"));
    EXPECT_TRUE(st["lighting"]["ev100"].is_number());
}

// ---------------------------------------------------------------------------------------------------------------------
// Without a device

TEST(LightingToolsNoDevice, RenderToolsNeedADevice)
{
    EngineApp app(lightingSetup(false));
    const Json g = call(app, "get_lighting", Json::object{});
    EXPECT_TRUE(isError(g));
    EXPECT_NE(text(g).find("no device"), std::string::npos);
    EXPECT_TRUE(isError(call(app, "set_lighting", obj(R"({"ev100": 3})"))));
    // The world state needs no device
    EXPECT_FALSE(isError(call(app, "set_time_of_day", obj(R"({"hours": 20})"))));
    EXPECT_EQ(app.world().readGlobal<WorldTime>().timeOfDay, 20);
    EXPECT_TRUE(result(call(app, "app_status", Json::object{}))["lighting"].is_null());
}

TEST(LightingToolsNoDevice, NoWorldTimeIsAnError)
{
    EngineAppSetup s            = lightingSetup(false);
    s.world.environment.enabled = false;
    EngineApp app(s);
    const Json r = call(app, "set_time_of_day", obj(R"({"hours": 20})"));
    EXPECT_TRUE(isError(r));
    EXPECT_NE(text(r).find("EnvironmentSetup::enabled"), std::string::npos);
    EXPECT_TRUE(isError(call(app, "set_weather", obj(R"({"cloudiness": 0.5})"))));
}

// --coop: the tools write the hidden host; the client gets the values by replication
TEST(EnvironmentWrites, CoopWritesTheHost)
{
    AppOptions o;
    o.coop      = true;
    o.coopDelay = 2;
    EngineApp app(lightingSetup(false), o);
    ASSERT_NE(app.hostWorld(), nullptr);
    app.runFrames(10, 1.0 / 60);
    const Json r = result(call(app, "set_time_of_day", obj(R"({"hours": 18, "rate": 0})")));
    EXPECT_EQ(r["world"], Json("host"));
    EXPECT_EQ(app.hostWorld()->readGlobal<WorldTime>().timeOfDay, 18);
    ASSERT_FALSE(isError(call(app, "set_weather", obj(R"({"cloudiness": 0.9, "wind": [0, 0]})"))));
    EXPECT_EQ(app.hostWorld()->readSpaceState<Weather>(0).cloudiness, 0.9);
    app.runFrames(20, 1.0 / 60);
    EXPECT_EQ(app.world().readGlobal<WorldTime>().timeOfDay, 18);
    EXPECT_EQ(app.world().readSpaceState<Weather>(0).cloudiness, 0.9);
}

// Every tool is a console command (DebugConsole through McpServer::callLocal)
TEST(DebugConsole, RunsLightingCommands)
{
    EngineApp app(lightingSetup(false));
    app.debugUi().console().submit("set_time_of_day hours=6");
    app.runFrames(2, 1.0 / 60);
    EXPECT_EQ(app.world().readGlobal<WorldTime>().timeOfDay, 6);
    app.debugUi().console().submit("set_weather cloudiness=0.25");
    app.runFrames(2, 1.0 / 60);
    EXPECT_EQ(app.world().readSpaceState<Weather>(0).cloudiness, 0.25);
}
