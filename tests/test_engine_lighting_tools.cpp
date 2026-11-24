/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// S36: the lighting's MCP tools (get_lighting, set_time_of_day, set_weather, set_lighting, app_status.lighting) through
// handleMessage, on a headless app per test (with a device: skipped without a Vulkan 1.3 driver; without a device;
// --coop), the console commands, and the app's environment and Lighting tab (F5) (with and without
// EnvironmentSetup::enabled).

#include "saiga/core/image/templatedImage.h"
#include "saiga/core/util/mcp/McpServer.h"
#include "saiga/engine/api/WorldApi.h"
#include "saiga/engine/app/DebugUi.h"
#include "saiga/engine/app/EngineApp.h"
#include "saiga/engine/components/LightSource.h"
#include "saiga/engine/components/Weather.h"
#include "saiga/engine/components/WorldTime.h"
#include "saiga/engine/core/Log.h"
#include "saiga/engine/lighting/EnvironmentWrites.h"
#include "saiga/engine/lighting/Systems.h"
#include "saiga/engine/sync/ExtractSystem.h"
#include "saiga/vulkan/Instance.h"
#include "saiga/vulkan/scene/LightingDebug.h"
#include "saiga/vulkan/scene/RenderScene.h"

#include "gtest/gtest.h"

#include <cfloat>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <set>

#include "lighting_test_parameters.h"
#include "mcp_test_helper.h"
#include "terrain_test_hooks.h"

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
    s.render.lighting           = testLightingParameters();
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

// The debug UI's apps: ImGui, large enough for the Lighting tab's sections to lay out
EngineAppSetup tabSetup(bool environment)
{
    EngineAppSetup s            = lightingSetup(true);
    s.width                     = 800;
    s.height                    = 480;
    s.render.imgui              = true;
    s.world.environment.enabled = environment;
    // S36a2: the cascades on (small: lavapipe)
    s.render.lighting.sunShadows        = true;
    s.render.lighting.cascadeResolution = 512;
    s.render.lighting.horizonShadows    = true;
    s.render.lighting.horizonResolution = 128;
    return s;
}

// Opens the sections of the Lighting tab and selects it
void showLightingTab(EngineApp& app)
{
    Json::object state = app.debugUi().state().object_items();
    state["sections"]  = Json::object{{"lighting/Time and weather", true},
                                      {"lighting/Sun and moon", true},
                                      {"lighting/Exposure", true},
                                      {"lighting/Ambient", true},
                                      {"lighting/Sky and clouds", true},
                                      {"lighting/GPU and memory", true},
                                      {"lighting/Cascades", true},
                                      {"lighting/Far terrain shadows", true},
                                      {"lighting/Local lights", true},
                                      {"lighting/Local lights: clusters", true},
                                      {"lighting/Local lights: shadow atlas", true}};
    app.debugUi().setState(state);
    app.debugUi().select(DebugTab::Lighting);
}

// The tab with its sections open renders (no validation error); S36a1b: the open Exposure section shows the luminance
// preview, converted once per preview interval (here once in the three frames; not again in a later case of the
// suite's app); S36a1c: also the Ambient section's environment cube, Sky and clouds' transmittance and sky-view LUTs
// and the clouds' previews; S36a2 Cascades the cascades' strip; S36b Local lights the clusters' heat map, the shadow
// atlas and the highlighted lamp's tile; a debug view (Exposure: with its legend) and the overlays render in the tab's
// frames (S36a2: the cascades' view and overlays, frozen)
void renderLightingTab(EngineApp& app)
{
    // S36b: a lamp (an entity light, highlighted: the table's selected row with its LightSource widgets) and the local
    // lights on; removed at the end (the suite's app)
    const Entity lamp = app.world().create();
    {
        app.world().get<Transform>(lamp).setPosition(Vec3(0, 1, 0));
        LightSource l;
        l.castShadows = true;
        app.world().add<LightSource>(lamp, l);
        Vulkan::LightingParameters lp = app.scene()->lightingParameters();
        lp.localLights                = true;
        app.scene()->setLightingParameters(lp);
        app.runFrames(1, 1.0 / 60);
        Vulkan::LightingDebugState d;
        d.highlightLight = app.world().extract().lightOf(lamp).id;
        EXPECT_NE(d.highlightLight, 0u);
        app.scene()->setLightingDebug(d);
    }
    showLightingTab(app);
    app.scene()->lightingDebugImages().setPreviewInterval(1e9);
    const Vulkan::LightingImage images[] = {
        Vulkan::LightingImage::Luminance,          Vulkan::LightingImage::Environment,
        Vulkan::LightingImage::TransmittanceLut,   Vulkan::LightingImage::SkyViewLut,
        Vulkan::LightingImage::CloudCoverage,      Vulkan::LightingImage::CloudShadow,
        Vulkan::LightingImage::MultiScatteringLut, Vulkan::LightingImage::CascadeStrip,
        Vulkan::LightingImage::Clusters,           Vulkan::LightingImage::Atlas,
        Vulkan::LightingImage::AtlasTile};
    uint32_t expected = 0;
    for (auto image : images) expected += app.scene()->lightingDebugImages().preview(image) == nullptr ? 1 : 0;
    uint32_t dispatches = 0;
    for (int f = 0; f < 3; ++f)
    {
        app.runFrames(1, 1.0 / 60);
        dispatches += app.scene()->lightingStatistics().previewDispatches;
    }
    EXPECT_TRUE(app.debugUi().isSelected(DebugTab::Lighting));
    EXPECT_TRUE(app.scene()->lightingStatistics().shadows);
    EXPECT_EQ(dispatches, expected);
    for (auto image : images) EXPECT_NE(app.scene()->lightingDebugImages().preview(image), nullptr);
    app.scene()->setLightingDebug(
        Vulkan::LightingDebugState{Vulkan::LightingDebugView::Exposure, Vulkan::OverlaySunDirection});
    app.runFrames(2, 1.0 / 60);
    app.scene()->setFreezeCascades(true);
    app.scene()->setLightingDebug(
        Vulkan::LightingDebugState{Vulkan::LightingDebugView::ShadowCascades, Vulkan::OverlayAll});
    app.runFrames(2, 1.0 / 60);
    app.scene()->setFreezeCascades(false);
    app.scene()->setLightingDebug(Vulkan::LightingDebugState());
    EXPECT_EQ(app.scene()->lightingStatistics().lights.submitted, 1u);
    app.world().destroy(lamp);
    // The Exposure view's legend leaves the toolbar: one frame for the layout (the item positions of a later click)
    app.runFrames(1, 1.0 / 60);
}

// A left click at a point: one input event per frame (ImGui spreads a move and a button change of one frame over two)
void click(EngineApp& app, ImVec2 at)
{
    ImGuiIO& io = ImGui::GetIO();
    io.AddMousePosEvent(at.x, at.y);
    app.runFrames(1, 1.0 / 60);
    io.AddMouseButtonEvent(ImGuiMouseButton_Left, true);
    app.runFrames(1, 1.0 / 60);
    io.AddMouseButtonEvent(ImGuiMouseButton_Left, false);
    app.runFrames(1, 1.0 / 60);
    io.AddMousePosEvent(-FLT_MAX, -FLT_MAX);
}

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
// A deferred tool: frames are rendered while the call waits (on the poller's thread)
Json callFrames(EngineApp& app, const std::string& name, const Json& args)
{
    Poller poller(app.mcp(), [&app]() { app.runFrames(1, 1.0 / 60); });
    return app.mcp().handleMessage(McpRequest("tools/call", Json::object{{"name", name}, {"arguments", args}}));
}
// The text block after an image (content[1])
Json imageBlock(const Json& r)
{
    std::string err;
    Json j = Json::parse(r["result"]["content"][1]["text"].string_value(), err);
    EXPECT_TRUE(err.empty()) << err;
    return j;
}
// The camera rotation looking along forward (y up)
Quat lookRotation(const Vec3& forward)
{
    const Vec3 back  = -forward.normalized();
    const Vec3 right = Vec3(0, 1, 0).cross(back).normalized();
    const Vec3 up    = back.cross(right);
    Mat3 R;
    for (int i = 0; i < 3; ++i)
    {
        R(i, 0) = right(i);
        R(i, 1) = up(i);
        R(i, 2) = back(i);
    }
    return Quat(R).normalized();
}
// A flat gray box of the half size (flat faces) in one LOD
Vulkan::MeshLods planeMesh(const vec3& half)
{
    Vulkan::MeshLod lod;
    lod.endDistance = 1000;
    const vec3 n(0, 1, 0);
    const vec3 corners[4] = {vec3(-1, 1, 1), vec3(1, 1, 1), vec3(1, 1, -1), vec3(-1, 1, -1)};
    for (const vec3& c : corners)
    {
        Vulkan::MeshVertex v;
        v.position = c.cwiseProduct(half);
        v.normal   = Vulkan::packNormalOct(n);
        v.color    = Vulkan::packColor(ucvec4(180, 180, 180, 255));
        lod.vertices.push_back(v);
    }
    for (uint32_t i : {0u, 1u, 2u, 0u, 2u, 3u}) lod.indices.push_back(i);
    Vulkan::MeshLods m;
    m.lods.push_back(lod);
    m.sphereRadius = half.norm();
    return m;
}
}  // namespace

// ---------------------------------------------------------------------------------------------------------------------
// With a device (an app per test)

class LightingTools : public ::testing::Test
{
   protected:
    void SetUp() override
    {
        if (!Vulkan::Instance::suitableDeviceAvailable()) GTEST_SKIP() << "No Vulkan 1.3 device available";
        app = std::make_unique<EngineApp>(lightingSetup(true));
        app->runFrames(2, 1.0 / 60);
        // Noon, the view in space 0
        app->setViewCameraByTool(0, Vec3(0, 2, 6), Quat::Identity(), 1, false);
        ASSERT_FALSE(isError(call(*app, "set_time_of_day", obj(R"({"hours": 12, "day": 0, "rate": 0})"))));
    }
    void TearDown() override { app.reset(); }
    std::unique_ptr<EngineApp> app;
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
        "rayleigh_scale": 1.5, "mie_scale": 0.5, "ozone_scale": 2, "fog_falloff": 0.002, "cloud_seed": 7,
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
    EXPECT_EQ(app->environmentSettings().atmosphere.rayleighScale, 1.5f);
    EXPECT_EQ(app->environmentSettings().clouds.seed, 7u);
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
    for (const char* args :
         {R"({"tone_map": "reinhard"})", R"({"exposure_mode": "auto"})", R"({"min_ambient": [1, 2]})",
          R"({"sun_override": {"azimuth": 3}})", R"({"environment_resolution": 1024})", R"({"clouds": 1})",
          R"({"local_lights": 1})", R"({"cascade_count": 5})", R"({"cascade_resolution": 1000})",
          R"({"cascade_interval": [2, 1, 1, 2]})", R"({"cloud_seed": 1.5})", R"({"cloud_seed": -1})",
          R"({"mie_scale": 11})"})
        EXPECT_TRUE(isError(call(*app, "set_lighting", obj(args)))) << args;
    // tone_map identity is clay: get_lighting says so
    ASSERT_FALSE(isError(call(*app, "set_lighting", obj(R"({"tone_map": "identity"})"))));
    const Json clay = result(call(*app, "get_lighting", obj(R"({"sections": ["tone_map"]})")));
    EXPECT_EQ(clay["tone_map"]["value"], Json("identity"));
}

// ---- S36a1b: the debug core's tools

// A plane under a camera looking down (the scene's own content: the world is empty)
class LightingDebugTools : public LightingTools
{
   protected:
    void SetUp() override
    {
        LightingTools::SetUp();
        if (IsSkipped() || HasFatalFailure()) return;
        Vulkan::RenderScene& s       = *app->scene();
        const Vulkan::MaterialId mat = Vulkan::MaterialId(s.reserveIds(Vulkan::SceneIdKind::Material, 1));
        const Vulkan::MeshId mesh    = Vulkan::MeshId(s.reserveIds(Vulkan::SceneIdKind::Mesh, 1));
        s.addMaterial(mat, "debug_plane", Vulkan::MaterialData());
        s.addMesh(mesh, "debug_plane", planeMesh(vec3(50, 1, 50)), mat);
        Vulkan::StaticInstance i;
        i.setPosition(vec3(0, -1, 0));
        i.scale = 1;
        i.mesh  = mesh;
        s.addInstanceSet(RegionId(), {i});
        app->setViewCameraByTool(0, Vec3(0, 6, 0.01), lookRotation(Vec3(0, -1, -0.002)), 1, false);
        app->runFrames(1, 1.0 / 60);
    }
};

TEST_F(LightingDebugTools, SetLightingDebug)
{
    const Vulkan::PreviewMapping mappingBefore = app->lightingPreviewMapping();
    // The tab's luminance is log2 by default
    EXPECT_TRUE(mappingBefore.log);
    for (int v = 0; v < int(Vulkan::LightingDebugView::Count); ++v)
    {
        const std::string name = Vulkan::lightingDebugViewName(Vulkan::LightingDebugView(v));
        const Json r           = call(*app, "set_lighting_debug", Json::object{{"view", name}});
        ASSERT_FALSE(isError(r)) << text(r);
        EXPECT_EQ(result(r)["view"], Json(name));
        EXPECT_EQ(int(app->scene()->lightingDebug().view), v);
    }
    const Json r = result(call(*app, "set_lighting_debug", obj(R"({"view": "normals", "overlays": ["sun_direction"],
                                       "mapping": {"range": [0, 2], "log": true, "color_map": "turbo"}})")));
    EXPECT_EQ(r["overlays"], Json(Json::array{"sun_direction"}));
    EXPECT_EQ(r["mapping"]["color_map"], Json("turbo"));
    EXPECT_FALSE(r["views"].array_items().empty());
    EXPECT_EQ(app->scene()->lightingDebug().overlays, uint32_t(Vulkan::OverlaySunDirection));
    EXPECT_TRUE(app->lightingPreviewMapping().log);
    const Json l = result(call(*app, "get_lighting", obj(R"({"sections": ["debug", "gpu", "memory"]})")));
    EXPECT_EQ(l["debug"]["view"], Json("normals"));
    EXPECT_TRUE(l["gpu"]["lighting.preview"].is_object()) << l.dump();
    EXPECT_GT(l["memory"]["bytes"].number_value(), 0);
    // Unknown names list the known ones; nothing changes
    const Json bad = call(*app, "set_lighting_debug", obj(R"({"view": "shiny"})"));
    EXPECT_TRUE(isError(bad));
    EXPECT_NE(text(bad).find("albedo"), std::string::npos) << text(bad);
    const Json badOverlay = call(*app, "set_lighting_debug", obj(R"({"overlays": ["light_cones"]})"));
    EXPECT_TRUE(isError(badOverlay));
    EXPECT_NE(text(badOverlay).find("sun_direction"), std::string::npos) << text(badOverlay);
    for (const char* args :
         {R"({"freeze_cascades": 1})", R"({"highlight_light": -3})", R"({"highlight_light": 1.5})",
          R"({"mapping": {"range": [2, 1]}})", R"({"mapping": {"gamma": 2}})", R"({"mapping": {"color_map": "jet"}})"})
        EXPECT_TRUE(isError(call(*app, "set_lighting_debug", obj(args)))) << args;
    EXPECT_EQ(app->scene()->lightingDebug().view, Vulkan::LightingDebugView::Normals);
    // The Exposure view's heat map (centre and range; the legend of the tab)
    const Json ev = result(call(*app, "set_lighting_debug", obj(R"({"exposure_view": {"centre": 12, "range": 4}})")));
    EXPECT_EQ(ev["exposure_view"]["centre"].number_value(), 12);
    EXPECT_EQ(app->scene()->exposureViewCentreSetting(), std::optional<float>(12.0f));
    EXPECT_EQ(app->scene()->exposureViewRangeSetting(), 4.0f);
    const Json evGet = result(call(*app, "get_lighting", obj(R"({"sections": ["debug"]})")));
    EXPECT_EQ(evGet["debug"]["exposure_view"]["range"].number_value(), 4) << evGet.dump();
    for (const char* args : {R"({"exposure_view": {"range": 0}})", R"({"exposure_view": {"centre": "x"}})",
                             R"({"exposure_view": {"width": 2}})"})
        EXPECT_TRUE(isError(call(*app, "set_lighting_debug", obj(args)))) << args;
    ASSERT_FALSE(
        isError(call(*app, "set_lighting_debug", obj(R"({"exposure_view": {"centre": "auto", "range": 8}})"))));
    EXPECT_FALSE(app->scene()->exposureViewCentreSetting().has_value());
    // The debug sections are reported apart from the lighting sections' budget
    EXPECT_TRUE(l["gpu"]["debug_ms"].is_number()) << l.dump();
    EXPECT_TRUE(l["gpu"]["lighting_ms"].is_number()) << l.dump();
    app->lightingPreviewMapping() = mappingBefore;
}

// screenshot(lighting_view, overlays): that frame shows the view (the Normals of the plane facing +y: (0.5, 1, 0.5) as
// a raw value, which the identity post pass shows unchanged), the next frame is back to the user's view
TEST_F(LightingDebugTools, ScreenshotLightingView)
{
    app->runFrames(1, 1.0 / 60);
    // nullopt if the png does not decode (a black fallback would pass the comparison below)
    auto centre = [](const Json& r) -> std::optional<ucvec3>
    {
        const std::string data = base64Decode(r["result"]["content"][0]["data"].string_value());
        TemplatedImage<ucvec3> img;
        if (!img.loadFromMemory(ArrayView<const char>(data.data(), data.size()), "png") || img.w == 0)
            return std::nullopt;
        // Left of the centre: the sun direction overlay's cross is at the centre
        return img(img.h / 2, img.w / 4);
    };
    const Json n = callFrames(*app, "screenshot", obj(R"({"format": "png", "lighting_view": "normals",
                                                          "overlays": ["sun_direction"]})"));
    ASSERT_FALSE(isError(n)) << text(n);
    const std::optional<ucvec3> viewCentre = centre(n);
    ASSERT_TRUE(viewCentre.has_value());
    const ucvec3 c = *viewCentre;
    EXPECT_NEAR(int(c(0)), 128, 2);
    EXPECT_NEAR(int(c(1)), 255, 2);
    EXPECT_NEAR(int(c(2)), 128, 2);
    EXPECT_EQ(app->scene()->lightingDebug().view, Vulkan::LightingDebugView::None);
    const Json normal = callFrames(*app, "screenshot", obj(R"({"format": "png"})"));
    ASSERT_FALSE(isError(normal));
    const std::optional<ucvec3> normalCentre = centre(normal);
    ASSERT_TRUE(normalCentre.has_value());
    EXPECT_NE(*normalCentre, c);
    EXPECT_EQ(app->scene()->frameLightingDebug().view, Vulkan::LightingDebugView::None);
    EXPECT_TRUE(isError(callFrames(*app, "screenshot", obj(R"({"lighting_view": "x-ray"})"))));
    EXPECT_TRUE(isError(callFrames(*app, "screenshot", obj(R"({"overlays": "sun_direction"})"))));
}

// capture_lighting: each a1b image returns a png of its size and the JSON block (mapping, stats with the histogram);
// stats_only returns the block alone; invalid arguments; a deadline error with rendering disabled, answered once
TEST_F(LightingDebugTools, CaptureLighting)
{
    const Json lum = callFrames(*app, "capture_lighting", obj(R"({"image": "luminance"})"));
    ASSERT_FALSE(isError(lum)) << text(lum);
    EXPECT_EQ(lum["result"]["content"][0]["mimeType"], Json("image/png"));
    const Json block = imageBlock(lum);
    EXPECT_EQ(block["image"], Json("luminance"));
    EXPECT_EQ(block["size"], Json(Json::array{64, 48}));
    EXPECT_EQ(block["mapping"]["log"], Json(true));
    EXPECT_EQ(block["stats"]["histogram"].array_items().size(), 16u);
    EXPECT_EQ(block["stats"]["count"].number_value(), 64 * 48);
    const Json lut = callFrames(*app, "capture_lighting", obj(R"({"image": "brdf_lut", "color_map": "turbo",
                                                                  "format": "jpg", "max_width": 64})"));
    ASSERT_FALSE(isError(lut)) << text(lut);
    EXPECT_EQ(lut["result"]["content"][0]["mimeType"], Json("image/jpeg"));
    EXPECT_EQ(imageBlock(lut)["size"], Json(Json::array{64, 64}));
    EXPECT_GT(imageBlock(lut)["stats"]["max"].number_value(), 0.5);
    const Json stats = callFrames(*app, "capture_lighting", obj(R"({"image": "brdf_lut", "stats_only": true,
                                                                    "range": [0, 1], "log": false})"));
    ASSERT_FALSE(isError(stats)) << text(stats);
    EXPECT_EQ(stats["result"]["content"].array_items().size(), 1u);
    EXPECT_EQ(result(stats)["mapping"]["range"], Json(Json::array{0.0, 1.0}));
    for (const char* args :
         {R"({"image": "cascade", "cascade": 7})", R"({})", R"({"image": "luminance", "max_width": 10})",
          R"({"image": "luminance", "cascade": 1})", R"({"image": "luminance", "range": [1]})"})
        EXPECT_TRUE(isError(callFrames(*app, "capture_lighting", obj(args)))) << args;
    // Rendering disabled: the deadline answers once (a short tool timeout; half of it, 0.25 s, is kept for the answer
    // so that it reaches the caller in time on a loaded machine)
    app->setRenderingEnabled(false);
    const double timeout            = app->mcp().toolTimeoutSeconds;
    const double margin             = app->mcp().answerMarginFraction;
    app->mcp().toolTimeoutSeconds   = 0.5;
    app->mcp().answerMarginFraction = 0.5;
    const Json late = call(*app, "capture_lighting", obj(R"({"image": "luminance", "stats_only": true})"));
    app->mcp().toolTimeoutSeconds   = timeout;
    app->mcp().answerMarginFraction = margin;
    app->setRenderingEnabled(true);
    EXPECT_TRUE(isError(late));
    EXPECT_NE(text(late).find("deadline"), std::string::npos) << text(late);
    // The stale request is computed with the next frame and not answered again
    app->runFrames(2, 1.0 / 60);
}

// capture_lighting, S36a1c: the atmosphere's LUTs and the environment map's images in the physical path (sizes,
// units), full: true completes the environment map first, mip only for environment; get_lighting.environment
TEST_F(LightingDebugTools, CaptureLightingAtmosphere)
{
    ASSERT_FALSE(isError(call(*app, "set_lighting", obj(R"({"tone_map": "aces"})"))));
    app->runFrames(2, 1.0 / 60);
    const std::vector<std::pair<std::string, Json>> images = {
        {R"({"image": "transmittance_lut"})", Json(Json::array{256, 64})},
        {R"({"image": "sky_view_lut"})", Json(Json::array{192, 108})},
        {R"({"image": "multi_scattering_lut"})", Json(Json::array{32, 32})},
        {R"({"image": "environment", "mip": 1, "full": true})", Json(Json::array{256, 192})},
        {R"({"image": "irradiance"})", Json(Json::array{128, 64})},
        {R"({"image": "cloud_coverage"})", Json(Json::array{512, 512})},
        {R"({"image": "cloud_shadow"})", Json(Json::array{256, 256})}};
    for (const auto& [args, size] : images)
    {
        const Json r = callFrames(*app, "capture_lighting", obj(args));
        ASSERT_FALSE(isError(r)) << args << " " << text(r);
        const Json block = imageBlock(r);
        EXPECT_EQ(block["size"], size) << args;
        EXPECT_FALSE(block["unit"].string_value().empty()) << args;
        if (args.find("environment") != std::string::npos)
        {
            EXPECT_EQ(block["mip"], Json(1)) << args;
        }
        // The cloud shadow may be uniform (a low sun fades it to the mean)
        if (args.find("cloud_shadow") == std::string::npos)
        {
            EXPECT_GT(block["stats"]["max"].number_value(), block["stats"]["min"].number_value()) << args;
        }
    }
    for (const char* args : {R"({"image": "sky_view_lut", "mip": 1})", R"({"image": "environment", "mip": 5})"})
        EXPECT_TRUE(isError(callFrames(*app, "capture_lighting", obj(args)))) << args;
    const Json env = result(call(*app, "get_lighting", obj(R"({"sections": ["environment"]})")));
    EXPECT_EQ(env["environment"]["valid"], Json(true)) << env.dump();
    EXPECT_GT(env["environment"]["captures"].number_value(), 0);
    EXPECT_GT(env["environment"]["lut_computations"].number_value(), 0);
}

// probe_lighting: a pixel of the lit plane returns its albedo, normal and a direct radiance > 0, 7 frames, consistent;
// the user's view is restored; a pixel outside the image and bad arguments are errors
TEST_F(LightingDebugTools, ProbeLighting)
{
    ASSERT_FALSE(isError(call(*app, "set_lighting", obj(R"({"tone_map": "aces"})"))));
    app->scene()->setLightingDebug(Vulkan::LightingDebugState{Vulkan::LightingDebugView::Roughness, 0});
    app->runFrames(1, 1.0 / 60);
    const Json r = callFrames(*app, "probe_lighting", obj(R"({"pixel": [20, 24]})"));
    ASSERT_FALSE(isError(r)) << text(r);
    const Json p = result(r);
    EXPECT_EQ(p["frames"].number_value(), Vulkan::RenderScene::kProbeFrames);
    EXPECT_EQ(p["consistent"], Json(true));
    EXPECT_EQ(p["hit"], Json(true));
    EXPECT_EQ(p["physical"], Json(true));
    EXPECT_NEAR(p["normal"][1].number_value(), 1.0, 0.01);
    // The vertex colour 180 (sRGB) -> linear 0.456
    EXPECT_NEAR(p["albedo"][0].number_value(), 0.456, 0.01);
    EXPECT_GT(p["radiance"]["direct"][0].number_value(), 0);
    // (a missing position would read as 0)
    ASSERT_EQ(p["position"].array_items().size(), 3u) << p.dump();
    ASSERT_TRUE(p["position"][1].is_number()) << p.dump();
    EXPECT_NEAR(p["position"][1].number_value(), 0, 0.05);
    EXPECT_EQ(app->scene()->lightingDebug().view, Vulkan::LightingDebugView::Roughness);
    // A world point: its pixel
    const Json w = callFrames(*app, "probe_lighting", obj(R"({"position": [0, 0, 0]})"));
    ASSERT_FALSE(isError(w)) << text(w);
    const Json wp = result(w);
    EXPECT_EQ(wp["hit"], Json(true)) << wp.dump();
    ASSERT_EQ(wp["position"].array_items().size(), 3u) << wp.dump();
    for (const Json& x : wp["position"].array_items()) ASSERT_TRUE(x.is_number()) << wp.dump();
    EXPECT_NEAR(wp["position"][0].number_value(), 0, 0.2);
    for (const char* args : {R"({"pixel": [64, 0]})", R"({"pixel": [-1, 3]})", R"({"pixel": [1.5, 2]})", R"({})",
                             R"({"pixel": [1, 2], "position": [0, 0, 0]})", R"({"pixel": [1, 2], "light": 3})"})
        EXPECT_TRUE(isError(callFrames(*app, "probe_lighting", obj(args)))) << args;
}

// S36a2: the cascades' tool keys: every set_lighting key of the part round-trips (with memory_bytes), get_lighting's
// cascades, capture_lighting(cascade) for one cascade and all side by side, freeze_cascades, and probe_lighting in the
// shadow of a card 1 m above the plane (the noon sun from the south: the shadow 1 m to the north) returns the cascade's
// factor and index
TEST_F(LightingDebugTools, CascadeTools)
{
    Vulkan::RenderScene& s = *app->scene();
    // The card (removed with the test's scene)
    const Vulkan::MaterialId mat = Vulkan::MaterialId(s.reserveIds(Vulkan::SceneIdKind::Material, 1));
    const Vulkan::MeshId mesh    = Vulkan::MeshId(s.reserveIds(Vulkan::SceneIdKind::Mesh, 1));
    s.addMaterial(mat, "shadow_card", Vulkan::MaterialData());
    s.addMesh(mesh, "shadow_card", planeMesh(vec3(0.5f, 0, 0.5f)), mat);
    Vulkan::StaticInstance i;
    i.setPosition(vec3(0, 1, 0));
    i.scale                              = 1;
    i.mesh                               = mesh;
    const Vulkan::InstanceSetHandle card = s.addInstanceSet(RegionId(), {i});
    // From shadows off (S36a2's flip made them the default)
    ASSERT_FALSE(isError(call(*app, "set_lighting", obj(R"({"tone_map": "aces", "sun_shadows": false})"))));
    const Json initial = result(call(*app, "get_lighting", obj(R"({"sections": ["parameters"]})")))["parameters"];
    // Every a2 key round-trips through previous: each requested key is in the answer's values and get_lighting's
    // parameters (float keys as floats), and the undo restores the initial values
    const Json request = obj(R"({"sun_shadows": true, "cascade_count": 3,
        "cascade_resolution": 1024, "shadow_distance": 100, "split_lambda": 0.7, "cascade_interval": [1, 1, 2, 2],
        "depth_bias_texels": 1.5, "slope_bias": 2, "normal_offset_texels": 0.5, "cascade_blend": 0.2,
        "shadow_lod_bias": 2, "small_caster_texels": 2, "max_caster_height": 150, "min_shadow_elevation_deg": 4,
        "cascade_caster_distance": 1000, "terrain_casts_in_cascades": false})");
    const Json set     = call(*app, "set_lighting", request);
    ASSERT_FALSE(isError(set)) << text(set);
    const Json values = result(set);
    EXPECT_EQ(values["previous"]["sun_shadows"], Json(false));
    EXPECT_EQ(values["memory_bytes"].number_value() >= 3.0 * 1024 * 1024 * 4, true) << values.dump();
    const Json params = result(call(*app, "get_lighting", obj(R"({"sections": ["parameters"]})")))["parameters"];
    auto same         = [](const Json& a, const Json& b)
    { return a.is_number() && b.is_number() ? float(a.number_value()) == float(b.number_value()) : a == b; };
    for (const auto& kv : request.object_items())
    {
        EXPECT_TRUE(same(values["values"][kv.first], kv.second)) << kv.first << ": " << values["values"].dump();
        EXPECT_TRUE(same(params[kv.first], kv.second)) << kv.first << ": " << params.dump();
    }
    EXPECT_EQ(s.lightingParameters().cascadeCount, 3);
    // Undo with previous, then the defaults with 4 cascades for the rest
    ASSERT_FALSE(isError(call(*app, "set_lighting", values["previous"])));
    EXPECT_FALSE(s.lightingParameters().sunShadows);
    const Json undone = result(call(*app, "get_lighting", obj(R"({"sections": ["parameters"]})")))["parameters"];
    for (const auto& kv : request.object_items())
        EXPECT_EQ(undone[kv.first], initial[kv.first]) << kv.first << " after the undo";
    EXPECT_TRUE(isError(call(*app, "set_lighting", obj(R"({"sun_shadows": true, "cascade_interval": [2, 1, 1, 1]})"))));
    EXPECT_FALSE(s.lightingParameters().sunShadows) << "all or nothing";
    ASSERT_FALSE(isError(call(*app, "set_lighting", obj(R"({"sun_shadows": true, "cascade_resolution": 1024,
                                                          "shadow_distance": 100})"))));
    app->runFrames(3, 1.0 / 60);
    const Json c = result(call(*app, "get_lighting", obj(R"({"sections": ["cascades"]})")))["cascades"];
    EXPECT_EQ(c["on"], Json(true)) << c.dump();
    EXPECT_EQ(c["shadow_light"], Json("sun"));
    ASSERT_EQ(c["list"].array_items().size(), 4u) << c.dump();
    EXPECT_GT(c["list"][0]["texel_m"].number_value(), 0);
    EXPECT_GT(c["list"][3]["split_far"].number_value(), 99.0);
    // capture_lighting: one cascade (1024 / 1) and every cascade side by side (4 x 1024 / 2)
    const Json one =
        callFrames(*app, "capture_lighting", obj(R"({"image": "cascade", "cascade": 0, "max_width": 64})"));
    ASSERT_FALSE(isError(one)) << text(one);
    EXPECT_EQ(imageBlock(one)["size"], Json(Json::array{1024, 1024}));
    EXPECT_EQ(imageBlock(one)["cascades"].array_items().size(), 1u);
    EXPECT_LT(imageBlock(one)["stats"]["min"].number_value(), 1.0) << "the card and the plane";
    const Json all = callFrames(*app, "capture_lighting", obj(R"({"image": "cascade", "stats_only": true})"));
    ASSERT_FALSE(isError(all)) << text(all);
    EXPECT_EQ(result(all)["size"], Json(Json::array{2048, 512}));
    EXPECT_EQ(result(all)["cascades"].array_items().size(), 4u);
    for (const char* args : {R"({"image": "cascade", "cascade": 4})", R"({"image": "cascade", "cascade": -1})",
                             R"({"image": "cascade_strip"})"})
        EXPECT_TRUE(isError(callFrames(*app, "capture_lighting", obj(args)))) << args;
    // freeze_cascades
    ASSERT_FALSE(isError(call(*app, "set_lighting_debug", obj(R"({"freeze_cascades": true})"))));
    app->runFrames(1, 1.0 / 60);
    EXPECT_EQ(result(call(*app, "get_lighting", obj(R"({"sections": ["cascades"]})")))["cascades"]["frozen"],
              Json(true));
    ASSERT_FALSE(isError(call(*app, "set_lighting_debug", obj(R"({"freeze_cascades": false})"))));
    // probe_lighting in the card's shadow
    const Json pr = callFrames(*app, "probe_lighting", obj(R"({"position": [0, 0, -1.2]})"));
    ASSERT_FALSE(isError(pr)) << text(pr);
    const Json shadow = result(pr)["shadow"];
    ASSERT_TRUE(shadow["cascade"].is_number()) << shadow.dump();
    EXPECT_LT(shadow["cascade"].number_value(), 0.1) << shadow.dump();
    EXPECT_EQ(shadow["cascade_index"], Json(0)) << shadow.dump();
    const Json lit = result(callFrames(*app, "probe_lighting", obj(R"({"position": [3, 0, 2]})")))["shadow"];
    EXPECT_GT(lit["cascade"].number_value(), 0.9) << lit.dump();
    // The defaults again
    ASSERT_FALSE(isError(call(*app, "set_lighting", obj(R"({"sun_shadows": true, "cascade_resolution": 2048,
                                                          "shadow_distance": 1500})"))));
    s.removeInstanceSet(card);
}

// S36a2: the horizon map's tool keys: the set_lighting keys round-trip, get_lighting.horizon, capture_lighting(horizon)
// (channels r and g, an error without a map), probe_lighting's horizon keys; the ridge pyramid set on the scene
TEST_F(LightingDebugTools, HorizonTools)
{
    Vulkan::RenderScene& s = *app->scene();
    // The horizon map alone (inside a cascade the receivers use G: the cascades shadow the near terrain)
    ASSERT_FALSE(isError(call(*app, "set_lighting", obj(R"({"tone_map": "aces", "sun_shadows": false})"))));
    // Without terrain: no map
    ASSERT_FALSE(isError(call(*app, "set_lighting", obj(R"({"horizon_shadows": true, "horizon_resolution": 128})"))));
    app->runFrames(2, 1.0 / 60);
    EXPECT_TRUE(isError(callFrames(*app, "capture_lighting", obj(R"({"image": "horizon"})"))));
    EXPECT_TRUE(isError(callFrames(*app, "capture_lighting", obj(R"({"image": "luminance", "channel": "g"})"))));
    const Json set = call(*app, "set_lighting",
                          obj(R"({"horizon_extent": 4096, "horizon_rows_per_frame": 32, "horizon_resolution": 256})"));
    ASSERT_FALSE(isError(set)) << text(set);
    EXPECT_EQ(result(set)["previous"]["horizon_resolution"].number_value(), 128);
    for (const char* args :
         {R"({"horizon_resolution": 100})", R"({"horizon_extent": 10})", R"({"horizon_rows_per_frame": 0})"})
        EXPECT_TRUE(isError(call(*app, "set_lighting", obj(args)))) << args;
    ASSERT_FALSE(isError(call(*app, "set_lighting", obj(R"({"horizon_resolution": 128})"))));
    // A 4 km space with the ridge (the pyramid only: the horizon map's source), the camera beside it at sunset
    const Vulkan::TerrainLayout layout = TerrainTestHooks::testLayout(16, 2);
    s.setTerrainPyramid(0, TerrainTestHooks::testPyramid(TerrainTestHooks::mountainRidgeHook, layout));
    TerrainTestHooks::renderUntilUploaded(s);
    Vulkan::EnvironmentSettings env = app->environmentSettings();
    env.sunOverride                 = Vulkan::SunOverride{90, 3};  // from the east (+x), 3 degrees
    app->setEnvironmentSettings(env);
    app->setViewCameraByTool(0, Vec3(1500, 120, 2048.5), lookRotation(Vec3(0.5, -1, 0)), 1, false);
    // The first map: 128 rows at 32 per frame
    app->runFrames(8, 1.0 / 60);
    const Json r =
        callFrames(*app, "capture_lighting", obj(R"({"image": "horizon", "full": true, "stats_only": true})"));
    ASSERT_FALSE(isError(r)) << text(r);
    EXPECT_EQ(result(r)["size"], Json(Json::array{128, 128}));
    EXPECT_EQ(result(r)["channel"], Json("r"));
    EXPECT_GT(result(r)["stats"]["max"].number_value(), result(r)["stats"]["min"].number_value());
    const Json g =
        callFrames(*app, "capture_lighting", obj(R"({"image": "horizon", "channel": "g", "max_width": 64})"));
    ASSERT_FALSE(isError(g)) << text(g);
    EXPECT_EQ(imageBlock(g)["channel"], Json("g"));
    const Json h = result(call(*app, "get_lighting", obj(R"({"sections": ["horizon"]})")))["horizon"];
    EXPECT_EQ(h["valid"], Json(true)) << h.dump();
    EXPECT_EQ(h["resolution"].number_value(), 128);
    EXPECT_GT(h["completed"].number_value(), 0);
    // A point behind the ridge (towards -x) at 3 degrees: in its shadow
    const Json pr = callFrames(*app, "probe_lighting", obj(R"({"pixel": [32, 24]})"));
    ASSERT_FALSE(isError(pr)) << text(pr);
    const Json shadow = result(pr)["shadow"];
    ASSERT_TRUE(shadow["horizon"].is_number()) << shadow.dump();
    EXPECT_LT(shadow["horizon"].number_value(), 0.1) << shadow.dump();
    EXPECT_GT(shadow["horizon_r_above_receiver"].number_value(), 4) << shadow.dump();
    // Back to the defaults of the suite
    env.sunOverride.reset();
    app->setEnvironmentSettings(env);
    ASSERT_FALSE(isError(call(*app, "set_lighting", obj(R"({"horizon_resolution": 1024, "horizon_extent": 32768,
                                                          "horizon_rows_per_frame": 64, "sun_shadows": true})"))));
    s.clearTerrain();
    app->runFrames(1, 1.0 / 60);
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
    // The EV100 is the curve's at the sun's elevation at 9.5 h, the same as get_lighting's
    const Json l          = result(call(*app, "get_lighting", obj(R"({"sections": ["sun", "exposure"]})")));
    const double expected = Vulkan::ExposureSettings().ev100At(float(l["sun"]["elevation_deg"].number_value()));
    EXPECT_NEAR(st["lighting"]["ev100"].number_value(), expected, 1e-4);
    EXPECT_NEAR(l["exposure"]["ev100"].number_value(), expected, 1e-4);
    // The effective exposure follows the EV100 and the multiplier (ACES); the rendered frame's equals it (no change
    // since)
    const double effective = Vulkan::exposureFromEv100(float(l["exposure"]["ev100"].number_value()),
                                                       float(l["exposure"]["multiplier"].number_value()));
    EXPECT_FLOAT_EQ(float(l["exposure"]["effective"].number_value()), float(effective));
    EXPECT_FLOAT_EQ(float(l["exposure"]["frame_effective"].number_value()), float(effective));
}

// Regression: get_lighting's exposure.ev100 and source were the last rendered frame's, curve_ev100 and compensation the
// current settings': right after set_lighting the two did not agree until a frame had rendered
TEST_F(LightingTools, ExposureAgreesRightAfterAChange)
{
    ASSERT_FALSE(isError(call(*app, "set_lighting", obj(R"({"tone_map": "aces"})"))));
    app->runFrames(1, 1.0 / 60);
    ASSERT_FALSE(isError(call(*app, "set_lighting", obj(R"({"exposure_compensation": 2})"))));
    const Json e = result(call(*app, "get_lighting", obj(R"({"sections": ["exposure"]})")))["exposure"];
    EXPECT_NEAR(e["ev100"].number_value(), e["curve_ev100"].number_value() - 2, 1e-4) << e.dump();
    EXPECT_NEAR(e["frame_ev100"].number_value(), e["curve_ev100"].number_value(), 1e-4) << e.dump();
    EXPECT_EQ(e["source"], Json("curve"));
    app->runFrames(1, 1.0 / 60);
    const Json f = result(call(*app, "get_lighting", obj(R"({"sections": ["exposure"]})")))["exposure"];
    EXPECT_NEAR(f["frame_ev100"].number_value(), f["ev100"].number_value(), 1e-4) << f.dump();
}

// S36: the app sends the environment every rendered frame; the scene's frame exposure follows its EV100 and the
// multiplier keeps its meaning (exposure())
TEST_F(LightingTools, EnvironmentReachesTheScene)
{
    EXPECT_TRUE(app->sendsEnvironment());
    app->scene()->setToneMap(Vulkan::ToneMap::Aces);
    app->scene()->setExposure(2);
    app->runFrames(1, 1.0 / 60);
    ASSERT_TRUE(app->scene()->hasEnvironment());
    const float ev100 = app->scene()->environment().ev100;
    EXPECT_NEAR(ev100, app->environmentSettings().exposure.ev100At(45), 1e-3);
    EXPECT_FLOAT_EQ(app->scene()->exposure(), 2.0f);
    EXPECT_FLOAT_EQ(app->scene()->frameExposure(), Vulkan::exposureFromEv100(ev100, 2));
}

// ---------------------------------------------------------------------------------------------------------------------
// The Lighting tab: the debug UI's apps with and without EnvironmentSetup::enabled (an app per test)

template <bool Environment>
class LightingTabSuite : public ::testing::Test
{
   protected:
    void SetUp() override
    {
        if (!Vulkan::Instance::suitableDeviceAvailable()) GTEST_SKIP() << "No Vulkan 1.3 device available";
        app = std::make_unique<EngineApp>(tabSetup(Environment));
    }
    void TearDown() override { app.reset(); }
    std::unique_ptr<EngineApp> app;
};
using LightingTabTest      = LightingTabSuite<true>;
using LightingTabPlainTest = LightingTabSuite<false>;

// S36: the Lighting tab (F5) with its a1a sections open renders headless (no validation error); its controls write:
// a click on the rate button x60 queues setWorldTime (WorldTime::rate after the next frame, an api line in the log),
// the "override sun" checkbox sets EnvironmentSettings::sunOverride; a click on the tab's button selects it
TEST_F(LightingTabTest, RendersAndWrites)
{
    renderLightingTab(*app);

    ImVec2 at;
    ASSERT_TRUE(app->debugUi().itemCenter("lighting/rate x60", at));
    const uint64_t before = engineLog().lastSequence();
    click(*app, at);
    app->runFrames(1, 1.0 / 60);
    EXPECT_EQ(app->world().readGlobal<WorldTime>().rate, 60);
    bool api = false;
    for (const auto& line : engineLog().since(before))
        api = api || (line.category == "api" && line.message.find("set_world_time") != std::string::npos);
    EXPECT_TRUE(api);

    // Time and weather collapsed: the sun's rows move up into the window
    ASSERT_FALSE(app->environmentSettings().sunOverride.has_value());
    Json::object state = app->debugUi().state().object_items();
    state["sections"]  = Json::object{{"lighting/Time and weather", false}, {"lighting/Sun and moon", true}};
    app->debugUi().setState(state);
    app->runFrames(1, 1.0 / 60);
    ASSERT_TRUE(app->debugUi().itemCenter("lighting/override sun", at));
    EXPECT_LT(at.y, ImGui::GetIO().DisplaySize.y);
    click(*app, at);
    EXPECT_TRUE(app->environmentSettings().sunOverride.has_value());

    // The tab button "F5 Lighting" is the fifth on the bar
    app->debugUi().select(DebugTab::Game);
    app->runFrames(2, 1.0 / 60);
    float x = 6;
    for (int i = 0; i < 4; ++i)
        x += ImGui::CalcTextSize(("F" + std::to_string(i + 1) + " " + app->debugUi().tabName(i)).c_str()).x + 2 * 6 + 2;
    click(*app, ImVec2(x + 10, app->debugUi().barHeight() / 2));
    EXPECT_TRUE(app->debugUi().isSelected(DebugTab::Lighting));
}

// Regression (round 1, finding 8): an entity light switched off in the tab stays selected and editable (its row is
// greyed with the state "off") and is switched on again by the same checkbox. The old tab lost the selection and the
// editor with the sink light, so the light could not be switched on from the tab
TEST_F(LightingTabTest, SwitchedOffLightStaysEditable)
{
    World& w          = app->world();
    const Entity lamp = w.create();
    w.get<Transform>(lamp).setPosition(Vec3(0, 1, 0));
    w.add<LightSource>(lamp, LightSource());
    Vulkan::LightingParameters lp = app->scene()->lightingParameters();
    lp.localLights                = true;
    app->scene()->setLightingParameters(lp);
    app->runFrames(1, 1.0 / 60);
    Vulkan::LightingDebugState d;
    d.highlightLight = w.extract().lightOf(lamp).id;
    ASSERT_NE(d.highlightLight, 0u);
    app->scene()->setLightingDebug(d);
    Json::object state = app->debugUi().state().object_items();
    // Only the Local lights section open (the editor inside the 800 x 480 window)
    state["sections"] = Json::object{{"lighting/Time and weather", false},
                                     {"lighting/Sun and moon", false},
                                     {"lighting/Exposure", false},
                                     {"lighting/Local lights", true}};
    app->debugUi().setState(state);
    app->debugUi().select(DebugTab::Lighting);
    app->runFrames(2, 1.0 / 60);
    for (bool on : {false, true})
    {
        ImVec2 at;
        ASSERT_TRUE(app->debugUi().itemCenter("lighting/light on", at)) << "the editor while on = " << !on;
        // The panel scrolled down to the editor (mouse wheel over the panel)
        ImGuiIO& io = ImGui::GetIO();
        for (int k = 0; k < 40 && at.y > io.DisplaySize.y - 20; ++k)
        {
            io.AddMousePosEvent(io.DisplaySize.x - 200, io.DisplaySize.y / 2);
            io.AddMouseWheelEvent(0, -2);
            app->runFrames(1, 1.0 / 60);
            ASSERT_TRUE(app->debugUi().itemCenter("lighting/light on", at));
        }
        ASSERT_LT(at.y, io.DisplaySize.y - 20);
        click(*app, at);
        app->runFrames(2, 1.0 / 60);
        EXPECT_EQ(w.read<LightSource>(lamp).on, on);
        EXPECT_EQ(bool(w.extract().lightOf(lamp)), on);
        EXPECT_EQ(app->scene()->lightingDebug().highlightLight, w.extract().lightOf(lamp).id) << "the selection";
    }
    w.destroy(lamp);
}

// capture_lighting's mip is a value of its request: with the tab's Ambient section open (its preview at mip 0 every
// frame) the capture returns the mip it asked for, and the tab's mip stays (review M5: the mip was scene state the tab
// overwrote each frame, and the call changed the tab's preview)
TEST_F(LightingTabTest, CaptureMipWithTabOpen)
{
    // Only the Ambient section open (its preview on screen)
    Json::object state = app->debugUi().state().object_items();
    state["sections"]  = Json::object{{"lighting/Ambient", true}};
    app->debugUi().setState(state);
    app->debugUi().select(DebugTab::Lighting);
    app->scene()->setToneMap(Vulkan::ToneMap::Aces);
    app->runFrames(2, 1.0 / 60);
    const Json r = callFrames(*app, "capture_lighting", obj(R"({"image": "environment", "mip": 3})"));
    ASSERT_FALSE(isError(r)) << text(r);
    const Json block = imageBlock(r);
    const int n      = app->scene()->lightingParameters().environmentResolution >> 3;
    EXPECT_EQ(block["size"], Json(Json::array{4 * n, 3 * n}));
    EXPECT_EQ(block["mip"], Json(3));
    // The mip is not left behind: a capture without it is mip 0
    const Json r0 = callFrames(*app, "capture_lighting", obj(R"({"image": "environment"})"));
    ASSERT_FALSE(isError(r0)) << text(r0);
    EXPECT_EQ(imageBlock(r0)["size"], Json(Json::array{32 * n, 24 * n}));
    EXPECT_EQ(imageBlock(r0)["mip"], Json(0));
}

// Regression (S36a2 review, finding 7): the horizon map's channel was scene state that the tab's Far terrain section
// set in every frame, so with the section open on R capture_lighting(horizon, channel: g) returned the R image (and
// said "r"). The channel is a value of the request now: the capture returns G, the tab's R preview stays
TEST_F(LightingTabTest, CaptureChannelWithTabOpen)
{
    Vulkan::RenderScene& s = *app->scene();
    s.setTerrainPyramid(
        0, TerrainTestHooks::testPyramid(TerrainTestHooks::mountainRidgeHook, TerrainTestHooks::testLayout(16, 2)));
    TerrainTestHooks::renderUntilUploaded(s);
    Vulkan::EnvironmentSettings env = app->environmentSettings();
    env.sunOverride                 = Vulkan::SunOverride{90, 3};
    app->setEnvironmentSettings(env);
    app->setViewCameraByTool(0, Vec3(1500, 120, 2048.5), lookRotation(Vec3(0.5, -1, 0)), 1, false);
    Json::object state = app->debugUi().state().object_items();
    state["sections"]  = Json::object{{"lighting/Far terrain shadows", true}};
    app->debugUi().setState(state);
    app->debugUi().select(DebugTab::Lighting);
    app->runFrames(4, 1.0 / 60);
    const Json r = callFrames(*app, "capture_lighting", obj(R"({"image": "horizon", "stats_only": true})"));
    ASSERT_FALSE(isError(r)) << text(r);
    const Json g =
        callFrames(*app, "capture_lighting", obj(R"({"image": "horizon", "channel": "g", "stats_only": true})"));
    ASSERT_FALSE(isError(g)) << text(g);
    EXPECT_EQ(result(r)["channel"], Json("r"));
    EXPECT_EQ(result(g)["channel"], Json("g"));
    // G (from cascadeCasterDistance on) is not R (from 64 m) over the ridge's lee side
    EXPECT_NE(result(g)["stats"]["mean"].number_value(), result(r)["stats"]["mean"].number_value());
    EXPECT_LT(result(g)["stats"]["max"].number_value(), result(r)["stats"]["max"].number_value());
}

// S36: the write of the Lighting tab's settings (setEnvironmentSettings, which its compensation slider calls; the
// slider's input itself is not driven here): a compensation of +2 EV lowers the next frame's EV100 by 2 (through the
// environment extract)
TEST_F(LightingTabTest, CompensationWriteLowersEv100)
{
    Json::object state = app->debugUi().state().object_items();
    state["sections"]  = Json::object{{"lighting/Exposure", true}, {"lighting/Ambient", true}};
    app->debugUi().setState(state);
    app->debugUi().select(DebugTab::Lighting);
    app->runFrames(2, 1.0 / 60);
    ASSERT_TRUE(app->scene()->hasEnvironment());
    const float ev = app->scene()->environment().ev100;
    // The settings write of the compensation slider
    Vulkan::EnvironmentSettings settings = app->environmentSettings();
    settings.exposure.compensation       = 2;
    app->setEnvironmentSettings(settings);
    app->runFrames(1, 1.0 / 60);
    EXPECT_NEAR(app->scene()->environment().ev100, ev - 2, 1e-4);
}

// S36a1c: a world without EnvironmentSetup::enabled also gets the environment (the defaults of WorldTime and Weather:
// noon); the multiplier keeps its meaning
TEST_F(LightingTabPlainTest, DefaultEnvironmentReachesTheScene)
{
    app->scene()->setToneMap(Vulkan::ToneMap::Aces);
    app->scene()->setExposure(2);
    app->runFrames(1, 1.0 / 60);
    EXPECT_TRUE(app->sendsEnvironment());
    ASSERT_TRUE(app->scene()->hasEnvironment());
    EXPECT_FLOAT_EQ(app->scene()->exposure(), 2.0f);
    EXPECT_FLOAT_EQ(app->scene()->frameExposure(), Vulkan::exposureFromEv100(app->scene()->environment().ev100, 2));
}

// The Lighting tab renders without WorldTime (the time and weather controls are replaced by a line)
TEST_F(LightingTabPlainTest, Renders)
{
    renderLightingTab(*app);
    ImVec2 at;
    EXPECT_FALSE(app->debugUi().itemCenter("lighting/rate x60", at));
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
    // S36a1b: the debug tools
    for (const auto& [tool, args] :
         std::vector<std::pair<const char*, const char*>>{{"set_lighting_debug", R"({"view": "albedo"})"},
                                                          {"capture_lighting", R"({"image": "luminance"})"},
                                                          {"probe_lighting", R"({"pixel": [1, 1]})"},
                                                          {"screenshot", R"({"lighting_view": "albedo"})"}})
    {
        const Json r = call(app, tool, obj(args));
        EXPECT_TRUE(isError(r)) << tool;
        EXPECT_NE(text(r).find("no device"), std::string::npos) << tool << ": " << text(r);
    }
    // The world state needs no device
    EXPECT_FALSE(isError(call(app, "set_time_of_day", obj(R"({"hours": 20})"))));
    EXPECT_EQ(app.world().readGlobal<WorldTime>().timeOfDay, 20);
    EXPECT_TRUE(result(call(app, "app_status", Json::object{}))["lighting"].is_null());
}

// Regression (round 1, finding 6): capture_lighting's schema offers every image the tool accepts (the hand-written
// enum missed atlas and atlas_tile) and every argument it accepts (layer, light and face were missing)
TEST(LightingToolsNoDevice, CaptureLightingSchemaListsEveryImage)
{
    EngineApp app(lightingSetup(false));
    std::string description;
    Json schema;
    ASSERT_TRUE(app.mcp().toolInfo("capture_lighting", description, schema));
    const Json& props = schema["properties"];
    std::set<std::string> images;
    for (const Json& n : props["image"]["enum"].array_items()) images.insert(n.string_value());
    for (int i = 0; i < int(Vulkan::LightingImage::Count); ++i)
    {
        const std::string name = Vulkan::lightingImageName(Vulkan::LightingImage(i));
        if (Vulkan::LightingImage(i) == Vulkan::LightingImage::CascadeStrip) continue;  // "cascade" without the index
        EXPECT_TRUE(images.count(name)) << name;
        EXPECT_NE(description.find(name), std::string::npos) << name;
    }
    for (const char* key : {"image", "range", "log", "color_map", "full", "max_width", "format", "stats_only", "mip",
                            "cascade", "channel", "slice", "layer", "light", "face"})
        EXPECT_TRUE(props[key].is_object()) << key;
    EXPECT_EQ(props["layer"]["enum"], Json(Json::array{"static", "dynamic"}));
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
    // The client's view (what get_lighting and app_status report) follows the host
    EXPECT_NEAR(app.sampleViewEnvironment().time.timeOfDay, 18, 1e-9);
    EXPECT_NEAR(app.sampleViewEnvironment().weather.cloudiness, 0.9, 1e-12);
    // The client world itself refuses the writes (the host writes it, R-NET-2; the tools never pick it in --coop, and
    // the app has no pure-client mode to call them on one)
    std::string error;
    EXPECT_FALSE(setWorldTime(app.world(), WorldTimeChange{6.0, std::nullopt, std::nullopt}, error));
    EXPECT_NE(error.find("replicated"), std::string::npos) << error;
    WeatherChange c;
    c.rain = 0.5;
    EXPECT_FALSE(setWeather(app.world(), 0, c, error));
    EXPECT_NE(error.find("replicated"), std::string::npos) << error;
    EXPECT_EQ(app.world().readSpaceState<Weather>(0).rain, 0);
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

// S36b: list_lights (paging, near, shadowed and visible filters, entity and scene lights), get_lighting.lights, the
// local light keys of set_lighting and set_lighting_debug's highlight_light
TEST_F(LightingDebugTools, LocalLightTools)
{
    Vulkan::RenderScene& s = *app->scene();
    ASSERT_FALSE(isError(call(*app, "set_lighting", obj(R"({"tone_map": "aces", "local_lights": true})"))));
    World& w = app->world();
    std::vector<Entity> lamps;
    for (int i = 0; i < 4; ++i)
    {
        const Entity e = w.create();
        w.get<Transform>(e).setPosition(Vec3(2.0 * i - 3, 1, 0));
        LightSource l;
        l.castShadows = i == 1;
        l.lumen       = 500 + 100 * i;
        w.add<LightSource>(e, l);
        lamps.push_back(e);
    }
    // A scene light far away (not visible) in a list
    const Vulkan::LightListHandle list = s.addLights(ChunkId{0, 20, 20}, {Vulkan::LightData(), Vulkan::LightData()});
    app->runFrames(3, 1.0 / 60);
    const Json all = result(call(*app, "list_lights", Json::object{}));
    ASSERT_EQ(all["lights"].array_items().size(), 6u) << all.dump();
    EXPECT_EQ(all["total"].number_value(), 6);
    EXPECT_TRUE(all["next"].is_null());
    int entities = 0, visible = 0;
    for (const Json& l : all["lights"].array_items())
    {
        entities += l["entity"].is_null() ? 0 : 1;
        visible += l["visible"].bool_value() ? 1 : 0;
    }
    EXPECT_EQ(entities, 4);
    EXPECT_EQ(visible, 4) << all.dump();
    // Paging
    const Json p1 = result(call(*app, "list_lights", obj(R"({"limit": 4})")));
    ASSERT_EQ(p1["lights"].array_items().size(), 4u);
    ASSERT_TRUE(p1["next"].is_number()) << p1.dump();
    const Json p2 = result(call(*app, "list_lights", Json::object{{"after", p1["next"]}, {"limit", 4}}));
    EXPECT_EQ(p2["lights"].array_items().size(), 2u);
    EXPECT_TRUE(p2["next"].is_null());
    EXPECT_GT(p2["lights"][0]["handle"].number_value(), p1["next"].number_value());
    // Filters
    EXPECT_EQ(result(call(*app, "list_lights", obj(R"({"shadowed": true})")))["lights"].array_items().size(), 1u);
    EXPECT_EQ(result(call(*app, "list_lights", obj(R"({"visible": false})")))["lights"].array_items().size(), 2u);
    const Json near = result(call(*app, "list_lights", obj(R"({"near": {"position": [-12, 1, 0], "radius": 0}})")));
    EXPECT_EQ(near["lights"].array_items().size(), 1u) << "range 10: only the lamp at x = -3" << near.dump();
    for (const char* args :
         {R"({"limit": 0})", R"({"limit": 1001})", R"({"near": {"position": [0, 0]}})",
          R"({"near": {"position": [0, 0, 0], "radius": 1, "r": 2}})", R"({"shadowed": 1})", R"({"page": 2})"})
        EXPECT_TRUE(isError(call(*app, "list_lights", obj(args)))) << args;
    // get_lighting.lights
    const Json l = result(call(*app, "get_lighting", obj(R"({"sections": ["lights"]})")))["lights"];
    EXPECT_EQ(l["on"], Json(true)) << l.dump();
    EXPECT_EQ(l["submitted"].number_value(), 6);
    EXPECT_EQ(l["visible"].number_value(), 4);
    EXPECT_EQ(l["shadowed"].number_value(), 1);
    // set_lighting's local light keys round-trip
    const Json set = call(*app, "set_lighting", obj(R"({"clustered_lights": false, "max_lights": 2})"));
    ASSERT_FALSE(isError(set)) << text(set);
    app->runFrames(2, 1.0 / 60);
    const Json params = result(call(*app, "get_lighting", obj(R"({"sections": ["parameters", "lights"]})")));
    EXPECT_EQ(params["parameters"]["max_lights"].number_value(), 2);
    EXPECT_EQ(params["parameters"]["clustered_lights"], Json(false));
    EXPECT_EQ(params["lights"]["over_budget"].number_value(), 2);
    ASSERT_FALSE(isError(call(*app, "set_lighting", result(set)["previous"])));
    EXPECT_TRUE(isError(call(*app, "set_lighting", obj(R"({"max_lights": 0})"))));
    // highlight_light
    const double h = all["lights"][0]["handle"].number_value();
    ASSERT_FALSE(isError(call(*app, "set_lighting_debug", Json::object{{"highlight_light", h}})));
    EXPECT_EQ(s.lightingDebug().highlightLight, uint32_t(h));
    const Json dbg = result(call(*app, "get_lighting", obj(R"({"sections": ["debug"]})")))["debug"];
    EXPECT_EQ(dbg["highlight_light"].number_value(), h);
    ASSERT_FALSE(isError(call(*app, "set_lighting_debug", obj(R"({"highlight_light": null})"))));
    EXPECT_EQ(s.lightingDebug().highlightLight, 0u);
    // S36b step 3: the clusters (get_lighting.clusters, capture_lighting clusters, probe_lighting's cluster keys)
    app->runFrames(3, 1.0 / 60);
    const Json cl = result(call(*app, "get_lighting", obj(R"({"sections": ["clusters"]})")))["clusters"];
    EXPECT_EQ(cl["grid"], Json(Json::array{16, 9, 24})) << cl.dump();
    EXPECT_GT(cl["max"].number_value(), 0) << cl.dump();
    EXPECT_EQ(cl["clustered"], Json(true));
    const Json heat = callFrames(*app, "capture_lighting", obj(R"({"image": "clusters", "max_width": 64})"));
    ASSERT_FALSE(isError(heat)) << text(heat);
    EXPECT_EQ(imageBlock(heat)["size"], Json(Json::array{256, 144}));
    EXPECT_EQ(imageBlock(heat)["clusters"]["slice"], Json("max"));
    EXPECT_GT(imageBlock(heat)["stats"]["max"].number_value(), 0);
    const Json slice =
        callFrames(*app, "capture_lighting", obj(R"({"image": "clusters", "slice": 3, "stats_only": true})"));
    ASSERT_FALSE(isError(slice)) << text(slice);
    EXPECT_EQ(result(slice)["clusters"]["depth"].array_items().size(), 2u);
    for (const char* args : {R"({"image": "clusters", "slice": 24})", R"({"image": "luminance", "slice": 1})"})
        EXPECT_TRUE(isError(call(*app, "capture_lighting", obj(args)))) << args;
    const Json probe = callFrames(*app, "probe_lighting", obj(R"({"position": [-1, 0, 0]})"));
    ASSERT_FALSE(isError(probe)) << text(probe);
    const Json pr = result(probe);
    EXPECT_TRUE(pr["cluster"].is_number()) << pr.dump();
    EXPECT_GE(pr["cluster_lights"].array_items().size(), 3u) << pr.dump();
    EXPECT_GT(pr["cluster_lights"][0]["radiance"][1].number_value(), 0) << pr.dump();
    // S36b step 4: the shadow atlas (get_lighting.atlas, capture_lighting atlas / atlas_tile, probe's atlas_shadow)
    ASSERT_FALSE(isError(call(*app, "set_lighting", obj(R"({"atlas_resolution": 1024, "atlas_tile": 128})"))));
    app->runFrames(3, 1.0 / 60);
    const Json at = result(call(*app, "get_lighting", obj(R"({"sections": ["atlas"]})")))["atlas"];
    EXPECT_EQ(at["on"], Json(true)) << at.dump();
    EXPECT_EQ(at["tiles_used"].number_value(), 6) << at.dump();
    EXPECT_EQ(at["tiles_total"].number_value(), 64);
    EXPECT_EQ(at["lights_with_tiles"].number_value(), 1);
    const Json shadowed = result(call(*app, "list_lights", obj(R"({"shadowed": true})")))["lights"][0];
    EXPECT_EQ(shadowed["atlas"]["tiles"].number_value(), 6) << shadowed.dump();
    EXPECT_EQ(shadowed["atlas"]["ready"], Json(true)) << shadowed.dump();
    const Json atlas = callFrames(*app, "capture_lighting", obj(R"({"image": "atlas", "max_width": 64})"));
    ASSERT_FALSE(isError(atlas)) << text(atlas);
    EXPECT_EQ(imageBlock(atlas)["size"], Json(Json::array{1024, 1024}));
    EXPECT_EQ(imageBlock(atlas)["tiles"].array_items().size(), 6u);
    EXPECT_EQ(imageBlock(atlas)["layer"], Json("static"));
    const Json tile = callFrames(
        *app, "capture_lighting",
        Json::object{{"image", "atlas_tile"}, {"light", shadowed["handle"]}, {"face", 3}, {"stats_only", true}});
    ASSERT_FALSE(isError(tile)) << text(tile);
    EXPECT_EQ(result(tile)["size"], Json(Json::array{128, 128}));
    EXPECT_LT(result(tile)["stats"]["min"].number_value(), 1.0) << "the floor below the lamp (face -y)";
    for (const char* args : {R"({"image": "atlas_tile", "light": 99999})", R"({"image": "atlas", "layer": "both"})",
                             R"({"image": "atlas", "face": 1})", R"({"image": "luminance", "layer": "static"})"})
        EXPECT_TRUE(isError(call(*app, "capture_lighting", obj(args)))) << args;
    const Json probeAt = result(callFrames(*app, "probe_lighting", obj(R"({"position": [-1, 0, 0]})")));
    EXPECT_TRUE(probeAt["atlas_shadow"].is_object()) << probeAt.dump();
    EXPECT_GE(probeAt["atlas_shadow"]["lights"].number_value(), 1) << probeAt.dump();
    // Cross-key: atlas_tile divides atlas_resolution; all or nothing
    const Json badTile = call(*app, "set_lighting", obj(R"({"atlas_tile": 300, "max_lights": 7})"));
    EXPECT_TRUE(isError(badTile));
    EXPECT_NE(text(badTile).find("atlas_resolution"), std::string::npos) << text(badTile);
    EXPECT_NE(s.lightingParameters().maxLights, 7);
    EXPECT_TRUE(isError(call(*app, "set_lighting", obj(R"({"faces_per_frame": 12})"))));
    ASSERT_FALSE(isError(call(*app, "set_lighting", obj(R"({"atlas_resolution": 4096, "atlas_tile": 512})"))));
    // Round 1, finding 8: an entity light that is off or invalid is listed (rendered false, its state) and can be
    // switched on again with set_component; the filter rendered, after_entity paging, matched_total
    ASSERT_FALSE(isError(call(*app, "set_component",
                              Json::object{{"entity", WorldApi::entityJson(lamps[0])},
                                           {"component", "LightSource"},
                                           {"values", obj(R"({"on": false})")}})));
    w.get<LightSource>(lamps[2]).range = 0;  // invalid, written in code (the API refuses it)
    app->runFrames(1, 1.0 / 60);
    const Json off = result(call(*app, "list_lights", obj(R"({"rendered": false})")));
    ASSERT_EQ(off["lights"].array_items().size(), 2u) << off.dump();
    EXPECT_EQ(off["matched_total"].number_value(), 2);
    EXPECT_EQ(off["total"].number_value(), 6);
    EXPECT_EQ(off["unrendered"].number_value(), 2);
    EXPECT_EQ(off["lights"][0]["entity"]["index"].number_value(), double(lamps[0].index));
    EXPECT_EQ(off["lights"][0]["state"], Json("off"));
    EXPECT_TRUE(off["lights"][0]["handle"].is_null());
    EXPECT_EQ(off["lights"][0]["rendered"], Json(false));
    EXPECT_NE(off["lights"][1]["state"].string_value().find("invalid: range"), std::string::npos) << off.dump();
    EXPECT_EQ(result(call(*app, "list_lights", obj(R"({"rendered": true})")))["lights"].array_items().size(), 4u);
    // Paging across both parts: 4 rendered, then the 2 unrendered (after_entity)
    const Json q1 = result(call(*app, "list_lights", obj(R"({"limit": 5})")));
    ASSERT_EQ(q1["lights"].array_items().size(), 5u);
    EXPECT_EQ(q1["matched_total"].number_value(), 6);
    ASSERT_TRUE(q1["next_entity"].is_number()) << q1.dump();
    const Json q2 = result(call(*app, "list_lights", Json::object{{"after_entity", q1["next_entity"]}, {"limit", 5}}));
    ASSERT_EQ(q2["lights"].array_items().size(), 1u) << q2.dump();
    EXPECT_TRUE(q2["next"].is_null() && q2["next_entity"].is_null());
    EXPECT_TRUE(isError(call(*app, "list_lights", obj(R"({"after": 1, "after_entity": 1})"))));
    ASSERT_FALSE(isError(call(*app, "set_component",
                              Json::object{{"entity", WorldApi::entityJson(lamps[0])},
                                           {"component", "LightSource"},
                                           {"values", obj(R"({"on": true})")}})));
    app->runFrames(1, 1.0 / 60);
    EXPECT_EQ(result(call(*app, "list_lights", obj(R"({"rendered": false})")))["lights"].array_items().size(), 1u);
    for (Entity e : lamps) w.destroy(e);
    s.removeLights(list);
    app->runFrames(2, 1.0 / 60);
    EXPECT_EQ(result(call(*app, "list_lights", Json::object{}))["total"].number_value(), 0);
}
