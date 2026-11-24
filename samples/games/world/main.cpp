/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

/**
 * The world game (docs/engine_design.md §19): the large procedural world prototype (S20: a 32 km space of height tiles
 * at 1 m streamed by the engine, 200k trees in the 8 km area [0, 8192]^2, grass around the camera) on the engine app
 * (EngineApp), rendered in a uniform gray. S44 builds the first gameplay on it.
 *
 * Controls: WASD + mouse (left or middle button) to fly, shift = fast, space = up, F toggles the walk mode, G the GUI,
 * F1-F5 the debug UI's tabs (F2 performance, F4 the console, F5 lighting), F12 a screenshot, ESC quits. T toggles the
 * terraform mode (S22, game policy): the left / right mouse button raises / lowers (1..4: add, subtract, flatten,
 * smooth) the terrain under the mouse cursor, 10 edits per second while held, on the host world (also with --coop);
 * radius and strength in the World window. The MCP tool edit_terrain does the same scriptably.
 *
 * MCP (http://127.0.0.1:8765/mcp, --mcp-port or SAIGA_MCP_PORT, 0 disables it): the engine's world and app tools and
 * the prototype's tools (screenshot, get_state, set_camera, move_camera, set_settings, terrain_query).
 *
 * Benchmarks (§17.7): game_world --benchmark <flyover|forest|terraform> flies a fixed camera path at 1920 x 1080
 * (validation, vsync, GUI and MCP off, time 0, the default world: SAIGA_WORLD_TREES is ignored), writes
 * benchmark_<name>.json to the working directory and exits with 0, or with 1 if the benchmark did not finish, the
 * result could not be written, the rendered size changed or a frame stalled (an old result file is deleted at the
 * start). Terrain residency is completed in every benchmark frame (EngineApp::setCompleteViewEachFrame); its time is
 * reported separately (terrain_stream_ms). terraform (S22) edits the terrain once per frame from a static camera
 * (TerraformBenchmark: one step per frame, no per-frame completeView). --base-spacing 2 (benchmark comparisons) runs
 * the same space at 2 m.
 * --headless runs without a window (agents, servers).
 */

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/util/mcp/McpServer.h"
#include "saiga/engine/app/EngineApp.h"
#include "saiga/engine/app/EngineMain.h"
#include "saiga/engine/lighting/EnvironmentExtract.h"
#include "saiga/engine/lighting/EnvironmentWrites.h"
#include "saiga/engine/terrain/TerrainBrush.h"
#include "saiga/engine/terrain/TerrainSystem.h"
#include "saiga/saiga_buildconfig.h"
#include "saiga/vulkan/scene/RenderScene.h"
#include "saiga/vulkan/world/PrototypeScene.h"
#include "saiga/vulkan/world/WorldBenchmark.h"
#include "saiga/vulkan/world/WorldMcp.h"

#include "games/common/Environment.h"
#include "games/common/PrototypeSpace.h"
#include "games/world/Lamps.h"
#include "games/world/TerraformBenchmark.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace Saiga;
using namespace Saiga::Engine;

namespace
{
// The height hook of the "islands" space (S21): the prototype noise 40 m lower (integer height units of 1/1024 m)
int32_t islandsHeight(uint64_t seed, int64_t x, int64_t z)
{
    return Vulkan::prototypeHeight(seed, x, z) - 40 * 1024;
}

// The game's state outside the World: the prototype's settings and height field and its camera controls (render state,
// not world state)
struct WorldGame
{
    // The trees and settings on the scene; the terrain is the engine's (World::terrain through the render sink)
    Vulkan::PrototypeScene prototype;
    // The prototype's 8 km area: its statistics on a 32 m grid (the start camera, game_get_state)
    Vulkan::PrototypeAreaStats area;
    uint32_t treeSeed  = 1;
    float lastLodRange = -1;
    // The terrain uploads of the measured frames (the benchmark's terrain block)
    int terrainUploads        = 0;
    double terrainUploadBytes = 0;
    std::unique_ptr<Vulkan::WorldBenchmark> bench;
    // S22: the edit schedule of --benchmark terraform
    std::unique_ptr<Game::TerraformBenchmark> terraform;
    std::string benchmarkName;
    bool benchmarkDone = false, benchmarkOk = false, sizeChanged = false;
    // S36b: --lamps N (-1: the default, 400; 0 in the day benchmarks, which keep their baseline world), the lamps
    // spawned, the local lights of the measured frames (sums)
    int lamps = -1, lampsSpawned = 0;
    uint64_t lightFrames = 0, lightsVisible = 0, lightsShadowed = 0, lightsWithoutTiles = 0, atlasStatic = 0,
             atlasDynamic = 0, atlasHits = 0;
    uint32_t clusterMax = 0;
    bool walkMode       = false;
    float eyeHeight     = 1.8f;
    bool fWasDown       = false;
    // The terraform mode (S22; game policy, R-MECH-1): the brush op of the left button (1..4; the right button applies
    // the opposite of add / subtract), its radius and strength, the buttons held (read in the GUI, applied in
    // beforeRender between steps), the time of the last edit
    bool terraformMode = false, tWasDown = false;
    int terraformOp       = 0;
    float terraformRadius = 8, terraformStrength = 0.5f;
    bool leftHeld = false, rightHeld = false;
    std::chrono::steady_clock::time_point lastEdit;
    // The frame interval: from the previous frame's beforeRender to this one's (renderFrame to renderFrame)
    std::chrono::steady_clock::time_point lastBeforeRender;
    bool hasLastBeforeRender         = false;
    double frameWork                 = 0;   // the app's CPU work of the attempts of the frame of lastBeforeRenderFrame
    std::array<double, 3> frameWaits = {};  // the waits (frame slot, acquire, present) of those attempts
    double frameStreamMs             = 0;   // the terrain stream time of those attempts
    int frameUploads                 = 0;   // the terrain tiles sent by those attempts
    // The RenderScene frame of lastBeforeRender: a frame that was not submitted is attempted again with the same number
    // (its interval runs from the first attempt, so a stall stays in the result)
    uint64_t lastBeforeRenderFrame = 0;
    // The engine's steps of the measured frames, taken at the frame boundaries that close them
    int64_t stepSamplesAtStart = -1;
    // The step times of the measured range, collected per frame (the profile keeps only the last 600 steps: at 30 fps
    // the 600 measured frames have about 1200 steps)
    std::vector<double> stepTimes, extractTimes;
    uint64_t stepSamplesSeen = 0;
    bool stepTaken           = false;
    json11::Json::object stepStat, extractStat;
    // The app's gpu profile (the whole frame and RenderScene's sections) of the measured frames, matched by the
    // RenderScene frame number: the measured frames whose sample did not arrive yet, the values by section
    std::set<uint64_t> gpuFramesPending;
    std::map<std::string, std::vector<double>> gpuFrameMs;
    int gpuFramesWaited = 0;
};
WorldGame* g_game = nullptr;

std::string resultFile(const std::string& name)
{
    return "benchmark_" + name + ".json";
}

// The S03 keys of the benchmark's gpu_ms from RenderScene's sections of one frame (gpu/<section> of the app's sample).
// "upload + cull" is the sum of two sections that can overlap (a section begins at TOP_OF_PIPE and ends at
// BOTTOM_OF_PIPE, so cull's begin may precede transfer's end): it can exceed their combined elapsed time by at most the
// shorter section per frame (transfer, a median of about 0.005 ms in both benchmarks; not measured per frame)
std::map<std::string, double> s03Passes(const std::map<std::string, double>& section)
{
    auto get = [&](const char* n)
    {
        auto it = section.find(n);
        return it == section.end() ? 0.0 : it->second;
    };
    return {{"upload + cull", get("transfer") + get("cull")},
            {"terrain + water", get("depth.terrain") + get("scene.terrain")},
            {"trees", get("scene.instances")},
            {"grass", get("scene.grass")},
            // S36a1c: the sky with the cloud layer and the environment pass (the atmosphere, the environment map)
            {"sky", get("scene.sky") + get("scene.clouds") + get("environment")},
            // S36a2: the cascades and the horizon map (the sub-sections are in gpu_frame_ms)
            {"shadows", get("shadow")}};
}

void initWorld(EngineApp& app)
{
    WorldGame& g = *g_game;
    World& w     = app.world();
    // A benchmark always uses the default world (its result is compared with the baseline)
    int treeCount = 200000;
    if (const char* trees = std::getenv("SAIGA_WORLD_TREES"); trees && g.benchmarkName.empty())
        treeCount = std::atoi(trees);
    const auto t0 = std::chrono::steady_clock::now();
    Vulkan::PrototypeArea area;
    area.seed          = w.spaceSeed(0);
    area.baseSpacing   = w.spaceTemplate(0).baseSpacing;
    const auto species = Vulkan::FoliageGenerator::treeSpecies(g.treeSeed);
    const auto trees   = Vulkan::ForestRenderer::placeTrees(area, treeCount, g.treeSeed, species);
    std::cout << "World: " << trees.size() << " trees placed in "
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << " s" << std::endl;
    g.prototype.attach(*app.scene(), trees, g.treeSeed);

    // Start on a hill: next to the highest point of the 8 km area, looking over its centre
    const Vec3 peak = g.area.highest;
    Vec3 center(Vulkan::kPrototypeArea * 0.5, 0, Vulkan::kPrototypeArea * 0.5);
    const Vec3 toPeak(peak(0) - center(0), 0, peak(2) - center(2));
    Vec3 eye  = peak + (toPeak.norm() > 1 ? toPeak.normalized() * 150.0 : Vec3(150, 0, 0));
    eye(1)    = std::max(w.terrainHeight(0, eye(0), eye(2)), peak(1) - 40) + 60;
    center(1) = w.terrainHeight(0, center(0), center(2));
    app.setViewCamera(0, eye, cameraLookRotation(center - eye), radians(double(g.prototype.settings.fovy)));

    // S36b: the lamps (on the host world: they replicate to a --coop client): a grid 250 m in front of the start
    // camera, or for --benchmark night a ring of 2500 m around the area's centre (the flyover path looks across it)
    {
        const bool nightBenchmark = g.benchmarkName == "night" || g.benchmarkName == "night_town";
        const bool dayBenchmark   = !g.benchmarkName.empty() && !nightBenchmark;
        const int count         = g.lamps >= 0 ? g.lamps : dayBenchmark ? 0 : 400;
        World& host             = app.startWorld();
        auto groundAt           = [&host](double x, double z) { return host.terrainHeight(0, x, z); };
        const double sea        = double(Vulkan::kPrototypeSeaLevel) * kHeightUnit;
        int spawned             = 0;
        if (g.benchmarkName == "night")
            spawned = Game::spawnLamps(host, count, Vulkan::kPrototypeArea * 0.5, Vulkan::kPrototypeArea * 0.5, 0, 2500,
                                       groundAt, sea);
        else if (g.benchmarkName == "night_town")
            spawned = Game::spawnLamps(host, count, Vulkan::kPrototypeArea * 0.5, Vulkan::kPrototypeArea * 0.5, 25, 0,
                                       groundAt, sea);
        else
        {
            Vec3 dir = center - eye;
            dir(1)   = 0;
            dir      = dir.norm() > 1 ? Vec3(dir.normalized()) : Vec3(1, 0, 0);
            spawned = Game::spawnLamps(host, count, eye(0) + dir(0) * 250, eye(2) + dir(2) * 250, 25, 0, groundAt, sea);
        }
        if (count > 0) std::cout << "World: " << spawned << " lamps" << std::endl;
        g.lampsSpawned = spawned;
    }

    if (!g.benchmarkName.empty())
    {
        std::string gpu = app.scene()->base().physicalDevice.getProperties().deviceName.data();
        // The ground of the paths: terrain or water, whichever is higher
        World* world = &w;
        auto ground  = [world](double x, double z)
        { return std::max(world->terrainHeight(0, x, z), double(Vulkan::kPrototypeSeaLevel) * kHeightUnit); };
        g.bench = std::make_unique<Vulkan::WorldBenchmark>(g.benchmarkName, ground, gpu);
        if (g.benchmarkName == "terraform")
        {
            // S22: the camera is fixed, so the measured frames run updateView with the patch step under its budget (a
            // game frame); exactly one step per frame (the edits run between steps); the first frame is a complete
            // point
            app.setCompleteViewEachFrame(false);
            app.setFixedFrameTime(w.dt());
            g.terraform = std::make_unique<Game::TerraformBenchmark>(w);
        }
        else
        {
            // Terrain residency is deterministic: every frame completes its view (E9)
            app.setCompleteViewEachFrame(true);
        }
        std::remove(resultFile(g.benchmarkName).c_str());
        std::cout << "Benchmark " << g.benchmarkName << " on " << gpu << " (base spacing "
                  << w.spaceTemplate(0).baseSpacing << " m)" << std::endl;
    }
}

// The terrain point under the mouse cursor (the screen centre without a mouse position): a march along the cursor's
// view ray over terrainHeight of the rendered world (1 m steps, at most 4 km), refined by bisection in the last step;
// false if the ray does not hit
bool terrainUnderCursor(EngineApp& app, Vec3& hit)
{
    const ViewCamera& cam = app.viewCamera();
    const World& w        = app.world();
    // The cursor in normalized device coordinates (x right, y up); ImGui's mouse and display size share window units
    const ImGuiIO& io = ImGui::GetIO();
    double nx = 0, ny = 0;
    if (io.DisplaySize.x > 0 && io.DisplaySize.y > 0 && ImGui::IsMousePosValid(&io.MousePos))
    {
        nx = 2.0 * double(io.MousePos.x) / double(io.DisplaySize.x) - 1.0;
        ny = 1.0 - 2.0 * double(io.MousePos.y) / double(io.DisplaySize.y);
    }
    const double t      = std::tan(cam.fovy * 0.5);
    const double aspect = io.DisplaySize.y > 0 ? double(io.DisplaySize.x) / double(io.DisplaySize.y) : 1.0;
    const Vec3 dir      = (cam.rotation * Vec3(nx * t * aspect, ny * t, -1)).normalized();
    auto below          = [&](const Vec3& q) { return q.y() <= w.terrainHeight(cam.space, q.x(), q.z()); };
    Vec3 p              = cam.position;
    if (below(p)) return false;
    for (int i = 0; i < 4000; ++i)
    {
        const Vec3 q = p + dir;
        if (below(q))
        {
            // p is above, q below the terrain: bisect the 1 m step to a few millimetres
            Vec3 a = p, b = q;
            for (int k = 0; k < 10; ++k)
            {
                const Vec3 m       = (a + b) * 0.5;
                (below(m) ? b : a) = m;
            }
            hit = Vec3(b.x(), w.terrainHeight(cam.space, b.x(), b.z()), b.z());
            return true;
        }
        p = q;
    }
    return false;
}

// The terraform mode's edit of this frame (between steps): on the host world (startWorld; with --coop the rendered
// world is a client and the edit reaches it through replication)
void terraformEdit(EngineApp& app)
{
    WorldGame& g = *g_game;
    if (!g.terraformMode || (!g.leftHeld && !g.rightHeld)) return;
    const auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration<double>(now - g.lastEdit).count() < 0.1) return;
    // Only open-world spaces have a height-map terrain: in an enclosed one nothing is edited (and nothing reported each
    // 0.1 s)
    if (app.world().spaceTemplate(app.viewCamera().space).kind != SpaceKind::OpenWorld) return;
    // Every pick attempt is throttled, also a miss (PR review: a ray above the terrain marched 4000 steps each frame)
    g.lastEdit = now;
    Vec3 hit;
    if (!terrainUnderCursor(app, hit)) return;
    TerrainBrush b;
    b.op = TerrainBrush::Op(g.terraformOp);
    // The right button lowers: subtract for add, add for subtract (flatten and smooth are the same on both)
    if (g.rightHeld && b.op == TerrainBrush::Add)
        b.op = TerrainBrush::Subtract;
    else if (g.rightHeld && b.op == TerrainBrush::Subtract)
        b.op = TerrainBrush::Add;
    b.center            = hit;
    b.radius            = double(g.terraformRadius);
    b.strength          = (b.op == TerrainBrush::Add || b.op == TerrainBrush::Subtract)
                              ? double(g.terraformStrength)
                              : std::min(1.0, double(g.terraformStrength));
    World& host         = app.startWorld();
    const SpaceId space = app.viewCamera().space;
    std::string error;
    if (host.checkTerrainEdit(space, b, &error))
        host.editTerrain(space, b);
    else
        std::cout << "terraform: " << error << std::endl;
}

void beforeRender(EngineApp& app)
{
    WorldGame& g               = *g_game;
    Vulkan::RenderScene& scene = *app.scene();
    // The frame being recorded
    const uint64_t frame = scene.frames() + 1;
    terraformEdit(app);
    // S22: the measured frames of the terraform benchmark edit after the frame's step, before its updateView; once per
    // frame (a frame that was not submitted is attempted again with the same number: no second edit or begin)
    const bool newFrame = !g.hasLastBeforeRender || frame != g.lastBeforeRenderFrame;
    if (newFrame && g.terraform && g.bench && g.bench->measuring(frame) && !g.benchmarkDone)
    {
        if (frame == uint64_t(Vulkan::WorldBenchmark::warmupFrames) + 1) g.terraform->begin(app.world());
        g.terraform->edit(app.world());
    }
    g.prototype.update();
    // The terrain's residency range follows the LOD range (a larger render range would only show coarser terrain)
    if (g.prototype.settings.terrainLodRange != g.lastLodRange)
    {
        g.lastLodRange                = g.prototype.settings.terrainLodRange;
        Engine::TerrainViewSettings v = app.world().terrain().viewSettings();
        // The terraform benchmark raises the floor (S22: its brush areas stay resident)
        const double floor = g.terraform ? Game::TerraformBenchmark::kResidencyRange0 : 400.0;
        v.residencyRange0  = std::max(floor, 1.3 * double(g.lastLodRange));
        app.world().terrain().setViewSettings(v);
    }
    // The app's free camera is this game's camera: the walk mode and the field of view of the settings adjust it as
    // tool writes (also while a tool holds it, S14)
    ViewCamera cam    = app.viewCamera();
    const double fovy = radians(double(g.prototype.settings.fovy));
    bool changed      = cam.fovy != fovy;
    cam.fovy          = fovy;
    if (g.walkMode)
    {
        const double y  = app.world().terrainHeight(cam.space, cam.position(0), cam.position(2)) + double(g.eyeHeight);
        changed         = changed || y != cam.position(1);
        cam.position(1) = y;
    }
    if (changed) app.setViewCameraByTool(cam.space, cam.position, cam.rotation, cam.fovy, false);
    if (g.bench)
    {
        auto pose = g.bench->currentPose(frame);
        app.setViewCamera(0, pose.eye.cast<double>(), cameraLookRotation((pose.target - pose.eye).cast<double>()),
                          fovy);
        if (g.bench->measuring(frame))
        {
            ivec2 size = scene.size();
            if (g.bench->width != 0 && (size(0) != g.bench->width || size(1) != g.bench->height)) g.sizeChanged = true;
            g.bench->width  = size(0);
            g.bench->height = size(1);
        }
    }
    // The app sets the camera of the frame (S36: the wind's phases come with the environment)

    // The CPU times of the previous frame: its interval (until now) and its recording and submit
    const auto now = std::chrono::steady_clock::now();
    // The app's CPU work of the last loop belongs to the frame it attempted: summed over every attempt of a frame (a
    // frame that was not submitted is attempted again with the same number)
    {
        Profiler& fp = app.frameProfile();
        double work  = 0;
        if (g.hasLastBeforeRender && fp.enabled && fp.lastSample(fp.scope("work"), work)) g.frameWork += work;
        // Its waits too (the renderer measures them per attempt)
        if (g.hasLastBeforeRender)
            for (size_t k = 0; k < 3; ++k) g.frameWaits[k] += scene.lastWaitParts()[k];
        // And its terrain stream time (every attempt streams; its completeView ran after its beforeRender)
        if (g.hasLastBeforeRender) g.frameStreamMs += app.lastTerrainStreamMs();
        // And the tiles it sent (an attempt that was not submitted sent its tiles too, a later one skips them)
        if (g.hasLastBeforeRender) g.frameUploads += app.world().terrain().stats().uploadsLastUpdate;
    }
    if (!g.hasLastBeforeRender || frame != g.lastBeforeRenderFrame)
    {
        if (g.bench && g.hasLastBeforeRender)
        {
            Vulkan::WorldBenchmark::Sample sample;
            sample.frame       = frame - 1;
            sample.intervalMs  = std::chrono::duration<double, std::milli>(now - g.lastBeforeRender).count();
            sample.workMs      = scene.lastRecordingMs();
            sample.frameWorkMs = g.frameWork;
            sample.waitPartsMs = g.frameWaits;
            // The previous frame's terrain stream time, summed over its attempts (the interval starts at the first)
            sample.terrainStreamMs = g.frameStreamMs;
            g.bench->addSample(sample);
            if (g.bench->measuring(frame - 1))
            {
                // S36b: the local lights and the shadow atlas of the previous frame (the night benchmark's block), once
                // per measured frame (CodeRabbit #86: counted per attempt, from the frame before the measured one)
                const Vulkan::LightingStatistics& ls = scene.lightingStatistics();
                g.lightFrames++;
                g.lightsVisible += ls.lights.visible;
                g.lightsShadowed += ls.lights.shadowed;
                g.lightsWithoutTiles += ls.lights.withoutTiles;
                g.atlasStatic += ls.atlas.staticRenders;
                g.atlasDynamic += ls.atlas.dynamicRenders;
                g.atlasHits += ls.atlas.hits;
                g.clusterMax = std::max(g.clusterMax, ls.lights.clusterMax);
                g.terrainUploads += g.frameUploads;
                g.terrainUploadBytes += double(scene.terrainAtlasStats().bytesRecorded);
                if (g.terraform) g.terraform->frameDone(app.world(), double(scene.terrainAtlasStats().bytesRecorded));
            }
        }
        g.frameWork             = 0;
        g.frameWaits            = {};
        g.frameStreamMs         = 0;
        g.frameUploads          = 0;
        g.lastBeforeRender      = now;
        g.lastBeforeRenderFrame = frame;
        g.hasLastBeforeRender   = true;
    }

    if (g.bench)
    {
        // Frame k is RenderScene's frame being recorded (W: the last warm-up frame, M1..Mn: the measured frames). In a
        // frame the steps run before beforeRender.
        const uint64_t lastWarm = Vulkan::WorldBenchmark::warmupFrames;
        const uint64_t lastMeas = lastWarm + Vulkan::WorldBenchmark::measuredFrames;
        Profiler& wp            = app.world().profiler();
        // Steps: from after W's steps (beforeRender of W) to after Mn's steps (beforeRender of Mn)
        if (frame == lastWarm)
        {
            g.stepSamplesAtStart = int64_t(wp.sampleCount());
            g.stepSamplesSeen    = wp.sampleCount();
        }
        if (g.stepSamplesAtStart >= 0 && !g.stepTaken && frame > lastWarm && frame <= lastMeas)
        {
            // This frame's steps: the newest samples of the step scope
            const uint64_t n                  = wp.sampleCount() - g.stepSamplesSeen;
            g.stepSamplesSeen                 = wp.sampleCount();
            const std::vector<double> history = wp.scopeHistory(wp.scope("step"));
            const size_t take                 = size_t(std::min<uint64_t>(n, history.size()));
            g.stepTimes.insert(g.stepTimes.end(), history.end() - std::ptrdiff_t(take), history.end());
            // The extract of those steps (S19b; its own scope inside phase 8)
            const std::vector<double> extract = wp.scopeHistory(wp.scope("step/phase:extract/extract"));
            const size_t takeExtract          = size_t(std::min<uint64_t>(n, extract.size()));
            g.extractTimes.insert(g.extractTimes.end(), extract.end() - std::ptrdiff_t(takeExtract), extract.end());
        }
        if (frame == lastMeas && g.stepSamplesAtStart >= 0 && !g.stepTaken)
        {
            const auto s  = Vulkan::WorldBenchmark::summarize(g.stepTimes);
            g.stepStat    = json11::Json::object{{"median", s.median}, {"p95", s.p95}};
            const auto x  = Vulkan::WorldBenchmark::summarize(g.extractTimes);
            g.extractStat = json11::Json::object{{"median", x.median}, {"p95", x.p95}};
            g.stepTaken   = true;
        }
        if (frame > lastWarm && frame <= lastMeas) g.gpuFramesPending.insert(frame);
        // The app's GPU samples of the measured frames (they arrive a few frames later); collected only after the
        // measured frames, outside of the timed recording: the app's log (kGpuLogSize = 600 + 64 frames) still holds
        // every measured frame then, the rest arrive during the drain
        std::vector<std::pair<std::string, double>> sample;
        const bool collect = frame > lastMeas;
        for (auto it = g.gpuFramesPending.begin(); collect && it != g.gpuFramesPending.end();)
        {
            if (!app.gpuFrameSample(*it, sample))
            {
                ++it;
                continue;
            }
            std::map<std::string, double> sections;
            for (const auto& v : sample)
            {
                // gpu and gpu/<section> (the sub-sections are flat names with a dot)
                const std::string& n = v.first;
                if (n == "gpu")
                    g.gpuFrameMs["gpu"].push_back(v.second);
                else if (n.rfind("gpu/", 0) == 0 && n.find('/', 4) == std::string::npos)
                {
                    g.gpuFrameMs[n.substr(4)].push_back(v.second);
                    sections[n.substr(4)] = v.second;
                }
            }
            Vulkan::WorldBenchmark::Sample gpuSample;
            gpuSample.frame     = *it;
            gpuSample.sectionMs = s03Passes(sections);
            g.bench->addSample(gpuSample);
            it = g.gpuFramesPending.erase(it);
        }
        if (frame > lastMeas) g.gpuFramesWaited++;
    }
    // The app's GPU samples of the measured frames, or 16 frames more (a frame without timestamps has none)
    const bool gpuFramesIn = g.gpuFramesPending.empty() || g.gpuFramesWaited > 16;
    if (g.bench && g.bench->finished() && gpuFramesIn && !g.benchmarkDone)
    {
        const std::string file = resultFile(g.benchmarkName);
        if (g.bench->width != 1920 || g.bench->height != 1080)
            std::cout << "Benchmark warning: rendered at " << g.bench->width << " x " << g.bench->height
                      << " instead of 1920 x 1080 (not comparable with the baseline)" << std::endl;
        if (g.sizeChanged) std::cout << "Benchmark error: the rendered size changed during the run" << std::endl;
        const bool complete = g.bench->complete();
        if (!complete) std::cout << "Benchmark error: CPU samples of measured frames are missing" << std::endl;
        json11::Json::object result = g.bench->result().object_items();
        // The engine's numbers of exactly the measured frames (S13): the steps and the command recording and submit
        // (without the waits for the frame slot and the image): cpu_work_ms, RenderScene's lastRecordingMs of the
        // submitted measured frames (the app profile's frame/render/recording has a 0 for an app loop without a frame)
        result["step"]      = g.stepStat;
        result["extract"]   = g.extractStat;
        result["recording"] = result["cpu_work_ms"];
        // The whole GPU frame and RenderScene's sections and sub-sections of the measured frames
        json11::Json::object gpuFrame;
        for (const auto& [section, values] : g.gpuFrameMs)
        {
            const auto s      = Vulkan::WorldBenchmark::summarize(values);
            gpuFrame[section] = json11::Json::object{{"median", s.median}, {"p95", s.p95}};
            // The whole GPU frame also with mean, p99 and max (the summary's GPU row)
            if (section == "gpu")
                gpuFrame[section] = json11::Json::object{
                    {"mean", s.mean}, {"median", s.median}, {"p95", s.p95}, {"p99", s.p99}, {"max", s.max}};
        }
        result["gpu_frame_ms"] = gpuFrame.empty() ? json11::Json() : json11::Json(gpuFrame);
        // S22: the terraform keys
        if (g.terraform)
        {
            g.terraform->end(app.world());
            g.terraform->addResult(result);
        }
        // The terrain (S20): the uploads of the measured frames, the resident tiles and the memory
        {
            const auto& ts = app.world().terrain().stats();
            int resident   = 0;
            for (const auto& l : ts.levels) resident += l.resident;
            const auto& mem   = scene.terrainAtlasStats().memory;
            result["terrain"] = json11::Json::object{{"uploads", g.terrainUploads},
                                                     {"upload_mb", g.terrainUploadBytes / (1 << 20)},
                                                     {"resident_tiles", resident},
                                                     {"cpu_mb", double(ts.cpuBytes + mem.cpuBytes()) / (1 << 20)},
                                                     {"gpu_mb", double(mem.gpuBytes()) / (1 << 20)},
                                                     {"base_spacing", app.world().spaceTemplate(0).baseSpacing}};
        }
        // S36a1c: the lighting of the measured frames (the time and weather are frozen)
        {
            const Engine::EnvironmentSample e    = app.sampleViewEnvironment();
            const Vulkan::LightingParameters& lp = scene.lightingParameters();
            result["lighting"]                   = json11::Json::object{
                                  {"time", e.hasTime ? json11::Json(e.time.timeOfDay) : json11::Json()},
                                  {"cloudiness", e.weather.cloudiness},
                                  {"ev100", double(scene.environment().ev100)},
                                  {"tone_map", scene.parameters().toneMap == Vulkan::ToneMap::Aces ? "aces" : "identity"},
                                  {"features", json11::Json::object{{"atmosphere", scene.physicalLighting()},
                                                                    {"clouds", scene.lightingStatistics().clouds},
                                                                    {"force_full_updates", lp.forceFullUpdates},
                                                                    {"sun_shadows", lp.sunShadows},
                                                                    {"horizon_shadows", lp.horizonShadows},
                                                                    {"cascade_count", lp.cascadeCount},
                                                                    {"cascade_resolution", lp.cascadeResolution},
                                                                    {"shadow_distance", double(lp.shadowDistance)},
                                                                    {"local_lights", lp.localLights},
                                                                    {"clustered_lights", lp.clusteredLights}}}};
            // S36b: the lamps and, per measured frame, the visible and shadowed lights, the atlas's renders and its
            // cache hit rate (hits / (hits + static renders))
            const double n     = double(std::max<uint64_t>(g.lightFrames, 1));
            const double faces = double(g.atlasHits + g.atlasStatic);
            result["local_lights"] =
                json11::Json::object{{"lamps", g.lampsSpawned},
                                     {"visible_mean", double(g.lightsVisible) / n},
                                     {"shadowed_mean", double(g.lightsShadowed) / n},
                                     {"without_tiles_mean", double(g.lightsWithoutTiles) / n},
                                     {"cluster_max", double(g.clusterMax)},
                                     {"atlas_static_renders_mean", double(g.atlasStatic) / n},
                                     {"atlas_dynamic_renders_mean", double(g.atlasDynamic) / n},
                                     {"atlas_hit_rate", faces > 0 ? double(g.atlasHits) / faces : 1.0}};
        }
        const int stalls = result["stalled_frames"].int_value();
        if (stalls > 0)
            std::cout << "Benchmark error: " << stalls
                      << " stalled frames (e.g. the window was minimized or the swap chain recreated)" << std::endl;
        if (result["gpu_ms"].is_null())
            std::cout << "Benchmark warning: no GPU times (timestamps not supported)" << std::endl;
        const bool written = Vulkan::WorldBenchmark::writeResult(file, result);
        // A readable summary; the JSON (the baseline format) is in the file
        // The committed baseline of this GPU, if the source tree is known
        json11::Json baseline;
        if (std::string(SAIGA_PROJECT_SOURCE_DIR).size() > 0)
        {
            std::ifstream in(std::string(SAIGA_PROJECT_SOURCE_DIR) + "/data/tests/benchmarks.json");
            std::stringstream text;
            text << in.rdbuf();
            std::string parseError;
            const json11::Json all = json11::Json::parse(text.str(), parseError);
            baseline               = all[result["gpu"].string_value()][g.benchmarkName];
        }
        std::cout << "Benchmark " << g.benchmarkName << ": " << Vulkan::WorldBenchmark::summaryText(result, baseline)
                  << (g.terraform ? g.terraform->summary() : std::string())
                  << (written ? "Benchmark result: " + file : "Benchmark error: could not write " + file) << std::endl;
        g.benchmarkOk   = written && complete && !g.sizeChanged && stalls == 0 && (!g.terraform || g.terraform->ok());
        g.benchmarkDone = true;
        app.quit(g.benchmarkOk ? 0 : 1);
    }
}

void gui(EngineApp& app)
{
    WorldGame& g = *g_game;
    // F toggles the walk mode (when ImGui does not have the keyboard)
    const bool fDown = !ImGui::GetIO().WantCaptureKeyboard && ImGui::IsKeyDown(ImGuiKey_F);
    if (fDown && !g.fWasDown) g.walkMode = !g.walkMode;
    g.fWasDown = fDown;
    // T toggles the terraform mode; 1..4 select the op; the mouse buttons outside ImGui windows edit (S22)
    const bool keys  = !ImGui::GetIO().WantCaptureKeyboard;
    const bool tDown = keys && ImGui::IsKeyDown(ImGuiKey_T);
    if (tDown && !g.tWasDown) g.terraformMode = !g.terraformMode;
    g.tWasDown = tDown;
    if (g.terraformMode && keys)
    {
        const ImGuiKey digits[4] = {ImGuiKey_1, ImGuiKey_2, ImGuiKey_3, ImGuiKey_4};
        for (int i = 0; i < 4; ++i)
            if (ImGui::IsKeyDown(digits[i])) g.terraformOp = i;
    }
    const bool mouse = g.terraformMode && !ImGui::GetIO().WantCaptureMouse;
    g.leftHeld       = mouse && ImGui::IsMouseDown(ImGuiMouseButton_Left);
    g.rightHeld      = mouse && ImGui::IsMouseDown(ImGuiMouseButton_Right);
    // Below the debug UI's bar, at the left (the debug panel is docked at the right)
    ImGui::SetNextWindowPos(ImVec2(10, app.debugUi().barHeight() + 10), ImGuiCond_FirstUseEver);
    ImGui::Begin("World", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::Checkbox("walk mode (F)", &g.walkMode);
    ImGui::Checkbox("terraform mode (T)", &g.terraformMode);
    if (g.terraformMode)
    {
        const char* ops = "add\0subtract\0flatten\0smooth\0";
        ImGui::Combo("brush (1-4)", &g.terraformOp, ops);
        ImGui::SliderFloat("brush radius", &g.terraformRadius, 1, float(kMaxBrushRadius));
        ImGui::SliderFloat("brush strength", &g.terraformStrength, 0.05f, 8.0f);
        ImGui::TextUnformatted("left: raise / apply, right: lower (10 edits/s)");
    }
    ImGui::SliderFloat("eye height", &g.eyeHeight, 0.5f, 50.0f);
    ImGui::SliderFloat("speed", &app.cameraSpeed(), 1, 200);
    ImGui::SliderFloat("speed fast", &app.cameraSpeedFast(), 10, 2000);
    const Vec3 p = app.viewCamera().position;
    ImGui::Text("camera %.1f %.1f %.1f (%.1f m above ground)", p(0), p(1), p(2),
                p(1) - app.world().terrainHeight(app.viewCamera().space, p(0), p(2)));
    // S21: the space of the view camera (the same pose in another space; a space not ready shows the loading screen)
    {
        World& w             = app.world();
        const ViewCamera cam = app.viewCamera();
        int space            = int(cam.space);
        std::string names;
        for (size_t i = 0; i < w.spaceCount(); ++i) names += w.spaceTemplate(SpaceId(i)).name + '\0';
        if (ImGui::Combo("space", &space, names.c_str()) && SpaceId(space) != cam.space)
            app.setViewCameraByTool(SpaceId(space), cam.position, cam.rotation, cam.fovy, false);
        if (const SpaceLoading* l = app.spaceLoading())
            ImGui::Text("loading '%s': %.0f %%", w.spaceTemplate(l->space).name.c_str(), l->progress * 100);
    }
    if (app.mcp().running()) ImGui::Text("MCP: http://127.0.0.1:%d/mcp", app.mcp().port());
    ImGui::End();
}
}  // namespace

int main(int argc, char* argv[])
{
    WorldGame game;
    g_game = &game;

    // --base-spacing 1|2 (a benchmark comparison flag, S20) and --lamps N (S36b): taken out before the engine's command
    // line
    int baseSpacing = 1;
    std::vector<char*> args;
    // An error of the game's own flags (round 3, nit 8: a flag without its value fell through to the engine's parser,
    // "unknown argument", whose usage names neither flag)
    auto flagError = [](const std::string& message)
    {
        std::cerr << message << "\nUsage: world [--lamps N] [--base-spacing 1|2] [the engine's options]" << std::endl;
        return 1;
    };
    for (int i = 0; i < argc; ++i)
    {
        const std::string a = argv[i];
        if (a == "--lamps" || a == "--base-spacing")
        {
            if (i + 1 >= argc) return flagError(a + ": expected a value");
            const std::string v = argv[++i];
            char* end           = nullptr;
            const long n        = std::strtol(v.c_str(), &end, 10);
            if (v.empty() || *end != 0) return flagError(a + ": expected an integer, got \"" + v + "\"");
            if (a == "--lamps")
            {
                if (n < 0 || n > 100000) return flagError("--lamps: 0..100000");
                game.lamps = int(n);
            }
            else
            {
                if (n != 1 && n != 2) return flagError("--base-spacing: 1 or 2");
                baseSpacing = int(n);
            }
            continue;
        }
        args.push_back(argv[i]);
    }

    EngineAppSetup setup;
    setup.name = "world";
    // The prototype's 32 km space (S20); the statistics of its 8 km area (space 0's seed is the world seed)
    // S21: a second open-world space "islands": the prototype noise lowered by 40 m (the sea makes islands); its
    // pyramid is generated only when it is viewed (the start camera and the benchmark stay in space 0)
    // S23a: both hooks are HookGenerators (the prototype fixture: not integer only)
    Engine::SpaceTemplate islands     = Game::prototypeSpaceTemplate(baseSpacing);
    islands.name                      = "islands";
    islands.generator                 = 2;
    setup.world.config.spaceTemplates = {Game::prototypeSpaceTemplate(baseSpacing), islands};
    setup.world.generators            = [](Engine::GeneratorList& list)
    {
        Game::registerPrototypeGenerator(list);
        list.add(2, std::make_shared<Engine::HookGenerator>("islands", islandsHeight, Engine::HookGrassRule{0.82, 0.4},
                                                            false));
    };
    // S36b: the lamps' meshes and their policy (on while the sun is low), before enableEnvironment adds its systems
    setup.world.renderAssets = Game::lampRenderAssets;
    setup.world.components   = Game::lampComponents;
    setup.scenarios          = Game::lampScenarios;
    setup.world.systems      = [](SystemList& l)
    { l.add(Phase::Gameplay, "lampPolicy", Game::lampPolicySystem, RunOn::Host); };
    // S36: the environment (data/world/environment.json), ACES; the time is 10:00, frozen (the benchmark keeps it)
    Game::enableEnvironment(setup, "world");
    game.area                = Vulkan::prototypeAreaStats(setup.world.seed, 0, 0, Vulkan::kPrototypeArea, 32);
    setup.render.msaaSamples = 4;
    setup.cameraSpeed        = 25;
    setup.cameraSpeedFast    = 250;
    setup.benchmarks         = Vulkan::WorldBenchmark::names();
    setup.mcpInstructions    = Vulkan::worldMcpInstructions();
    EngineApp* appPtr        = nullptr;
    setup.renderHooks.init   = [&](EngineApp& app)
    {
        appPtr             = &app;
        game.benchmarkName = app.options().benchmark;
        initWorld(app);
        // The world prototype's settings and tree counts as a section of the debug UI (S19c)
        app.debugUi().addSection(DebugTab::Engine, "World", [&]() { game.prototype.imgui(); });
    };
    setup.renderHooks.shutdown     = [&](EngineApp&) { game.prototype.detach(); };
    setup.renderHooks.beforeRender = [&](EngineApp& app, double) { beforeRender(app); };
    setup.renderHooks.gui          = [&]() { gui(*appPtr); };
    setup.gameTools                = [&](McpServer& server, EngineApp& app)
    {
        if (!app.scene()) return;
        Vulkan::WorldMcpContext context;
        World* world     = &app.world();
        context.settings = &game.prototype.settings;
        context.stats    = [&game, &app]()
        {
            json11::Json::object o = game.prototype.statsJson().object_items();
            const auto& t          = app.world().terrain().stats();
            int resident           = 0;
            for (const auto& l : t.levels) resident += l.resident;
            o["terrain"] = json11::Json::object{{"resident_tiles", resident},
                                                {"pending", t.pending},
                                                {"cpu_mb", double(t.cpuBytes) / (1 << 20)},
                                                {"last_complete_ms", t.lastCompleteMs}};
            return json11::Json(o);
        };
        // The view camera's space (PR review, S21: space 0 also in the islands)
        context.ground = [world, &app](double x, double z)
        { return world->terrainHeight(app.viewCamera().space, x, z); };
        context.worldSize    = world->spaceTemplate(0).sizeMeters();
        context.seaLevel     = double(Vulkan::kPrototypeSeaLevel) * kHeightUnit;
        context.highestPoint = game.area.highest;
        // The view camera's space like ground (CodeRabbit, S21: space 0's also in the islands): the whole space's from
        // the pyramid once it exists, else the 8 km area's (space 0) or the ground under the camera (other spaces)
        context.heightRange = [world, &game, &app]()
        {
            const ViewCamera& cam = app.viewCamera();
            if (const auto* p = world->terrain().pyramid(cam.space))
                return std::make_pair(double(p->minHeight) * kHeightUnit, double(p->maxHeight) * kHeightUnit);
            if (cam.space == 0) return std::make_pair(game.area.minHeight, game.area.maxHeight);
            const double h = world->terrainHeight(cam.space, cam.position(0), cam.position(2));
            return std::make_pair(h, h);
        };
        context.getCamera = [&app]()
        {
            Vulkan::WorldMcpCamera c;
            c.position = app.viewCamera().position;
            c.rotation = app.viewCamera().rotation;
            c.fovy     = app.viewCamera().fovy;
            return c;
        };
        // A tool write without a hold (game_set_camera moves the camera this game renders from)
        context.setCamera = [&app](const Vulkan::WorldMcpCamera& c)
        { app.setViewCameraByTool(app.viewCamera().space, c.position, c.rotation, c.fovy, false); };
        context.walkMode  = &game.walkMode;
        context.eyeHeight = &game.eyeHeight;
        Vulkan::registerWorldMcpTools(server, context);
    };

    // The benchmark's time (its JSON's lighting.time): a failed write fails the benchmark
    bool timeSet   = true;
    const int code = engineMain(int(args.size()), args.data(), setup, {},
                                [&timeSet](EngineApp& app, const AppOptions& options)
                                {
                                    WorldTimeChange morning;
                                    // S36b: --benchmark night and night_town at 22:00
                                    morning.hours =
                                        options.benchmark == "night" || options.benchmark == "night_town" ? 22.0 : 10.0;
                                    morning.rate  = 0.0;
                                    std::string error;
                                    timeSet = setWorldTime(app.startWorld(), morning, error);
                                    if (!timeSet) std::cout << "The time of day was not set: " << error << std::endl;
                                });
    if (code == 0 && !game.benchmarkName.empty() && !timeSet)
    {
        std::cout << "Benchmark error: the time of day was not set" << std::endl;
        return 1;
    }
    // A window closed before the end of the path
    if (code == 0 && !game.benchmarkName.empty() && !game.benchmarkDone)
    {
        std::cout << "Benchmark error: did not finish" << std::endl;
        return 1;
    }
    return code;
}
