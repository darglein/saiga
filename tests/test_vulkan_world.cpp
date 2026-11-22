/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/scene/GrassPass.h"
#include "saiga/vulkan/scene/PrototypeWorld.h"
#include "saiga/vulkan/scene/RenderScene.h"
#include "saiga/vulkan/window/HeadlessWindow.h"
#include "saiga/vulkan/world/ForestRenderer.h"
#include "saiga/vulkan/world/PrototypeForest.h"
#include "saiga/vulkan/world/PrototypeScene.h"
#include "saiga/vulkan/world/PrototypeTerrain.h"
#include "saiga/vulkan/world/WorldBenchmark.h"
#include "saiga/vulkan/world/WorldCommon.h"
#include "saiga/vulkan/world/WorldMcp.h"
#include "saiga/vulkan/world/WorldSettings.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <thread>

#include "forest_reference.h"
#include "golden_image.h"
#include "mcp_test_helper.h"
#include "render_scene_test_context.h"
#include "terrain_test_hooks.h"
#include "vulkan_test_context.h"

using namespace Saiga;
using namespace Saiga::Vulkan;


TEST(VulkanWorld, FrustumPlanes)
{
    // Camera at the origin looking along -z
    mat4 proj   = reversedZInfinitePerspective(radians(60.0f), 1.0f, 0.1f);
    auto planes = frustumPlanes(proj);
    EXPECT_TRUE(boxInFrustum(planes, vec3(-1, -1, -11), vec3(1, 1, -9)));
    // Behind the camera
    EXPECT_FALSE(boxInFrustum(planes, vec3(-1, -1, 5), vec3(1, 1, 7)));
    // Far to the side
    EXPECT_FALSE(boxInFrustum(planes, vec3(100, -1, -11), vec3(102, 1, -9)));
    // No far plane: very far away is visible
    EXPECT_TRUE(boxInFrustum(planes, vec3(-1, -1, -1e6f), vec3(1, 1, -1e6f + 2)));

    // Reversed depth: the near plane maps to 1, infinity to 0
    vec4 n = proj * vec4(0, 0, -0.1f, 1);
    EXPECT_NEAR(n(2) / n(3), 1.0f, 1e-5f);
    vec4 f = proj * vec4(0, 0, -1e7f, 1);
    EXPECT_NEAR(f(2) / f(3), 0.0f, 1e-6f);
}


TEST(VulkanWorld, SettingsJson)
{
    // The pre-pass is off by default; a value other than the destination's default round-trips
    EXPECT_FALSE(WorldSettings().depthPrePass);
    WorldSettings s;
    s.treeLod1     = 123;
    s.depthPrePass = true;
    WorldSettings r;
    auto unknown = r.fromJson(s.toJson());
    EXPECT_TRUE(unknown.empty());
    EXPECT_EQ(r.treeLod1, 123);
    EXPECT_TRUE(r.depthPrePass);
    // S19a: wireframe, freeze_time and freeze_culling are dropped (the environment's wind phases cover goldens and
    // benchmarks; S36)
    EXPECT_EQ(
        r.fromJson(json11::Json::object{{"wireframe", true}, {"freeze_time", true}, {"freeze_culling", true}}).size(),
        3u);
    EXPECT_TRUE(s.toJson()["wireframe"].is_null());
    EXPECT_TRUE(s.toJson()["freeze_time"].is_null());
    EXPECT_EQ(s.toJson()["shading"].string_value(), "clay");
    EXPECT_TRUE(r.shading == WorldShading::Clay);
    unknown = r.fromJson(json11::Json::object{{"no_such_setting", 1}});
    ASSERT_EQ(unknown.size(), 1u);
    // An unknown shading mode is reported like an unknown key
    unknown = r.fromJson(json11::Json::object{{"shading", "pbr"}});
    ASSERT_EQ(unknown.size(), 1u);
    // Regression: values from MCP were not limited (a tiny grass cell size drew billions of instances)
    r.fromJson(json11::Json::object{{"grass_cell_size", 0.0001}, {"debug_mode", 1e12}, {"fovy", NAN}});
    EXPECT_FLOAT_EQ(r.grassCellSize, 0.1f);
    // Regression: integers above 1000 became 0 instead of the upper limit
    EXPECT_EQ(r.debugMode, 1);
    r.fromJson(json11::Json::object{{"debug_mode", 1001}});
    EXPECT_EQ(r.debugMode, 1);
    r.fromJson(json11::Json::object{{"debug_mode", -5}});
    EXPECT_EQ(r.debugMode, 0);
    EXPECT_FLOAT_EQ(r.fovy, 20.0f);
    // Regression: values of the wrong JSON type were converted silently ("90" -> 0, "true" -> false)
    r.fovy  = 60;
    r.trees = true;
    unknown = r.fromJson(json11::Json::object{{"fovy", "90"}, {"trees", "false"}, {"debug_mode", "1"}});
    EXPECT_EQ(unknown.size(), 3u);
    EXPECT_FLOAT_EQ(r.fovy, 60.0f);
    EXPECT_TRUE(r.trees);
    // Huge numbers are clamped (no undefined float conversion)
    EXPECT_TRUE(r.fromJson(json11::Json::object{{"tree_lod1", 1e300}}).empty());
    EXPECT_FLOAT_EQ(r.treeLod1, 500.0f);
    // The prototype settings of RenderScene follow
    r.grass = false;
    r.wind  = 3;
    EXPECT_FALSE(r.prototype().grass);
    EXPECT_FLOAT_EQ(r.prototype().wind, 3.0f);
}

// S19a: the trees as instance sets, one per 256 m chunk (region level 0), region-relative, in placeTrees order with
// deterministic keys. S20: the placed trees of a PrototypeArea (1 km: 4 x 4 chunks)
TEST(VulkanWorld, PrototypeTreeSets)
{
    PrototypeArea area;
    area.size                   = 1024;
    area.baseSpacing            = 2;
    auto species                = FoliageGenerator::treeSpecies(1);
    const std::vector<MeshId> m = {11, 12, 13};
    auto trees                  = ForestRenderer::placeTrees(area, 3000, 1, species);
    auto sets                   = prototypeTreeSets(trees, species, m);
    size_t total                = 0;
    std::vector<bool> seen(trees.size(), false);
    for (size_t k = 0; k < sets.size(); ++k)
    {
        const auto& s = sets[k];
        EXPECT_EQ(s.region.level, 0);
        EXPECT_EQ(s.region.space, 0);
        EXPECT_GE(s.region.x, 0);
        EXPECT_LT(s.region.x, 4);
        EXPECT_LT(s.region.z, 4);
        if (k > 0)
        {
            EXPECT_TRUE(sets[k - 1].region < s.region);
        }
        uint32_t lastKey = 0;
        for (const StaticInstance& inst : s.instances)
        {
            EXPECT_EQ(inst.key >> 24, 1u);
            const uint32_t i = inst.key & 0xFFFFFFu;
            ASSERT_LT(i, trees.size());
            // placeTrees order inside a set
            EXPECT_TRUE(i >= lastKey);
            lastKey       = i;
            seen[i]       = true;
            const auto& t = trees[i];
            const Vec3 o  = regionOrigin(s.region);
            EXPECT_FLOAT_EQ(float(inst.position[0] + o(0)), t.posYaw(0));
            EXPECT_FLOAT_EQ(inst.position[1], t.posYaw(1));
            EXPECT_FLOAT_EQ(float(inst.position[2] + o(2)), t.posYaw(2));
            EXPECT_GE(inst.position[0], 0);
            EXPECT_LE(inst.position[0], 256);
            EXPECT_EQ(inst.scale, t.scaleSpecies(0));
            EXPECT_EQ(inst.mesh, m[size_t(t.scaleSpecies(1))]);
            // The rotation about y by the yaw: the x axis goes to (cos, 0, -sin)
            const vec3 x = inst.getRotation() * vec3(1, 0, 0);
            EXPECT_NEAR(x(0), std::cos(t.posYaw(3)), 1e-4);
            EXPECT_NEAR(x(2), -std::sin(t.posYaw(3)), 1e-4);
        }
        total += s.instances.size();
    }
    EXPECT_EQ(total, trees.size());
    for (bool b : seen) EXPECT_TRUE(b);
}

// S19a: the tree LOD distances of the settings (each clamped independently) as non-decreasing end distances: the
// prototype's rule d < lod1 -> 0, d < lod2 -> 1, else 2, culled beyond the max distance
TEST(VulkanWorld, TreeLodEnds)
{
    WorldSettings s;
    auto e = prototypeTreeLodEnds(s);
    EXPECT_EQ(e, (std::array<float, 3>{70, 300, 30000}));
    // Crossed: LOD 1 is empty
    s.treeLod1 = 500;
    s.treeLod2 = 50;
    e          = prototypeTreeLodEnds(s);
    EXPECT_EQ(e, (std::array<float, 3>{500, 500, 30000}));
    // Beyond the max distance
    s.treeLod1        = 400;
    s.treeLod2        = 3000;
    s.treeMaxDistance = 500;
    e                 = prototypeTreeLodEnds(s);
    EXPECT_EQ(e, (std::array<float, 3>{400, 500, 500}));
    s.treeLod1 = 600;
    e          = prototypeTreeLodEnds(s);
    EXPECT_EQ(e, (std::array<float, 3>{500, 500, 500}));
    s.wind = 4;
    EXPECT_FLOAT_EQ(prototypeTreePadding(s), 0.1f);
}

// S19a: the color word of the PrototypeFoliage class: the layer (0xFF opaque), AO and wind weight as unorm8
TEST(VulkanWorld, FoliageColorWord)
{
    EXPECT_EQ(packPrototypeFoliageColor(-1, 1, 0), 0x0000FFFFu);
    EXPECT_EQ(packPrototypeFoliageColor(2, 0, 1), 0x00FF0002u);
    EXPECT_EQ((packPrototypeFoliageColor(1, 0.5f, 0.25f) >> 8) & 0xFF, 128u);
    EXPECT_EQ((packPrototypeFoliageColor(1, 0.5f, 0.25f) >> 16) & 0xFF, 64u);
    // The tree meshes: the species' sphere, the end distances and the padding
    auto species = FoliageGenerator::treeSpecies(1);
    MeshLods m   = prototypeTreeMesh(species[0], {70, 300, 30000}, 0.025f);
    ASSERT_EQ(m.lods.size(), 3u);
    EXPECT_EQ(m.lods[2].endDistance, 30000);
    EXPECT_EQ(m.sphereCenter(1), species[0].bounds(0));
    EXPECT_EQ(m.sphereRadius, species[0].bounds(1));
    EXPECT_EQ(m.boundsPadding, 0.025f);
    EXPECT_EQ(m.lods[0].vertices.size(), species[0].lods[0].vertices.size());
#ifdef SAIGA_ASSERTS
    EXPECT_DEATH(packPrototypeFoliageColor(0, 1.5f, 0), "AO must be in");
    EXPECT_DEATH(packPrototypeFoliageColor(0, 0, -0.1f), "wind weight must be in");
#endif
}

TEST(WorldBenchmark, Summarize)
{
    auto s = WorldBenchmark::summarize({});
    EXPECT_EQ(s.median, 0);
    EXPECT_EQ(s.p95, 0);
    s = WorldBenchmark::summarize({3, 1, 2});
    EXPECT_EQ(s.median, 2);
    EXPECT_EQ(s.p95, 3);
    s = WorldBenchmark::summarize({4, 1, 3, 2});
    EXPECT_EQ(s.median, 2.5);
    // 1..100: the nearest rank of 95% is the 95th value
    std::vector<double> v;
    for (int i = 100; i >= 1; --i) v.push_back(i);
    s = WorldBenchmark::summarize(v);
    EXPECT_EQ(s.median, 50.5);
    EXPECT_EQ(s.p95, 95);
    // The tail: mean, p99, max and the mean of the largest 1% (ceil(n / 100) values)
    EXPECT_EQ(s.mean, 50.5);
    EXPECT_EQ(s.p99, 99);
    EXPECT_EQ(s.max, 100);
    EXPECT_EQ(s.worst1Percent, 100);
    v.push_back(200);  // 101 values: the largest 2
    EXPECT_EQ(WorldBenchmark::summarize(v).worst1Percent, 150);
}

// The benchmark's console summary: the stutter lines, the verdict and the baseline comparison
TEST(WorldBenchmark, SummaryText)
{
    using json11::Json;
    auto stat = [](double mean, double median, double p99, double max)
    { return Json::object{{"mean", mean}, {"median", median}, {"p95", p99}, {"p99", p99}, {"max", max}}; };
    Json::object r{
        {"gpu", "GPU"},
        {"build", Json::object{{"type", "Release"}, {"asserts", false}, {"archnative", false}}},
        {"frames", 600},
        {"width", 1920},
        {"height", 1080},
        {"fps", Json::object{{"mean", 400.0}, {"low_1pct", 180.0}}},
        {"long_frames", 300},
        {"stalled_frames", 0},
        {"pacing_p99_ms", 4.8},
        {"alternating", true},
        {"wait_ms", Json::object{{"frame_slot", 0.1}, {"acquire", 1.5}, {"present", 0.05}}},
        {"frame_interval_ms", stat(2.5, 2.4, 5.3, 5.7)},
        {"frame_work_ms", stat(0.2, 0.2, 0.3, 0.4)},
        {"gpu_frame_ms", Json::object{{"gpu", stat(2.0, 2.0, 2.5, 2.7)}, {"scene", stat(1.9, 1.9, 2.4, 2.6)}}},
        {"cpu_ms", Json::object{{"median", 2.4}, {"p95", 5.0}}},
        {"gpu_ms", Json::object{{"trees", Json::object{{"median", 1.5}, {"p95", 2.0}}}}},
    };
    std::string text = WorldBenchmark::summaryText(r);
    EXPECT_NE(text.find("max frame 5.70 ms"), std::string::npos) << text;
    // Alternating intervals: uneven pacing, not single stutters
    EXPECT_NE(text.find("uneven frame pacing"), std::string::npos) << text;
    // GPU busy 80% of the frame time: mostly GPU-bound, the largest wait is named
    EXPECT_NE(text.find("mostly GPU-bound; the CPU waits mostly for the swap chain image acquire"), std::string::npos)
        << text;
    EXPECT_EQ(text.find("vs baseline"), std::string::npos) << text;
    // Many long frames without alternation
    r["alternating"] = false;
    text             = WorldBenchmark::summaryText(r);
    EXPECT_NE(text.find("frequent long frames"), std::string::npos) << text;
    // A baseline of the same build: the changes; of another build: the warning first
    Json::object base{{"build", r["build"]},
                      {"cpu_ms", Json::object{{"median", 2.0}}},
                      {"gpu_ms", Json::object{{"trees", Json::object{{"median", 1.0}}}}}};
    text = WorldBenchmark::summaryText(r, base);
    EXPECT_NE(text.find("frame interval median +20.0%"), std::string::npos) << text;
    EXPECT_NE(text.find("trees +50.0%"), std::string::npos) << text;
    EXPECT_EQ(text.find("not comparable"), std::string::npos) << text;
    base["build"] = Json::object{{"type", "Release"}, {"asserts", true}, {"archnative", false}};
    text          = WorldBenchmark::summaryText(r, base);
    EXPECT_LT(text.find("baseline not comparable"), text.find("vs baseline")) << text;
    // Regression (CodeRabbit): no measured wait names no wait as the limit
    {
        Json::object idle    = r;
        idle["gpu_frame_ms"] = Json::object{{"gpu", stat(0.5, 0.5, 0.6, 0.7)}};
        idle["wait_ms"]      = Json::object{{"frame_slot", 0.0}, {"acquire", 0.0}, {"present", 0.0}};
        const std::string t  = WorldBenchmark::summaryText(idle);
        EXPECT_NE(t.find("limit undetermined"), std::string::npos) << t;
    }
    // Without GPU times (no timestamps)
    r["gpu_frame_ms"] = Json();
    r["gpu_ms"]       = Json();
    // Regression (CodeRabbit): missing GPU times were a "GPU busy 0%" verdict
    text = WorldBenchmark::summaryText(r);
    EXPECT_NE(text.find("no GPU times"), std::string::npos) << text;
    EXPECT_EQ(text.find("GPU busy 0%"), std::string::npos) << text;
}

TEST(WorldBenchmark, Paths)
{
    // S20: the paths over a ground callback around the centre (4096, 4096) of the prototype's 8 km area
    auto ground    = [](double x, double z) { return 20 + 0.01 * x - 0.02 * z + 5 * std::sin(x / 300); };
    const double c = 4096;
    const int n    = WorldBenchmark::measuredFrames;
    for (int i : {0, 150, 300, 599})
    {
        auto pose = WorldBenchmark::pose("flyover", ground, i);
        // On the 3 km circle around the center, 300 m above the ground below, looking at the center on the ground
        EXPECT_NEAR(std::hypot(pose.eye(0) - c, pose.eye(2) - c), 3000.0f, 0.01f);
        EXPECT_NEAR(pose.eye(1), ground(pose.eye(0), pose.eye(2)) + 300.0, 1e-3);
        EXPECT_NEAR(pose.target(0), c, 1e-3f);
        EXPECT_NEAR(pose.target(2), c, 1e-3f);
        EXPECT_NEAR(pose.target(1), ground(c, c), 1e-3);
    }
    // One revolution in measuredFrames frames
    auto first = WorldBenchmark::pose("flyover", ground, 0), full = WorldBenchmark::pose("flyover", ground, n);
    EXPECT_NEAR((first.eye - full.eye).norm(), 0.0f, 0.05f);
    EXPECT_GT((first.eye - WorldBenchmark::pose("flyover", ground, n / 2).eye).norm(), 5999.0f);

    auto start = WorldBenchmark::pose("forest", ground, 0), end = WorldBenchmark::pose("forest", ground, n - 1);
    EXPECT_NEAR(start.eye(0), c - 300.0, 1e-3);
    EXPECT_NEAR(end.eye(0), c + 300.0, 1e-3);
    for (int i : {0, 200, n - 1})
    {
        auto pose = WorldBenchmark::pose("forest", ground, i);
        EXPECT_NEAR(pose.eye(2), c, 1e-3f);
        EXPECT_NEAR(pose.eye(1), ground(pose.eye(0), c) + 2.0, 1e-3);
        // Looking along +x
        vec3 dir = (pose.target - pose.eye).normalized();
        EXPECT_NEAR(dir(0), 1.0f, 1e-6f);
    }
    EXPECT_TRUE(WorldBenchmark::known("forest"));
    EXPECT_FALSE(WorldBenchmark::known("forests"));
}

// Regression: a minimized window or a swap chain recreation during the measured frames inflated p95 unnoticed
TEST(WorldBenchmark, CountStalls)
{
    std::vector<double> v(600, 10.0);
    EXPECT_EQ(WorldBenchmark::countStalls(v), 0);
    v[100] = 2000;  // minimized for 2 s
    v[200] = 45;    // 4.5 x the median, but not longer than 50 ms
    EXPECT_EQ(WorldBenchmark::countStalls(v), 1);
    // A slow device: 4 x the median counts, not the 50 ms
    std::vector<double> slow(600, 200.0);
    slow[5] = 700;
    EXPECT_EQ(WorldBenchmark::countStalls(slow), 0);
    slow[6] = 900;
    EXPECT_EQ(WorldBenchmark::countStalls(slow), 1);
    EXPECT_EQ(WorldBenchmark::countStalls({}), 0);
}

// S19a: the benchmark's own samples (RenderScene frame numbers): CPU times (interval, work) and GPU times by the S03
// keys arrive separately; only the measured frames count
TEST(WorldBenchmark, ResultJson)
{
    auto ground = [](double, double) { return 10.0; };
    WorldBenchmark bench("forest", ground, "test gpu");
    const int warmup = WorldBenchmark::warmupFrames, n = WorldBenchmark::measuredFrames;
    EXPECT_FALSE(bench.measuring(uint64_t(warmup)));
    EXPECT_TRUE(bench.measuring(uint64_t(warmup + 1)));
    EXPECT_FALSE(bench.measuring(uint64_t(warmup + n + 1)));
    // The warm-up frames use the first pose, the measured frames walk the path
    const auto p0 = bench.currentPose(1), p1 = bench.currentPose(uint64_t(warmup + 1));
    EXPECT_EQ((p0.eye - p1.eye).norm(), 0.0f);
    EXPECT_GT((bench.currentPose(uint64_t(warmup + n)).eye - p1.eye).norm(), 1.0f);

    // Frame f has the interval f ms (delivered with the next frame), the GPU times of frame f arrive 2 frames later:
    // key k has (k + 1) * f / 100 ms
    const auto& keys = WorldBenchmark::gpuKeys();
    ASSERT_EQ(keys.size(), 5u);
    for (int frame = 1; frame <= warmup + n + 2; ++frame)
    {
        EXPECT_FALSE(bench.finished()) << frame;
        WorldBenchmark::Sample cpu;
        cpu.frame      = uint64_t(frame - 1);
        cpu.intervalMs = frame - 1;
        cpu.workMs     = (frame - 1) * 0.5;
        if (frame > 1) bench.addSample(cpu);
        if (frame > 2)
        {
            WorldBenchmark::Sample gpu;
            gpu.frame = uint64_t(frame - 2);
            for (size_t k = 0; k < keys.size(); ++k) gpu.sectionMs[keys[k]] = double(k + 1) * (frame - 2) / 100.0;
            bench.addSample(gpu);
            // A repeated part is ignored
            bench.addSample(gpu);
        }
    }
    EXPECT_TRUE(bench.finished());

    auto r = bench.result();
    EXPECT_EQ(r["gpu"].string_value(), "test gpu");
    EXPECT_EQ(r["quality"].string_value(), "default");
    // Only the measured frames 61..660: the warm-up frames are excluded
    EXPECT_EQ(r["frames"].int_value(), n);
    EXPECT_EQ(r["gpu_frames"].int_value(), n);
    EXPECT_DOUBLE_EQ(r["cpu_ms"]["median"].number_value(), warmup + 300.5);
    EXPECT_DOUBLE_EQ(r["cpu_ms"]["p95"].number_value(), warmup + 570);
    EXPECT_DOUBLE_EQ(r["cpu_work_ms"]["median"].number_value(), (warmup + 300.5) * 0.5);
    EXPECT_TRUE(r["build"]["type"].is_string());
    EXPECT_TRUE(r["build"]["asserts"].is_bool());
    EXPECT_TRUE(r["build"]["archnative"].is_bool());
    EXPECT_EQ(r["stalled_frames"].int_value(), 0);
    for (size_t k = 0; k < keys.size(); ++k)
    {
        auto pass = r["gpu_ms"][keys[k]];
        ASSERT_TRUE(pass.is_object()) << keys[k];
        EXPECT_NEAR(pass["median"].number_value(), double(k + 1) * (warmup + 300.5) / 100.0, 1e-9);
    }

    // Without GPU samples (no timestamps) gpu_ms is null instead of zeros that look like a fast result
    WorldBenchmark noGpu("forest", ground, "test gpu");
    for (int frame = 1; frame <= warmup + n; ++frame)
    {
        WorldBenchmark::Sample t;
        t.frame      = uint64_t(frame);
        t.intervalMs = frame;
        noGpu.addSample(t);
    }
    EXPECT_TRUE(noGpu.result()["gpu_ms"].is_null());
    EXPECT_EQ(noGpu.result()["gpu_frames"].int_value(), 0);
}

// S20: the terrain stream time of a frame is not part of its interval (cpu_ms, frame_interval_ms, stalled_frames: a
// 60 ms stream on a 3 ms frame is no stall); it is reported as terrain_stream_ms and wait_ms.terrain_stream; the old
// aggregate initializers of Sample keep their meaning; the ground-callback paths
TEST(WorldBenchmark, TerrainStreamTime)
{
    auto ground = [](double x, double z) { return 0.01 * x + 0.02 * z; };
    WorldBenchmark bench("flyover", ground, "test gpu");
    const int warmup = WorldBenchmark::warmupFrames, n = WorldBenchmark::measuredFrames;
    for (int frame = 1; frame <= warmup + n + 2; ++frame)
    {
        WorldBenchmark::Sample cpu{uint64_t(frame - 1), 3.0 + (frame % 7 == 0 ? 60.0 : 0.0), 1.0, {}};
        cpu.terrainStreamMs = frame % 7 == 0 ? 60.0 : 0.0;
        if (frame > 1) bench.addSample(cpu);
        if (frame > 2) bench.addSample(WorldBenchmark::Sample{uint64_t(frame - 2), -1, -1, {{"trees", 1.0}}});
    }
    ASSERT_TRUE(bench.finished());
    const auto r = bench.result();
    EXPECT_DOUBLE_EQ(r["cpu_ms"]["median"].number_value(), 3.0);
    EXPECT_DOUBLE_EQ(r["frame_interval_ms"]["max"].number_value(), 3.0);
    EXPECT_EQ(r["stalled_frames"].int_value(), 0);
    EXPECT_EQ(r["long_frames"].int_value(), 0);
    EXPECT_DOUBLE_EQ(r["terrain_stream_ms"]["max"].number_value(), 60.0);
    EXPECT_NEAR(r["wait_ms"]["terrain_stream"].number_value(), 60.0 / 7, 0.5);
    EXPECT_TRUE(r["terrain"].is_null());
    bench.setTerrainInfo(json11::Json::object{{"resident_tiles", 12}});
    EXPECT_EQ(bench.result()["terrain"]["resident_tiles"].int_value(), 12);
    EXPECT_NE(WorldBenchmark::summaryText(bench.result()).find("terrain stream"), std::string::npos);
    // The paths over the ground: the flyover 300 m above it, around (4096, 4096)
    const auto p = WorldBenchmark::pose("flyover", ground, 0);
    EXPECT_NEAR(p.eye(0), 4096 + 3000, 1e-3);
    EXPECT_NEAR(p.eye(1), ground(7096, 4096) + 300, 1e-3);
    EXPECT_NEAR(p.target(1), ground(4096, 4096), 1e-3);
    const auto f = WorldBenchmark::pose("forest", ground, 0);
    EXPECT_NEAR(f.eye(1), ground(4096 - 300, 4096) + 2, 1e-3);
}

// The prototype's hook and its surface (S20)
TEST(PrototypeTerrain, SurfaceHeightMatchesRule)
{
    for (int bs : {1, 2})
        for (int k = 0; k < 200; ++k)
        {
            const double x = 1000 + k * 37.37, z = 2000 + k * 13.13;
            const double gx = x / bs, gz = z / bs;
            const int64_t i = int64_t(std::floor(gx)), j = int64_t(std::floor(gz));
            auto h                = [&](int64_t a, int64_t b) { return prototypeHeight(7, a * bs, b * bs); };
            const double expected = terrainSurfaceDouble(h(i, j), h(i + 1, j), h(i, j + 1), h(i + 1, j + 1),
                                                         gx - double(i), gz - double(j)) *
                                    kHeightUnit;
            EXPECT_EQ(prototypeSurfaceHeight(7, bs, x, z), expected);
        }
    // Integer metres: the sample itself; clamped to the space at its edges
    EXPECT_EQ(prototypeSurfaceHeight(3, 1, 500, 700), prototypeHeight(3, 500, 700) * kHeightUnit);
    EXPECT_EQ(prototypeSurfaceHeight(3, 2, -50, 700), prototypeSurfaceHeight(3, 2, 0, 700));
    // The space ends in the water: below the sea level at the edges of the 32 km space
    EXPECT_LT(prototypeHeight(1, 0, 16000), kPrototypeSeaLevel);
    EXPECT_LT(prototypeHeight(1, 32768, 16000), kPrototypeSeaLevel);
    // The hook is clamped and deterministic
    EXPECT_EQ(prototypeHeight(5, 4096, 4096), prototypeHeight(5, 4096, 4096));
}

// The trees of a PrototypeArea stand on the surface (exactly: the float of prototypeSurfaceHeight - 0.25), above the
// sea, inside the area, deterministic
TEST(PrototypeTerrain, TreePlacement)
{
    PrototypeArea area;
    area.x0            = 1024;
    area.z0            = 1024;
    area.size          = 1024;
    area.seed          = 1;
    area.baseSpacing   = 2;
    const auto species = FoliageGenerator::treeSpecies(1);
    const auto trees   = ForestRenderer::placeTrees(area, 3000, 1, species);
    ASSERT_GT(trees.size(), 1000u);
    EXPECT_LE(trees.size(), 3000u);
    for (const auto& t : trees)
    {
        const float x = t.posYaw(0), z = t.posYaw(2);
        ASSERT_GE(x, 1024.f);
        ASSERT_LE(x, 2048.f);
        ASSERT_EQ(t.posYaw(1), float(prototypeSurfaceHeight(1, 2, double(x), double(z))) - 0.25f);
        ASSERT_GT(double(t.posYaw(1)) + 0.25, double(kPrototypeSeaLevel) * kHeightUnit);
    }
    const auto again = ForestRenderer::placeTrees(area, 3000, 1, species);
    ASSERT_EQ(again.size(), trees.size());
    EXPECT_EQ(again.back().posYaw, trees.back().posYaw);
}

TEST(PrototypeTerrain, HighestPoint)
{
    const PrototypeAreaStats s = prototypeAreaStats(1, 0, 0, 8192, 64);
    EXPECT_GT(s.maxHeight, 100);
    EXPECT_LT(s.minHeight, 6);
    EXPECT_EQ(s.highest(1), s.maxHeight);
    EXPECT_EQ(prototypeHeight(1, int64_t(s.highest(0)), int64_t(s.highest(2))) * kHeightUnit, s.maxHeight);
    // A finer grid finds at least as high a point
    EXPECT_GE(prototypeAreaStats(1, 0, 0, 8192, 32).maxHeight, s.maxHeight);
}

// Regression (S19a review): finished() only proves that the CPU samples reached the last measured frame; a run that
// missed one is not complete (game_world reports it as a failure)
TEST(WorldBenchmark, IncompleteRun)
{
    auto ground = [](double, double) { return 10.0; };
    WorldBenchmark bench("forest", ground, "test gpu");
    const int warmup = WorldBenchmark::warmupFrames, n = WorldBenchmark::measuredFrames;
    for (int frame = 1; frame <= warmup + n; ++frame)
    {
        if (frame == warmup + 40) continue;
        WorldBenchmark::Sample t;
        t.frame      = uint64_t(frame);
        t.intervalMs = 1;
        t.workMs     = 0.5;
        t.sectionMs  = {{"trees", 1.0}};
        bench.addSample(t);
    }
    EXPECT_TRUE(bench.finished());
    EXPECT_FALSE(bench.complete());
    EXPECT_EQ(bench.result()["frames"].int_value(), n - 1);
}

using namespace ForestReferenceTests;

// ---------------------------------------------------------------------------------------------------------------------
// S14: the game tools of the world prototype (game_world) through handleMessage

namespace
{
McpServer::Json callWorldTool(McpServer& server, const std::string& name, const McpServer::Json& args)
{
    Poller poller(server);
    return server.handleMessage(McpRequest("tools/call", McpServer::Json::object{{"name", name}, {"arguments", args}}));
}
McpServer::Json toolResultJson(const McpServer::Json& r)
{
    std::string err;
    return McpServer::Json::parse(r["result"]["content"][0]["text"].string_value(), err);
}
}  // namespace


// ---------------------------------------------------------------------------------------------------------------------
// S20: the prototype on the new path (PrototypeScene: the environment and the trees; the terrain as test tiles of the
// prototype hook through fillTestTerrain, no engine)

// RenderSceneTest (96 x 64, 4x MSAA) with the prototype: trees by placeTrees over the 1 km area [1024, 2048]^2, the
// prototype hook at spacing 2 in a 16-chunk layout, the sea level 6 m
class VulkanPrototypeScene : public RenderSceneTest
{
   protected:
    // Not about shadows (render_scene_test_context.h)
    bool Shadows() const override { return false; }
    static constexpr int kSpacing = 2;
    TerrainLayout Layout() const { return TerrainTestHooks::testLayout(16, kSpacing); }

    void Attach(PrototypeScene& p, int treeCount, uint32_t seed = 1)
    {
        attachedSeed = seed;
        PrototypeArea area;
        area.x0                    = 1024;
        area.z0                    = 1024;
        area.size                  = 1024;
        area.seed                  = seed;
        area.baseSpacing           = kSpacing;
        species                    = FoliageGenerator::treeSpecies(seed);
        trees                      = ForestRenderer::placeTrees(area, treeCount, seed, species);
        p.settings.terrainLodRange = 40;
        p.attach(*scene, trees, seed);
    }
    // The terrain around the camera, with grass masks: the tiles of level k within 100 m * 2^k (the default residency
    // rule has 400 m * 2^k; test speed: a quarter of the tiles to upload, and coarser nodes beyond them, i.e. fewer
    // triangles for lavapipe's vertex stage; no test here looks at the distant terrain's detail)
    void Terrain(const Vec3& camera, bool withMasks = false)
    {
        TerrainTestHooks::fillTestTerrain(*scene, Layout(), prototypeHeight, attachedSeed, kPrototypeSeaLevel, camera,
                                          withMasks, 0, 100);
    }
    double Ground(double x, double z) const { return prototypeSurfaceHeight(attachedSeed, kSpacing, x, z); }
    double Sea() const { return double(kPrototypeSeaLevel) * kHeightUnit; }

    // The camera (the settings' fovy) at eye looking at target
    void Look(const PrototypeScene& p, const Vec3& eye, const Vec3& target)
    {
        SetCamera(0, eye, viewRotation(lookAt(eye.cast<float>(), target.cast<float>(), vec3(0, 1, 0))),
                  double(radians(p.settings.fovy)));
    }
    // A frame with the settings applied first
    TemplatedImage<ucvec4> Frame(PrototypeScene& p)
    {
        p.update();
        return RenderFrame();
    }
    // The terrain of the view, then n frames (the last returned, waited for)
    TemplatedImage<ucvec4> Shot(PrototypeScene& p, const Vec3& eye, const Vec3& target, int n = 3,
                                bool withMasks = false)
    {
        Look(p, eye, target);
        p.update();
        Terrain(eye, withMasks);
        TemplatedImage<ucvec4> img;
        for (int i = 0; i < n; ++i) img = Frame(p);
        scene->waitIdle();
        return img;
    }
    ForestReference Reference(const PrototypeScene& p) const
    {
        return forestReference(trees, species, p.settings, cameraEye, cameraRotation, double(width) / height);
    }

    uint32_t attachedSeed = 1;
    std::vector<TreeSpecies> species;
    std::vector<ForestRenderer::TreeInstance> trees;
};

namespace
{
int pixelsDiffering(const TemplatedImage<ucvec4>& a, const TemplatedImage<ucvec4>& b, int threshold = 0)
{
    if (a.w == 0 || a.h == 0 || a.w != b.w || a.h != b.h)
    {
        ADD_FAILURE() << "pixelsDiffering: images of " << a.w << "x" << a.h << " and " << b.w << "x" << b.h;
        return std::numeric_limits<int>::max();
    }
    int d = 0;
    for (int y = 0; y < a.h; ++y)
        for (int x = 0; x < a.w; ++x)
            for (int c = 0; c < 3; ++c)
                if (std::abs(int(a(y, x)(c)) - int(b(y, x)(c))) > threshold)
                {
                    d++;
                    break;
                }
    return d;
}
}  // namespace

TEST_F(VulkanPrototypeScene, RendersTerrainTreesAndSky)
{
    PrototypeScene p;
    Attach(p, 2000);
    EXPECT_TRUE(scene->hasFoliageTexture());
    // Above the area's centre, looking at the horizon
    Vec3 c(1536, 0, 1536);
    c(1)           = Ground(c(0), c(2));
    const Vec3 eye = c + Vec3(0, 60, 200);
    // Without the sky pass the top of the image is the clear color
    p.settings.sky = false;
    auto noSky     = Shot(p, eye, c + Vec3(0, 20, 0), 1);
    p.settings.sky = true;
    auto img       = Shot(p, eye, c + Vec3(0, 20, 0));
    ASSERT_EQ(img.w, 96);
    // The sky at the top is a uniform gray that differs from the clear color, the terrain at the bottom too
    ucvec4 top = img(1, 48), clearTop = noSky(1, 48), bottom = img(62, 48);
    EXPECT_EQ(top(0), top(1));
    EXPECT_EQ(top(1), top(2));
    EXPECT_GT(std::abs(int(top(0)) - int(clearTop(0))), 2) << "sky " << int(top(0)) << " clear " << int(clearTop(0));
    EXPECT_GT(std::abs(int(bottom(0)) - int(clearTop(0))), 2);
    EXPECT_GT(scene->terrainPassStats().nodes, 0);
    EXPECT_GT(scene->statistics().terrainNodes, 0u);
    const auto fs = p.forest().stats();
    EXPECT_GT(fs.visible[0] + fs.visible[1] + fs.visible[2], 0u);
    EXPECT_GT(fs.triangles, 0u);
    expectForest(fs, Reference(p), "overview");
}

// Regression (S01): far from the origin and close to the ground, dFdx/dFdy of a world-space float position (flat
// shading) was rounding noise. S20: at 30 km on the slopeRidgeHook terrain of a 128-chunk layout, 15 cm above the
// ground with a narrow field of view, the ground in front of the camera is one or two triangles
TEST_F(VulkanPrototypeScene, FlatShadingIsStableFarFromTheOrigin)
{
    PrototypeScene p;
    Attach(p, 10);
    p.settings.grass       = false;
    p.settings.trees       = false;
    p.settings.flatShading = true;
    // No fog (S36: the studio environment's)
    StudioEnvironment noFog = prototypeStudio();
    noFog.fog               = false;
    scene->setStudioEnvironment(noFog);
    p.settings.fovy   = 20;
    const auto layout = TerrainTestHooks::testLayout(128, 1);
    const auto source = TerrainHeightSource::fromHook(TerrainTestHooks::slopeRidgeHook, 1);
    const Vec3 ground(30017.3, 0, 30011.7);
    const double h    = double(TerrainTestHooks::slopeRidgeHook(1, 30017, 30011)) * kHeightUnit;
    const Vec3 eye    = Vec3(ground(0), h + 0.15, ground(2));
    const Vec3 target = eye + Vec3(0.6, -1.0, 0.4);
    Look(p, eye, target);
    TerrainTestHooks::fillTestTerrain(*scene, layout, source, std::nullopt, eye, false);
    // Reference without terrain and water: the check below must see terrain, not a uniform background
    p.settings.terrain = false;
    p.settings.water   = false;
    p.update();
    auto reference     = Capture(1);
    p.settings.terrain = true;
    p.update();
    auto img = Capture(1);
    ASSERT_EQ(img.w, 96);
    int differs = 0, pixels = 0;
    for (int y = 8; y < 56; ++y)
        for (int x = 8; x < 88; ++x)
        {
            differs += std::abs(int(img(y, x)(0)) - int(reference(y, x)(0))) > 2;
            pixels++;
        }
    EXPECT_GT(differs, pixels * 9 / 10) << "the terrain does not cover the image";
    int rough = 0, total = 0;
    for (int y = 8; y < 56; ++y)
        for (int x = 8; x < 87; ++x)
        {
            rough += std::abs(int(img(y, x)(0)) - int(img(y, x + 1)(0))) > 20;
            total++;
        }
    EXPECT_LT(rough, total / 50) << rough << " of " << total << " neighbouring pixels differ by more than 20";
}

// The drawn trees per LOD equal the CPU reference also after the settings change (wind, crossed LOD distances, the
// trees toggle). The counts come from the culling, not from the terrain: no terrain set and no terrain draws (test
// speed: lavapipe's vertex stage)
TEST_F(VulkanPrototypeScene, ForestFollowsSettings)
{
    PrototypeScene p;
    Attach(p, 5000);
    p.settings.treeLod1 = 30;
    p.settings.treeLod2 = 120;
    p.settings.terrain  = false;
    p.settings.water    = false;
    Vec3 eye(1536, 0, 1536);
    eye(1)            = Ground(eye(0), eye(2)) + 40;
    const Vec3 target = eye + Vec3(1, -0.3, 0.2);
    Look(p, eye, target);
    p.update();
    // One frame per check: statistics() is the newest frame's, which already has the changed settings
    auto check = [&](const char* what)
    {
        p.update();
        Capture(1);
        const auto st = p.forest().stats();
        EXPECT_EQ(st.frame, scene->frames()) << what;
        const auto ref = Reference(p);
        expectForest(st, ref, what);
        return st;
    };
    p.settings.wind = 0;
    check("wind 0");
    p.settings.wind = 4;
    check("wind 4");
    // Crossed (LOD 1 empty); LOD 0 to 100 m (test speed: a LOD 0 tree is the expensive mesh)
    p.settings.treeLod1 = 100;
    p.settings.treeLod2 = 50;
    auto crossed        = check("crossed LOD distances");
    EXPECT_EQ(crossed.visible[1], 0u);
    EXPECT_GT(crossed.visible[0], 0u);
    p.settings.trees = false;
    auto off         = check("trees off");
    EXPECT_EQ(off.visible[0] + off.visible[1] + off.visible[2], 0u);
    p.settings.trees = true;
    auto on          = check("trees on again");
    EXPECT_EQ(on.visible, crossed.visible);
}

// The grass toggle (racing turns it off): grass from the masks, no blocks and no grass pixels when off
TEST_F(VulkanPrototypeScene, GrassToggle)
{
    // Compares frames: golden mode
    GoldenMode();
    PrototypeScene p;
    Attach(p, 10);
    p.settings.trees = false;
    // Low above a grassy point (flat, above the sea), looking down
    Vec3 eye(0, 0, 0);
    for (double x = 1100; x < 2000; x += 7)
    {
        const double h = Ground(x, x);
        if (h > Sea() + 2 && std::abs(Ground(x + 2, x) - h) < 0.3 && std::abs(Ground(x, x + 2) - h) < 0.3)
        {
            eye = Vec3(x, h + 1.5, x);
            break;
        }
    }
    ASSERT_GT(eye(0), 0);
    const Vec3 target = eye + Vec3(1, -0.8, 0.3);
    auto withGrass    = Shot(p, eye, target, 2, true);
    EXPECT_GT(scene->grassPass().stats().clumps, 0);
    p.settings.grass = false;
    p.update();
    auto noGrass = Capture(2);
    EXPECT_EQ(scene->grassPass().stats().clumps, 0);
    EXPECT_FALSE(scene->prototypeSettings().grass);
    EXPECT_GT(pixelsDiffering(withGrass, noGrass, 8), 50);
}

// The wind's time comes from the environment's phases (S36; S19a: setTime), not the wall clock: the same phases give
// the same image after a pause, other phases move the wind (trees and grass)
TEST_F(VulkanPrototypeScene, TimeIndependentOfWallClock)
{
    // Compares frames: golden mode
    GoldenMode();
    PrototypeScene p;
    // 1000 trees (a few dozen in view) and the grass show the wind (test speed: 5000 cost 120 ms per frame)
    Attach(p, 1000);
    p.settings.wind = 4;
    // The grass within 20 m (in view, a few metres away; test speed: 60 m were 350k triangles per frame)
    p.settings.grassRadius = 20;
    Vec3 eye(1536 - 28, 0, 1536 - 80);
    eye(1)            = std::max(Ground(eye(0), eye(2)), Sea()) + 2.0;
    const Vec3 target = eye + Vec3(1, -0.2, 0);
    EnvironmentState still;
    scene->setEnvironment(still);
    auto a = Shot(p, eye, target, 2, true);
    // 100 ms of wall clock at wind 4 would move the foliage; one frame each: the scene's state is settled after a
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    auto b = Capture(1);
    EXPECT_EQ(pixelsDiffering(a, b), 0);
    EnvironmentState moved = still;
    for (int i = 0; i < 5; ++i) moved.phases[size_t(i)] = 1.7f * float(i + 1);
    scene->setEnvironment(moved);
    auto c = Capture(1);
    EXPECT_GT(pixelsDiffering(a, c), 0) << "the wind does not move";
}

// The depth is resolved with SAMPLE_ZERO (MSAA) or sampled directly (1 sample), with and without the depth pre-pass:
// validation clean and the same image
TEST_F(VulkanPrototypeScene, DepthResolveAndPrePass)
{
    for (int samples : {4, 1})
    {
        Create(96, 64, samples);
        // Compares frames: golden mode (per scene: Create makes a new one)
        GoldenMode();
        PrototypeScene p;
        Attach(p, 500);
        Vec3 c(1536, 0, 1536);
        c(1)                    = Ground(c(0), c(2));
        p.settings.depthPrePass = true;
        auto with               = Shot(p, c + Vec3(0, 40, 120), c, 2);
        p.settings.depthPrePass = false;
        p.update();
        // The frame right after the switch (the settings are a per-frame snapshot)
        auto without = Capture(1);
        EXPECT_EQ(pixelsDiffering(with, without), 0) << samples << " samples";
    }
}

// RenderScene's sections and sub-sections arrive for every completed frame, in order and without gaps; a sub-section is
// inside its pass (the S19a names are kept: WorldBenchmark::gpuKeys and game_world's s03Passes rely on them)
TEST_F(VulkanPrototypeScene, Sections)
{
    const auto& names                       = RenderScene::gpuSectionNames();
    const std::vector<std::string> expected = {
        "transfer",         "environment",  "cull",      "lights",        "shadow",      "shadow.cascades",
        "shadow.horizon",   "shadow.atlas", "depth",     "depth.terrain", "scene",       "scene.terrain",
        "scene.instances",  "scene.grass",  "scene.sky", "scene.clouds",  "scene.debug", "resolve",
        "lighting.preview", "post",         "imgui"};
    ASSERT_EQ(names.size(), expected.size());
    for (size_t i = 0; i < names.size(); ++i) EXPECT_EQ(names[i], expected[i]);
    PrototypeScene p;
    Attach(p, 500);
    Vec3 c(1536, 0, 1536);
    const Vec3 eye = c + Vec3(0, Ground(c(0), c(2)) + 80, 100);
    Look(p, eye, c);
    p.update();
    Terrain(eye);
    // The uploads' frames are not compared
    FrameTimes t;
    scene->waitIdle();
    while (scene->takeGpuFrameTimes(t))
    {
    }
    const uint64_t first = scene->frames() + 1;
    std::vector<uint64_t> frames;
    for (int i = 0; i < 12; ++i)
    {
        p.settings.depthPrePass = i % 2 == 0;
        Frame(p);
        while (scene->takeGpuFrameTimes(t))
        {
            if (t.frame < first) continue;
            frames.push_back(t.frame);
            ASSERT_EQ(t.gpuSectionMs.size(), names.size());
            for (double ms : t.gpuSectionMs) EXPECT_GE(ms, 0);
            // The sub-sections of a pass (by name: CodeRabbit #84, an index list left out scene.debug) inside it
            for (size_t i = 0; i < names.size(); ++i)
            {
                const std::string pass = names[i];
                if (pass.find('.') != std::string::npos) continue;
                double sub = 0;
                for (size_t j = 0; j < names.size(); ++j)
                    if (std::string(names[j]).rfind(pass + ".", 0) == 0) sub += t.gpuSectionMs[j];
                EXPECT_LE(sub, t.gpuSectionMs[i] + 0.01) << pass;
            }
            EXPECT_GE(t.gpuMs, t.gpuSectionMs[10]);
        }
    }
    scene->waitIdle();
    Frame(p);
    while (scene->takeGpuFrameTimes(t))
        if (t.frame >= first) frames.push_back(t.frame);
    ASSERT_GE(frames.size(), 12u);
    for (size_t i = 0; i < frames.size(); ++i) EXPECT_EQ(frames[i], first + i) << i;
    EXPECT_EQ(scene->lastGpuFrameTimes().frame, frames.back());
}

// A second PrototypeScene::attach to the same scene reuses the tree meshes and the material by name and renders the
// same image with the same counts; another seed has other species meshes
TEST_F(VulkanPrototypeScene, ReattachReusesTreeMeshes)
{
    // Compares frames: golden mode
    GoldenMode();
    PrototypeScene p;
    Attach(p, 2000);
    Vec3 c(1536, 0, 1536);
    c(1)                = Ground(c(0), c(2));
    const Vec3 eye      = c + Vec3(0, 40, 120);
    auto a              = Shot(p, eye, c, 2);
    const auto sa       = p.forest().stats();
    const auto meshes   = p.forest().meshes();
    const auto material = p.forest().material();
    ASSERT_GT(sa.visible[0] + sa.visible[1] + sa.visible[2], 0u);
    p.detach();
    Attach(p, 2000);
    EXPECT_EQ(p.forest().meshes(), meshes);
    EXPECT_EQ(p.forest().material(), material);
    auto b        = Shot(p, eye, c, 2);
    const auto sb = p.forest().stats();
    EXPECT_EQ(sb.visible, sa.visible);
    EXPECT_EQ(pixelsDiffering(a, b), 0);
    // Another seed: other species meshes (their names carry the seed), the same material; a frame with them over the
    // seed 1 terrain (the trees are what is drawn here, a seed 2 terrain would cost new tiles)
    Attach(p, 2000, 2);
    EXPECT_NE(p.forest().meshes(), meshes);
    EXPECT_EQ(p.forest().material(), material);
    p.update();
    Capture(1);
}

// setFoliageTexture again with an attached forest (waits for the device, rebinds the foliage texture) gives the
// same image and counts
TEST_F(VulkanPrototypeScene, SetEnvironmentAgain)
{
    // Compares frames: golden mode
    GoldenMode();
    PrototypeScene p;
    Attach(p, 2000);
    Vec3 c(1536, 0, 1536);
    c(1)           = Ground(c(0), c(2));
    const Vec3 eye = c + Vec3(0, 40, 120);
    auto a         = Shot(p, eye, c, 2, true);
    const auto sa  = p.forest().stats();
    const auto ta  = scene->statistics().terrainNodes;
    ASSERT_GT(sa.visible[0] + sa.visible[1] + sa.visible[2], 0u);
    scene->setFoliageTexture(1);
    auto b = Capture(3);
    EXPECT_EQ(p.forest().stats().visible, sa.visible);
    EXPECT_EQ(scene->statistics().terrainNodes, ta);
    EXPECT_EQ(pixelsDiffering(a, b), 0);
}

// A resize recreates the depth image and its resolve target; the prototype then renders at the new size, with the
// pre-pass the same image
TEST_F(VulkanPrototypeScene, ResizeWithPrototype)
{
    // Compares frames: golden mode
    GoldenMode();
    PrototypeScene p;
    Attach(p, 2000);
    Vec3 c(1536, 0, 1536);
    c(1)           = Ground(c(0), c(2));
    const Vec3 eye = c + Vec3(0, 40, 120);
    Shot(p, eye, c, 2);
    window->setSize(160, 80);
    // The first frames at the new size
    auto without = Capture(2);
    EXPECT_EQ(without.w, 160);
    EXPECT_EQ(without.h, 80);
    EXPECT_GT(scene->statistics().terrainNodes, 0u);
    p.settings.depthPrePass = true;
    p.update();
    auto with = Capture(1);
    EXPECT_EQ(with.w, 160);
    EXPECT_EQ(pixelsDiffering(with, without), 0);
}

// The trees toggle in consecutive frames (the sets removed and added while earlier frames are in flight) ends in the
// same image and counts
TEST_F(VulkanPrototypeScene, TreesToggleInFlight)
{
    // Compares frames: golden mode
    GoldenMode();
    PrototypeScene p;
    // 2000 trees still fill every 256 m chunk's set of the area (the sets are what is toggled in flight); no terrain
    // draws (test speed: lavapipe's vertex stage)
    p.settings.terrain = false;
    p.settings.water   = false;
    Attach(p, 2000);
    Vec3 c(1536, 0, 1536);
    c(1)           = Ground(c(0), c(2));
    const Vec3 eye = c + Vec3(0, 60, 200);
    auto a         = Shot(p, eye, c, 2);
    const auto sa  = p.forest().stats();
    ASSERT_GT(sa.visible[0] + sa.visible[1] + sa.visible[2], 0u);
    for (int i = 0; i < 6; ++i)
    {
        p.settings.trees = i % 2 == 1;
        p.update();
        scene->renderFrame([](vk::CommandBuffer) {}, [](vk::CommandBuffer) {}, nullptr);
    }
    p.settings.trees = true;
    p.update();
    // Capture waits for the frames in flight; its last frame is the first with the trees on for good
    auto b = Capture(2);
    EXPECT_EQ(p.forest().stats().visible, sa.visible);
    EXPECT_EQ(pixelsDiffering(a, b), 0);
}

// Regression (S19a review): a frame uses one snapshot of the prototype settings taken after the transfer callback; a
// change in the render callback is the next frame's (setPrototypeSettings stays the per-frame setter)
TEST_F(VulkanPrototypeScene, PrototypeSettingsSnapshotPerFrame)
{
    // Compares frames: golden mode
    GoldenMode();
    PrototypeScene p;
    Attach(p, 0);
    Vec3 eye(1536, 0, 1536);
    eye(1)            = std::max(Ground(eye(0), eye(2)), Sea()) + 30;
    const Vec3 target = eye + Vec3(1, 0.1, 0);
    auto a            = Shot(p, eye, target, 2);
    auto frame        = [&](const RenderScene::Callback& transfer, const RenderScene::Callback& render)
    {
        TemplatedImage<ucvec4> img;
        scene->requestScreenshot([&](const TemplatedImage<ucvec4>& shot) { img = shot; });
        scene->renderFrame(transfer, render, nullptr);
        scene->waitIdle();
        return img;
    };
    auto noop             = [](vk::CommandBuffer) {};
    PrototypeSettings off = scene->prototypeSettings();
    off.sky               = false;
    auto b                = frame(noop, [&](vk::CommandBuffer) { scene->setPrototypeSettings(off); });
    EXPECT_EQ(pixelsDiffering(a, b), 0);
    auto c = frame(noop, noop);
    EXPECT_GT(pixelsDiffering(a, c), 100) << "the sky is not switched off";
    PrototypeSettings on = off;
    on.sky               = true;
    auto d               = frame([&](vk::CommandBuffer) { scene->setPrototypeSettings(on); }, noop);
    EXPECT_EQ(pixelsDiffering(a, d), 0);
}

// A PrototypeScene attached again starts fresh statistics: its forest (the same mesh IDs) reports no counts before its
// first frame; the GPU times of a new scene count from its first frames (RenderScene::takeGpuFrameTimes)
TEST_F(VulkanPrototypeScene, ReattachStartsFreshStatistics)
{
    // The forest's counts and the GPU times: no terrain (test speed: lavapipe's vertex stage)
    PrototypeScene p;
    p.settings.terrain = false;
    p.settings.water   = false;
    Attach(p, 2000);
    Vec3 c(1536, 0, 1536);
    c(1)           = Ground(c(0), c(2));
    const Vec3 eye = c + Vec3(0, 40, 120);
    Look(p, eye, c);
    p.update();
    auto frames = [&](int n)
    {
        bool gpu = false;
        for (int i = 0; i < n; ++i)
        {
            Frame(p);
            FrameTimes t;
            while (scene->takeGpuFrameTimes(t)) gpu = gpu || t.gpuMs >= 0;
        }
        scene->waitIdle();
        return gpu;
    };
    frames(4);
    const auto sa = p.forest().stats();
    ASSERT_GT(sa.visible[0] + sa.visible[1] + sa.visible[2], 0u);
    // The same scene and seed without trees: no counts before its first frame, none after it
    Attach(p, 0);
    auto s0 = p.forest().stats();
    EXPECT_EQ(s0.frame, 0u);
    EXPECT_EQ(s0.visible, (std::array<uint32_t, 3>{0, 0, 0}));
    frames(2);
    s0 = p.forest().stats();
    EXPECT_GT(s0.frame, 0u);
    EXPECT_EQ(s0.visible, (std::array<uint32_t, 3>{0, 0, 0}));
    // A new scene: its GPU times count from its first frames
    p.detach();
    Create(96, 64, 4);
    Attach(p, 2000);
    Look(p, eye, c);
    p.update();
    EXPECT_TRUE(frames(4) || !scene->base().physicalDevice.getProperties().limits.timestampComputeAndGraphics);
    EXPECT_EQ(p.forest().stats().visible, sa.visible);
}

// The PrototypeFoliage class needs the foliage texture of setFoliageTexture: S23c (F4) moved the check from
// addMaterial / setMaterial into the frame (a game registers its foliage material with its world before the render
// hooks set the environment). Registration passes; a frame with an instance of the class to draw dies; the same frame
// after setFoliageTexture renders the tree
TEST_F(VulkanPrototypeScene, FoliageClassNeedsTexture)
{
    // Compares frames: golden mode
    GoldenMode();
    MaterialData foliage;
    foliage.pipelineClass = uint8_t(PipelineClass::PrototypeFoliage);
    foliage.twoSided      = true;
    scene->addMaterial(1, "foliage", foliage);
    scene->addMaterial(2, "flat", MaterialData());
    scene->setMaterial(2, foliage);
    scene->setMaterial(2, MaterialData());
    species = FoliageGenerator::treeSpecies(1);
    scene->addMesh(1, "tree", prototypeTreeMesh(species[0], {200, 400, 800}, 0), 1);
    SetCamera(0, Vec3(1030, 8, 1060), viewRotation(lookAt(vec3(1030, 8, 1060), vec3(1030, 8, 1030), vec3(0, 1, 0))),
              1.0);
    // Registered, nothing drawn: the frame renders
    const TemplatedImage<ucvec4> empty = Capture(2);
    ASSERT_GT(empty.w, 0);
    StaticInstance tree;
    tree.mesh = 1;
    tree.setPosition(vec3(6, 0, 6));
    tree.scale                  = 16;
    const InstanceSetHandle set = scene->addInstanceSet(RegionId{0, 0, 4, 4}, {tree});
#ifdef SAIGA_ASSERTS
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_DEATH(RenderFrame(), "PrototypeFoliage instances need the foliage texture of setFoliageTexture");
#endif
    scene->setFoliageTexture(1);
    const TemplatedImage<ucvec4> drawn = Capture(3);
    scene->removeInstanceSet(set);
    const TemplatedImage<ucvec4> background = Capture(3);
    EXPECT_GT(pixelsDiffering(background, drawn, 8), 50) << "the tree is drawn after setFoliageTexture";
}

// The game tools (S20: the context's terrain fields: ground callback, size, sea level, highest point, height range;
// the statistics from PrototypeScene)
TEST_F(VulkanPrototypeScene, GameTools)
{
    using Json = McpServer::Json;
    PrototypeScene p;
    Attach(p, 100);
    Vulkan::WorldMcpCamera camera;
    camera.position  = Vec3(1100, 300, 1100);
    camera.rotation  = Quat(AngleAxis(0.3, Vec3(0, 1, 0))) * Quat(AngleAxis(-0.5, Vec3(1, 0, 0)));
    int cameraWrites = 0;
    bool walkMode    = false;
    float eyeHeight  = 1.7f;
    McpServer server("test");
    Vulkan::WorldMcpContext context;
    context.settings     = &p.settings;
    context.stats        = [&p]() { return p.statsJson(); };
    context.ground       = [this](double x, double z) { return Ground(x, z); };
    context.worldSize    = 4096;
    context.seaLevel     = Sea();
    context.highestPoint = Vec3(1500, 300, 1600);
    context.heightRange  = []() { return std::make_pair(-19.0, 400.0); };
    context.getCamera    = [&]() { return camera; };
    context.setCamera    = [&](const Vulkan::WorldMcpCamera& c)
    {
        camera = c;
        cameraWrites++;
    };
    context.walkMode  = &walkMode;
    context.eyeHeight = &eyeHeight;
    Vulkan::registerWorldMcpTools(server, context);

    Shot(p, camera.position, Vec3(1536, 0, 1536), 3);
    FrameTimes gpuTimes;
    bool gpuTaken = false;
    while (scene->takeGpuFrameTimes(gpuTimes)) gpuTaken = true;
    Json s = toolResultJson(callWorldTool(server, "game_get_state", Json::object{}));
    EXPECT_EQ(s["world"]["size_m"].number_value(), 4096) << s.dump();
    EXPECT_EQ(s["world"]["min_height"].number_value(), -19) << s.dump();
    EXPECT_EQ(s["world"]["max_height"].number_value(), 400) << s.dump();
    EXPECT_NEAR(s["world"]["water_level"].number_value(), 6, 1e-9) << s.dump();
    EXPECT_NEAR(s["world"]["highest_point"][1].number_value(), 300, 1e-9) << s.dump();
    EXPECT_NEAR(s["world"]["center"][1].number_value(), Ground(2048, 2048), 0.01) << s.dump();
    EXPECT_TRUE(s["stats"]["gpu_frame_ms"].is_number()) << s.dump();
    EXPECT_TRUE(s["stats"]["gpu_section_ms"].is_object()) << s.dump();
    EXPECT_TRUE(s["stats"]["cpu_ms"].is_null()) << s.dump();  // S20: the prototype's CPU terrain selection is gone
    if (gpuTaken)
    {
        for (const char* name : RenderScene::gpuSectionNames())
            EXPECT_TRUE(s["stats"]["gpu_section_ms"][name].is_number()) << name << " " << s.dump();
    }
    const auto fs = p.forest().stats();
    EXPECT_EQ(s["stats"]["trees_total"].int_value(), p.forest().treeCount()) << s.dump();
    EXPECT_EQ(s["stats"]["trees_visible_lod0"].int_value(), int(fs.visible[0])) << s.dump();
    EXPECT_EQ(s["stats"]["tree_triangles"].number_value(), double(fs.triangles)) << s.dump();
    EXPECT_GT(s["stats"]["terrain_nodes"].int_value(), 0) << s.dump();
    EXPECT_TRUE(s["settings"].is_object());
    EXPECT_NEAR(s["camera"]["ground_height"].number_value(), Ground(1100, 1100), 0.01);
    EXPECT_FALSE(s["camera"]["walk_mode"].bool_value());

    s = toolResultJson(callWorldTool(server, "game_set_settings", Json::object{{"fovy", 1000.0}, {"nope", 1}}));
    EXPECT_LT(s["settings"]["fovy"].number_value(), 1000) << s.dump();
    EXPECT_EQ(s["invalid_keys"][0].string_value(), "nope") << s.dump();
    EXPECT_EQ(cameraWrites, 1);
    EXPECT_NEAR(camera.fovy, s["settings"]["fovy"].number_value() * pi<double>() / 180.0, 1e-6);
    callWorldTool(server, "game_set_settings", Json::object{{"grass", false}});
    p.update();
    EXPECT_FALSE(scene->prototypeSettings().grass);

    // game_terrain_query: the heights on the §12.2 diagonal (the ground callback), the normal at 2 m, under water
    const double x = 1300.5, z = 1700.25;
    Json::array points;
    points.push_back(Json::array{x, z});
    points.push_back(Json::array{-500.0, 10.0});
    Json q = toolResultJson(callWorldTool(server, "game_terrain_query", Json::object{{"points", points}}));
    EXPECT_NEAR(q[0]["height"].number_value(), Ground(x, z), 0.01) << q.dump();
    EXPECT_EQ(q[0]["under_water"].bool_value(), Ground(x, z) < Sea()) << q.dump();
    EXPECT_GT(q[0]["normal"][1].number_value(), 0) << q.dump();
    Json::array badPoints;
    badPoints.push_back(Json::array{1});
    EXPECT_TRUE(callWorldTool(server, "game_terrain_query", Json::object{{"points", badPoints}})["result"]["isError"]
                    .bool_value());

    const Quat rotation = camera.rotation;
    Json c              = toolResultJson(callWorldTool(
        server, "game_set_camera", Json::object{{"position", Json::array{x, z}}, {"height_above_ground", 30.0}}));
    EXPECT_NEAR(c["height_above_ground"].number_value(), 30, 0.02) << c.dump();
    EXPECT_NEAR(c["ground_height"].number_value(), Ground(x, z), 0.02) << c.dump();
    EXPECT_NEAR(camera.position(0), x, 1e-2);
    EXPECT_TRUE(camera.rotation.isApprox(rotation));
    c = toolResultJson(callWorldTool(server, "game_set_camera", Json::object{{"walk_mode", true}}));
    EXPECT_TRUE(walkMode);
    EXPECT_NEAR(c["height_above_ground"].number_value(), eyeHeight, 0.02) << c.dump();
    callWorldTool(server, "game_set_camera", Json::object{{"walk_mode", false}});
    EXPECT_FALSE(walkMode);
    EXPECT_TRUE(
        callWorldTool(server, "game_set_camera", Json::object{{"position", Json::array{1}}})["result"]["isError"]
            .bool_value());
}
