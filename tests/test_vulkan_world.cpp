/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/scene/PrototypeWorld.h"
#include "saiga/vulkan/scene/RenderScene.h"
#include "saiga/vulkan/window/HeadlessWindow.h"
#include "saiga/vulkan/world/ForestRenderer.h"
#include "saiga/vulkan/world/HeightField.h"
#include "saiga/vulkan/world/PrototypeForest.h"
#include "saiga/vulkan/world/WorldBenchmark.h"
#include "saiga/vulkan/world/WorldCommon.h"
#include "saiga/vulkan/world/WorldMcp.h"
#include "saiga/vulkan/world/WorldRenderer.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <thread>

#include "golden_image.h"
#include "mcp_test_helper.h"
#include "vulkan_test_context.h"

using namespace Saiga;
using namespace Saiga::Vulkan;

static HeightField SmallHeightField()
{
    HeightField hf;
    HeightField::Parameters p;
    p.size    = 257;
    p.spacing = 2.0f;
    hf.generate(p);
    return hf;
}

TEST(VulkanWorld, HeightFieldInterpolation)
{
    HeightField hf = SmallHeightField();
    EXPECT_FLOAT_EQ(hf.worldSize(), 512.0f);
    // At the samples the height is exact
    EXPECT_FLOAT_EQ(hf.height(10 * 2.0f, 20 * 2.0f), hf.sample(10, 20));
    // Within a cell: the triangle (i,j) (i+1,j) (i,j+1), as the terrain mesh
    float x = 10.25f * 2, z = 20.5f * 2;
    float ha = hf.sample(10, 20), hb = hf.sample(11, 20), hc = hf.sample(10, 21);
    EXPECT_NEAR(hf.height(x, z), ha + 0.25f * (hb - ha) + 0.5f * (hc - ha), 1e-4f);
    // Normals point up
    EXPECT_GT(hf.normal(100, 100)(1), 0.0f);
    EXPECT_LE(hf.minHeight(), hf.maxHeight());
}

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

TEST(VulkanWorld, TreePlacement)
{
    HeightField hf = SmallHeightField();
    auto species   = FoliageGenerator::treeSpecies(1);
    auto trees     = ForestRenderer::placeTrees(hf, 500, 3, species);
    EXPECT_GT(trees.size(), 0u);
    EXPECT_LE(trees.size(), 500u);
    for (auto& t : trees)
    {
        float x = t.posYaw(0), z = t.posYaw(2);
        ASSERT_GE(x, 0);
        ASSERT_LE(x, hf.worldSize());
        // On the ground (sunk in a little), above the water
        EXPECT_NEAR(t.posYaw(1), hf.height(x, z) - 0.25f, 1e-3f);
        EXPECT_GT(hf.height(x, z), hf.waterLevel());
        int s = int(t.scaleSpecies(1));
        ASSERT_GE(s, 0);
        ASSERT_LT(s, (int)species.size());
        EXPECT_GE(t.scaleSpecies(0), species[s].heightRange(0));
        EXPECT_LE(t.scaleSpecies(0), species[s].heightRange(1));
    }
    // Deterministic: every tree equal (the normalisation sum is serial, so the scheduling cannot change the placement)
    auto again = ForestRenderer::placeTrees(hf, 500, 3, species);
    ASSERT_EQ(again.size(), trees.size());
    for (size_t i = 0; i < trees.size(); ++i)
    {
        EXPECT_EQ(again[i].posYaw, trees[i].posYaw);
        EXPECT_EQ(again[i].scaleSpecies, trees[i].scaleSpecies);
    }
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
    // S19a: wireframe, freeze_time and freeze_culling are dropped (setTime covers goldens and benchmarks)
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
// deterministic keys; a tree at the world's far edge belongs to the last chunk
TEST(VulkanWorld, PrototypeTreeSets)
{
    HeightField hf;
    HeightField::Parameters p;
    p.size    = 513;
    p.spacing = 2.0f;  // 1024 m: 4 x 4 chunks
    hf.generate(p);
    auto species                = FoliageGenerator::treeSpecies(1);
    const std::vector<MeshId> m = {11, 12, 13};
    auto trees                  = ForestRenderer::placeTrees(hf, 3000, 1, species);
    auto sets                   = prototypeTreeSets(hf, 3000, 1, species, m);
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
}

TEST(WorldBenchmark, Paths)
{
    // An 8 km world like the default one, at a coarse resolution
    HeightField hf;
    HeightField::Parameters p;
    p.size    = 513;
    p.spacing = 16.0f;
    hf.generate(p);
    float c = hf.worldSize() * 0.5f;
    ASSERT_FLOAT_EQ(c, 4096.0f);
    const int n = WorldBenchmark::measuredFrames;

    for (int i : {0, 150, 300, 599})
    {
        auto pose = WorldBenchmark::pose("flyover", hf, i);
        // On the 3 km circle around the center, 300 m above the ground below, looking at the center on the ground
        EXPECT_NEAR(std::hypot(pose.eye(0) - c, pose.eye(2) - c), 3000.0f, 0.01f);
        EXPECT_NEAR(pose.eye(1), hf.height(pose.eye(0), pose.eye(2)) + 300.0f, 1e-3f);
        EXPECT_NEAR(pose.target(0), c, 1e-3f);
        EXPECT_NEAR(pose.target(2), c, 1e-3f);
        EXPECT_NEAR(pose.target(1), hf.height(c, c), 1e-3f);
    }
    // One revolution in measuredFrames frames
    auto first = WorldBenchmark::pose("flyover", hf, 0), full = WorldBenchmark::pose("flyover", hf, n);
    EXPECT_NEAR((first.eye - full.eye).norm(), 0.0f, 0.05f);
    EXPECT_GT((first.eye - WorldBenchmark::pose("flyover", hf, n / 2).eye).norm(), 5999.0f);

    auto start = WorldBenchmark::pose("forest", hf, 0), end = WorldBenchmark::pose("forest", hf, n - 1);
    EXPECT_NEAR(start.eye(0), c - 300.0f, 1e-3f);
    EXPECT_NEAR(end.eye(0), c + 300.0f, 1e-3f);
    for (int i : {0, 200, n - 1})
    {
        auto pose = WorldBenchmark::pose("forest", hf, i);
        EXPECT_NEAR(pose.eye(2), c, 1e-3f);
        float ground = std::max(hf.height(pose.eye(0), c), hf.waterLevel());
        EXPECT_NEAR(pose.eye(1), ground + 2.0f, 1e-3f);
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
    HeightField hf = SmallHeightField();
    WorldBenchmark bench("forest", hf, "test gpu");
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
    WorldBenchmark noGpu("forest", hf, "test gpu");
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

// Regression (S19a review): finished() only proves that the CPU samples reached the last measured frame; a run that
// missed one is not complete (game_world reports it as a failure)
TEST(WorldBenchmark, IncompleteRun)
{
    HeightField hf = SmallHeightField();
    WorldBenchmark bench("forest", hf, "test gpu");
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

namespace
{
// The CPU reference of the trees' culling (the prototype's rule as the instance path implements it, §17.5): per LOD
// the drawn trees and the ambiguous ones (within 1e-3 relative of a plane or an end distance)
struct ForestReference
{
    std::array<int, 3> perLod{};
    int ambiguous = 0;
    int total() const { return perLod[0] + perLod[1] + perLod[2]; }
};

// The inward camera-relative frustum planes of RenderScene (left, right, bottom, top, near) for the camera
std::array<Vec4, 5> cameraPlanes(const Quat& rotation, double fovy, double aspect, double nearPlane)
{
    const Mat3 R      = rotation.matrix();
    const double tanY = std::tan(fovy / 2), tanX = tanY * aspect;
    std::array<Vec4, 5> p;
    const Vec3 n[4] = {Vec3(1, 0, -tanX), Vec3(-1, 0, -tanX), Vec3(0, 1, -tanY), Vec3(0, -1, -tanY)};
    for (int k = 0; k < 4; ++k)
    {
        const Vec3 m = R * n[k].normalized();
        p[k]         = Vec4(m(0), m(1), m(2), 0);
    }
    const Vec3 f = R * Vec3(0, 0, -1);
    p[4]         = Vec4(f(0), f(1), f(2), -nearPlane);
    return p;
}

ForestReference forestReference(const WorldRenderer& world, const Vec3& eye, const Quat& rotation, double aspect)
{
    ForestReference r;
    auto species = FoliageGenerator::treeSpecies(world.parameters().seed);
    auto trees =
        ForestRenderer::placeTrees(world.heightField(), world.parameters().treeCount, world.parameters().seed, species);
    const auto ends   = prototypeTreeLodEnds(world.settings);
    const double pad  = prototypeTreePadding(world.settings);
    const auto planes = cameraPlanes(rotation, double(radians(world.settings.fovy)), aspect, 0.1);
    if (!world.settings.trees) return r;
    for (const auto& t : trees)
    {
        const TreeSpecies& sp = species[size_t(t.scaleSpecies(1))];
        const double scale    = t.scaleSpecies(0);
        const Vec3 center =
            Vec3(t.posYaw(0), t.posYaw(1), t.posYaw(2)) + Vec3(0, double(sp.bounds(0)) * scale, 0) - eye;
        const double radius = (double(sp.bounds(1)) + pad) * scale;
        const double d      = center.norm();
        bool amb            = false;
        for (float e : ends) amb = amb || std::abs(d - e) <= 1e-3 * e;
        bool inside = d <= ends[2];
        for (const Vec4& p : planes)
        {
            const double s = p(0) * center(0) + p(1) * center(1) + p(2) * center(2) + p(3) + radius;
            amb            = amb || std::abs(s) <= 1e-3 * std::max(d, 1.0);
            inside         = inside && s >= 0;
        }
        if (amb) r.ambiguous++;
        if (!inside) continue;
        int lod = 0;
        while (lod < 2 && d > ends[size_t(lod)]) lod++;
        r.perLod[size_t(lod)]++;
    }
    return r;
}

void expectForest(const PrototypeForest::Stats& s, const ForestReference& r, const std::string& what)
{
    int total = 0;
    for (int l = 0; l < 3; ++l)
    {
        EXPECT_LE(std::abs(int(s.visible[size_t(l)]) - r.perLod[size_t(l)]), r.ambiguous)
            << what << ": LOD " << l << " gpu " << s.visible[size_t(l)] << " reference " << r.perLod[size_t(l)];
        total += int(s.visible[size_t(l)]);
    }
    EXPECT_LE(std::abs(total - r.total()), r.ambiguous) << what;
}
}  // namespace

// The world prototype drawn by RenderScene (S19a: setPrototypeTerrain, the trees as instance sets through
// WorldRenderer::attach) in a headless window: MSAA (default 4x), clay mode, time 0, ImGui off
class VulkanWorldScene : public ::testing::Test
{
   protected:
    void SetUp() override
    {
        if (!Vulkan::Instance::suitableDeviceAvailable()) GTEST_SKIP() << "No Vulkan 1.3 device available";
        Create(96, 64, 4);
        // The instance draws need it (asserted by RenderScene); Vulkan 1.3 does not require it
        if (!scene->base().enabledFeatures.drawIndirectFirstInstance)
            GTEST_SKIP() << "drawIndirectFirstInstance is not supported";
    }
    void TearDown() override { Destroy(); }

    void Create(int w, int h, int samples)
    {
        Destroy();
        width  = w;
        height = h;
        WindowParameters windowParameters;
        windowParameters.width  = w;
        windowParameters.height = h;
        windowParameters.saigaParameters.shaderDirectory.push_back(SAIGA_PROJECT_SOURCE_DIR "/shader");
        Vulkan::VulkanParameters vulkanParameters;
        vulkanParameters.enableImgui = false;
        RenderSceneParameters params;
        params.imgui       = false;
        params.msaaSamples = samples;
        params.nearPlane   = 0.1f;
        window             = std::make_unique<Vulkan::HeadlessWindow>(windowParameters);
        scene              = std::make_unique<RenderScene>(*window, vulkanParameters, params);
    }
    void Destroy()
    {
        if (scene) scene->waitIdle();
        scene.reset();
        window.reset();
    }

    std::string deviceName() const
    {
        return std::string(scene->base().physicalDevice.getProperties().deviceName.data());
    }

    void Attach(WorldRenderer& world, const WorldParameters& params,
                std::shared_ptr<const HeightField> heightField = nullptr)
    {
        if (heightField)
            world.init(params, std::move(heightField));
        else
            world.init(params);
        world.attach(*scene);
    }

    // The camera (the settings' fovy, 60 by default) at eye looking at target
    void SetCamera(const WorldRenderer& world, const vec3& eye, const vec3& target)
    {
        cameraRotation = viewRotation(lookAt(eye, target, vec3(0, 1, 0)));
        cameraEye      = eye.cast<double>();
        scene->setCamera(0, cameraEye, cameraRotation, double(radians(world.settings.fovy)));
    }

    // Renders a frame (the settings applied first) and returns it
    TemplatedImage<ucvec4> RenderFrame(WorldRenderer& world)
    {
        world.update();
        TemplatedImage<ucvec4> img;
        scene->requestScreenshot([&](const TemplatedImage<ucvec4>& shot) { img = shot; });
        scene->renderFrame([](vk::CommandBuffer) {}, [](vk::CommandBuffer) {}, nullptr);
        return img;
    }

    // Renders n frames and returns the last; waits for it (its statistics are the newest)
    TemplatedImage<ucvec4> Capture(WorldRenderer& world, const vec3& eye, const vec3& target, int n = 5)
    {
        SetCamera(world, eye, target);
        TemplatedImage<ucvec4> img;
        for (int i = 0; i < n; ++i) img = RenderFrame(world);
        scene->waitIdle();
        return img;
    }

    ForestReference Reference(const WorldRenderer& world) const
    {
        return forestReference(world, cameraEye, cameraRotation, double(width) / height);
    }

    int width = 0, height = 0;
    Vec3 cameraEye      = Vec3::Zero();
    Quat cameraRotation = Quat::Identity();
    std::unique_ptr<Vulkan::HeadlessWindow> window;
    std::unique_ptr<RenderScene> scene;
};

// The number of pixels that differ by more than threshold in a channel; INT_MAX (and a test failure) for an empty image
// or images of different sizes (a failed capture must not compare as equal)
static int differingPixels(const TemplatedImage<ucvec4>& a, const TemplatedImage<ucvec4>& b, int threshold = 0)
{
    if (a.w == 0 || a.h == 0 || a.w != b.w || a.h != b.h)
    {
        ADD_FAILURE() << "differingPixels: images of " << a.w << "x" << a.h << " and " << b.w << "x" << b.h;
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

TEST_F(VulkanWorldScene, RendersTerrainTreesAndSky)
{
    WorldRenderer world;
    WorldParameters params;
    params.terrain.size = 513;
    params.treeCount    = 2000;
    Attach(world, params);
    EXPECT_TRUE(scene->hasPrototypeTerrain());

    // Above the center of the world, looking down at the horizon
    auto& hf = world.heightField();
    vec3 c   = vec3(hf.worldSize() * 0.5f, 0, hf.worldSize() * 0.5f);
    c(1)     = hf.height(c(0), c(2));
    vec3 eye = c + vec3(0, 60, 200);

    // Without the sky pass the top of the image is the clear color
    world.settings.sky           = false;
    TemplatedImage<ucvec4> noSky = Capture(world, eye, c + vec3(0, 20, 0), 1);
    world.settings.sky           = true;
    // A few frames: the statistics are read back from completed frames
    TemplatedImage<ucvec4> img = Capture(world, eye, c + vec3(0, 20, 0));
    ASSERT_EQ(img.w, 96);

    // The sky at the top is a uniform gray that differs from the clear color (regression: the test passed without a
    // sky pass, because the clear color is gray as well), the terrain at the bottom differs from the clear color
    ucvec4 top = img(1, 48), clearTop = noSky(1, 48), bottom = img(62, 48);
    EXPECT_EQ(top(0), top(1));
    EXPECT_EQ(top(1), top(2));
    EXPECT_GT(std::abs(int(top(0)) - int(clearTop(0))), 2) << "sky " << int(top(0)) << " clear " << int(clearTop(0));
    EXPECT_GT(std::abs(int(bottom(0)) - int(clearTop(0))), 2);
    world.update();
    EXPECT_GT(world.stats().terrainNodes, 0);
    EXPECT_GT(scene->statistics().terrainNodes, 0u);
    const auto fs = world.forest().stats();
    EXPECT_GT(fs.visible[0] + fs.visible[1] + fs.visible[2], 0u);
    EXPECT_GT(fs.triangles, 0u);
    expectForest(fs, Reference(world), "overview");
}

// An external height field (S13b: generated once for a game's simulation and its renderer) is shared, not copied, and
// renders like a generated one
TEST_F(VulkanWorldScene, SharesAnExternalHeightField)
{
    HeightField::Parameters terrain;
    terrain.size  = 257;
    auto external = std::make_shared<HeightField>();
    external->generate(terrain);

    WorldRenderer world;
    WorldParameters params;
    params.terrain.size = 129;  // ignored with an external height field
    params.treeCount    = 500;
    Attach(world, params, external);
    EXPECT_EQ(&world.heightField(), external.get());
    EXPECT_EQ(&scene->prototypeWorld()->heightField(), external.get());
    EXPECT_EQ(world.heightField().size(), 257);
    EXPECT_EQ(external.use_count(), 3);  // here, in the world renderer and in RenderScene's prototype
    auto& hf = world.heightField();
    vec3 c   = vec3(hf.worldSize() * 0.5f, 0, hf.worldSize() * 0.5f);
    c(1)     = hf.height(c(0), c(2));
    Capture(world, c + vec3(0, 60, 200), c + vec3(0, 20, 0), 3);
    EXPECT_GT(scene->statistics().terrainNodes, 0u);
    world.destroy();
    EXPECT_EQ(external.use_count(), 2);  // released by destroy(); RenderScene keeps its prototype
    Destroy();
    EXPECT_EQ(external.use_count(), 1);
}

// Regression: positions were passed to the fragment shader in world space as floats. Far from the origin (x = 7 km:
// float steps of 0.5 mm) and close to the ground, dFdx/dFdy of the position (flat shading) were rounding noise and
// the ground showed salt-and-pepper noise. The world passes now work with positions relative to the camera.
TEST_F(VulkanWorldScene, FlatShadingIsStableFarFromTheOrigin)
{
    WorldRenderer world;
    WorldParameters params;
    params.terrain.size    = 2049;
    params.terrain.spacing = 4;  // 8 km world
    params.treeCount       = 10;
    Attach(world, params);
    world.settings.grass       = false;
    world.settings.trees       = false;
    world.settings.fog         = false;
    world.settings.flatShading = true;
    world.settings.fovy        = 20;

    auto& hf = world.heightField();
    // 15 cm above a point far from the origin that is above the water (x ~ 7 km: float steps of 0.5 mm; with the narrow
    // field of view a pixel covers less than 1 mm of ground, like a full HD window 0.3 m above the ground)
    vec3 eye(0, 0, 0);
    for (float x = hf.worldSize() * 0.85f; x > hf.worldSize() * 0.5f; x -= 16)
    {
        eye = vec3(x, hf.height(x, x) + 0.15f, x);
        if (hf.height(x, x) > hf.waterLevel() + 2) break;
    }
    // Looking down at the ground in front of the camera
    const vec3 target = eye + vec3(0.6f, -1.0f, 0.4f);

    // Reference without terrain and water: the check below must see terrain, not a uniform background
    world.settings.terrain           = false;
    world.settings.water             = false;
    TemplatedImage<ucvec4> reference = Capture(world, eye, target, 1);
    // Water stays off: its uniform plane could pass both checks without any terrain
    world.settings.terrain     = true;
    TemplatedImage<ucvec4> img = Capture(world, eye, target, 1);
    ASSERT_EQ(img.w, 96);
    ASSERT_EQ(reference.w, 96);
    int differs = 0, pixels = 0;
    for (int y = 8; y < 56; ++y)
    {
        for (int x = 8; x < 88; ++x)
        {
            differs += std::abs(int(img(y, x)(0)) - int(reference(y, x)(0))) > 2;
            pixels++;
        }
    }
    EXPECT_GT(differs, pixels * 9 / 10) << "the terrain does not cover the image";

    // The ground in front of the camera is one or two terrain triangles: neighbouring pixels have about the same
    // shade (with the bug about half of them differed by more than 20)
    int rough = 0, total = 0;
    for (int y = 8; y < 56; ++y)
    {
        for (int x = 8; x < 87; ++x)
        {
            rough += std::abs(int(img(y, x)(0)) - int(img(y, x + 1)(0))) > 20;
            total++;
        }
    }
    EXPECT_LT(rough, total / 50) << rough << " of " << total << " neighbouring pixels differ by more than 20";
}

// S19a: the drawn trees per LOD equal the CPU reference of the prototype's rule (§17.5 band) also after the settings
// change through PrototypeForest::apply: the wind 0 -> 4 (the bounds padding), crossed LOD distances (an empty LOD 1)
// and the trees toggle (the sets removed and re-added)
TEST_F(VulkanWorldScene, ForestFollowsSettings)
{
    WorldRenderer world;
    WorldParameters params;
    params.terrain.size = 513;
    params.treeCount    = 5000;
    Attach(world, params);
    world.settings.treeLod1 = 30;
    world.settings.treeLod2 = 120;
    auto& hf                = world.heightField();
    vec3 eye(hf.worldSize() * 0.5f, 0, hf.worldSize() * 0.5f);
    eye(1)            = hf.height(eye(0), eye(2)) + 40;
    const vec3 target = eye + vec3(1, -0.3f, 0.2f);

    auto check = [&](const char* what)
    {
        Capture(world, eye, target, 3);
        const auto st = world.forest().stats();
        EXPECT_EQ(st.frame, scene->frames()) << what;
        const auto ref = Reference(world);
        std::cout << what << ": gpu " << st.visible[0] << " " << st.visible[1] << " " << st.visible[2] << ", reference "
                  << ref.perLod[0] << " " << ref.perLod[1] << " " << ref.perLod[2] << " (" << ref.ambiguous
                  << " ambiguous)" << std::endl;
        expectForest(st, ref, what);
        return st;
    };
    world.settings.wind = 0;
    check("wind 0");
    world.settings.wind = 4;
    check("wind 4");
    world.settings.treeLod1 = 500;
    world.settings.treeLod2 = 50;
    auto crossed            = check("crossed LOD distances");
    EXPECT_EQ(crossed.visible[1], 0u);
    EXPECT_GT(crossed.visible[0], 0u);

    world.settings.trees = false;
    auto off             = check("trees off");
    EXPECT_EQ(off.visible[0] + off.visible[1] + off.visible[2], 0u);
    world.settings.trees = true;
    auto on              = check("trees on again");
    EXPECT_EQ(on.visible, crossed.visible);
}

// S19a: the grass toggle (racing turns it off): no grass clumps and no grass pixels
TEST_F(VulkanWorldScene, GrassToggle)
{
    WorldRenderer world;
    WorldParameters params;
    params.terrain.size = 513;
    params.treeCount    = 10;
    Attach(world, params);
    world.settings.trees = false;
    auto& hf             = world.heightField();
    // Low above a grassy point, looking down
    vec3 eye(0, 0, 0);
    for (float x = 300; x < 700; x += 7)
    {
        const vec3 n = hf.normal(x, x, 1.0f);
        if (hf.height(x, x) > hf.waterLevel() + 2 && n(1) > 0.95f)
        {
            eye = vec3(x, hf.height(x, x) + 1.5f, x);
            break;
        }
    }
    ASSERT_GT(eye(0), 0);
    const vec3 target = eye + vec3(1, -0.8f, 0.3f);
    auto withGrass    = Capture(world, eye, target, 2);
    EXPECT_GT(scene->prototypeWorld()->stats().grassClumps, 0);
    world.settings.grass = false;
    auto noGrass         = Capture(world, eye, target, 2);
    EXPECT_EQ(scene->prototypeWorld()->stats().grassClumps, 0);
    EXPECT_FALSE(scene->prototypeSettings().grass);
    EXPECT_GT(differingPixels(withGrass, noGrass, 8), 50);
}

// S19a: the time comes from setTime, not the wall clock: the same time gives the same image after a pause, another
// time moves the wind (trees and grass)
TEST_F(VulkanWorldScene, TimeIndependentOfWallClock)
{
    WorldRenderer world;
    WorldParameters params;
    params.terrain.size = 513;
    params.treeCount    = 5000;
    Attach(world, params);
    world.settings.wind = 4;
    auto& hf            = world.heightField();
    vec3 eye(hf.worldSize() * 0.5f - 28, 0, hf.worldSize() * 0.5f - 80);
    eye(1)            = std::max(hf.height(eye(0), eye(2)), hf.waterLevel()) + 2.0f;
    const vec3 target = eye + vec3(1, -0.2f, 0);
    scene->setTime(0);
    auto a = Capture(world, eye, target, 2);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    auto b = Capture(world, eye, target, 2);
    EXPECT_EQ(differingPixels(a, b), 0);
    scene->setTime(1.7);
    auto c = Capture(world, eye, target, 2);
    EXPECT_GT(differingPixels(a, c), 0) << "the wind does not move";
}

// S19a: the scene's depth is resolved with SAMPLE_ZERO (MSAA) or sampled directly (1 sample), with and without the
// depth pre-pass: validation clean (the layer aborts on an error) and the same image with and without the pre-pass
TEST_F(VulkanWorldScene, DepthResolveAndPrePass)
{
    for (int samples : {4, 1})
    {
        Create(96, 64, samples);
        WorldRenderer world;
        WorldParameters params;
        params.terrain.size = 257;
        params.treeCount    = 500;
        Attach(world, params);
        auto& hf                    = world.heightField();
        vec3 c                      = vec3(hf.worldSize() * 0.5f, 0, hf.worldSize() * 0.5f);
        c(1)                        = hf.height(c(0), c(2));
        world.settings.depthPrePass = true;
        auto with                   = Capture(world, c + vec3(0, 40, 120), c, 2);
        world.settings.depthPrePass = false;
        auto without                = Capture(world, c + vec3(0, 40, 120), c, 2);
        EXPECT_EQ(differingPixels(with, without), 0) << samples << " samples";
        world.destroy();
    }
}

// S19a (replaces the prototype's lastFrameTimes test): RenderScene's sections and sub-sections arrive for every
// completed frame, in order and without gaps; a sub-section is inside its pass
TEST_F(VulkanWorldScene, Sections)
{
    const auto& names                       = RenderScene::gpuSectionNames();
    const std::vector<std::string> expected = {
        "transfer",        "cull",        "depth",     "depth.terrain", "scene", "scene.terrain",
        "scene.instances", "scene.grass", "scene.sky", "resolve",       "post",  "imgui"};
    ASSERT_EQ(names.size(), expected.size());
    for (size_t i = 0; i < names.size(); ++i) EXPECT_EQ(names[i], expected[i]);
    WorldRenderer world;
    WorldParameters params;
    params.terrain.size = 257;
    params.treeCount    = 500;
    Attach(world, params);
    auto& hf = world.heightField();
    vec3 c   = vec3(hf.worldSize() * 0.5f, 0, hf.worldSize() * 0.5f);
    vec3 eye = c + vec3(0, hf.height(c(0), c(2)) + 80, 100);
    SetCamera(world, eye, c);
    std::vector<uint64_t> frames;
    for (int i = 0; i < 12; ++i)
    {
        world.settings.depthPrePass = i % 2 == 0;
        RenderFrame(world);
        FrameTimes t;
        while (scene->takeGpuFrameTimes(t))
        {
            frames.push_back(t.frame);
            ASSERT_EQ(t.gpuSectionMs.size(), names.size());
            for (double ms : t.gpuSectionMs) EXPECT_GE(ms, 0);
            // The sub-sections are within their pass (timer resolution aside)
            EXPECT_LE(t.gpuSectionMs[3], t.gpuSectionMs[2] + 0.01);
            EXPECT_LE(t.gpuSectionMs[5] + t.gpuSectionMs[6] + t.gpuSectionMs[7] + t.gpuSectionMs[8],
                      t.gpuSectionMs[4] + 0.01);
            EXPECT_GE(t.gpuMs, t.gpuSectionMs[4]);
        }
    }
    scene->waitIdle();
    // The last frames' timestamps are read back when the next frame starts: a flush frame
    RenderFrame(world);
    FrameTimes t;
    while (scene->takeGpuFrameTimes(t)) frames.push_back(t.frame);
    // Regression (S03): the readback only took the newest completed frame and lost about half of the frames. Every
    // one of the 12 frames arrives, in order
    ASSERT_GE(frames.size(), 12u);
    for (size_t i = 0; i < frames.size(); ++i) EXPECT_EQ(frames[i], uint64_t(i + 1)) << i;
    EXPECT_EQ(scene->lastGpuFrameTimes().frame, frames.back());
}

// S19a review: a WorldRenderer attached to the same scene again (init + attach after destroy) asserted on the
// registered name "prototype/foliage" and leaked reserved IDs; it now reuses the tree meshes and the material by name
// and renders the same image with the same counts
TEST_F(VulkanWorldScene, ReattachReusesTreeMeshes)
{
    WorldRenderer world;
    WorldParameters params;
    params.terrain.size = 257;
    params.treeCount    = 2000;
    Attach(world, params);
    auto& hf            = world.heightField();
    vec3 c              = vec3(hf.worldSize() * 0.5f, 0, hf.worldSize() * 0.5f);
    c(1)                = hf.height(c(0), c(2));
    const vec3 eye      = c + vec3(0, 40, 120);
    auto a              = Capture(world, eye, c, 3);
    const auto sa       = world.forest().stats();
    const auto meshes   = world.forest().meshes();
    const auto material = world.forest().material();
    ASSERT_GT(sa.visible[0] + sa.visible[1] + sa.visible[2], 0u);

    world.destroy();
    Attach(world, params);
    EXPECT_EQ(world.forest().meshes(), meshes);
    EXPECT_EQ(world.forest().material(), material);
    auto b        = Capture(world, eye, c, 3);
    const auto sb = world.forest().stats();
    EXPECT_EQ(sb.visible, sa.visible);
    EXPECT_EQ(differingPixels(a, b), 0);

    // Another seed: other species meshes (their names carry the seed), the same material
    world.destroy();
    params.seed = 2;
    Attach(world, params);
    EXPECT_NE(world.forest().meshes(), meshes);
    EXPECT_EQ(world.forest().material(), material);
    Capture(world, eye, c, 2);
}

// S19a review (untested path): setPrototypeTerrain again with an attached forest (waits for the device, rebinds the
// foliage texture of the instance draws) gives the same image and tree counts
TEST_F(VulkanWorldScene, ReplacePrototypeTerrain)
{
    WorldRenderer world;
    WorldParameters params;
    params.terrain.size = 257;
    params.treeCount    = 2000;
    Attach(world, params);
    auto& hf       = world.heightField();
    vec3 c         = vec3(hf.worldSize() * 0.5f, 0, hf.worldSize() * 0.5f);
    c(1)           = hf.height(c(0), c(2));
    const vec3 eye = c + vec3(0, 40, 120);
    auto a         = Capture(world, eye, c, 3);
    const auto sa  = world.forest().stats();
    const auto ta  = scene->statistics().terrainNodes;
    ASSERT_GT(sa.visible[0] + sa.visible[1] + sa.visible[2], 0u);
    scene->setPrototypeTerrain(world.sharedHeightField(), world.settings.prototype(), params.seed);
    auto b = Capture(world, eye, c, 3);
    EXPECT_EQ(world.forest().stats().visible, sa.visible);
    EXPECT_EQ(scene->statistics().terrainNodes, ta);
    EXPECT_EQ(differingPixels(a, b), 0);
}

// S19a review (untested path): a resize recreates the depth image and the SAMPLE_ZERO depth resolve target; the
// prototype then renders at the new size, with the pre-pass the same image (validation clean)
TEST_F(VulkanWorldScene, ResizeWithPrototype)
{
    WorldRenderer world;
    WorldParameters params;
    params.terrain.size = 257;
    params.treeCount    = 2000;
    Attach(world, params);
    auto& hf       = world.heightField();
    vec3 c         = vec3(hf.worldSize() * 0.5f, 0, hf.worldSize() * 0.5f);
    c(1)           = hf.height(c(0), c(2));
    const vec3 eye = c + vec3(0, 40, 120);
    Capture(world, eye, c, 2);
    window->setSize(160, 80);
    auto without = Capture(world, eye, c, 3);
    EXPECT_EQ(without.w, 160);
    EXPECT_EQ(without.h, 80);
    EXPECT_GT(scene->statistics().terrainNodes, 0u);
    world.settings.depthPrePass = true;
    auto with                   = Capture(world, eye, c, 3);
    EXPECT_EQ(with.w, 160);
    EXPECT_EQ(differingPixels(with, without), 0);
}

// S19a review (untested path): the trees toggle in consecutive frames (the sets removed and added while earlier frames
// are in flight) ends in the same image and counts
TEST_F(VulkanWorldScene, TreesToggleInFlight)
{
    WorldRenderer world;
    WorldParameters params;
    params.terrain.size = 513;
    params.treeCount    = 5000;
    Attach(world, params);
    auto& hf       = world.heightField();
    vec3 c         = vec3(hf.worldSize() * 0.5f, 0, hf.worldSize() * 0.5f);
    c(1)           = hf.height(c(0), c(2));
    const vec3 eye = c + vec3(0, 60, 200);
    auto a         = Capture(world, eye, c, 3);
    const auto sa  = world.forest().stats();
    ASSERT_GT(sa.visible[0] + sa.visible[1] + sa.visible[2], 0u);
    // Submitted without a screenshot (its readback waits for the frame): the frames overlap on the GPU
    for (int i = 0; i < 6; ++i)
    {
        world.settings.trees = i % 2 == 1;
        world.update();
        scene->renderFrame([](vk::CommandBuffer) {}, [](vk::CommandBuffer) {}, nullptr);
    }
    world.settings.trees = true;
    auto b               = Capture(world, eye, c, 3);
    EXPECT_EQ(world.forest().stats().visible, sa.visible);
    EXPECT_EQ(differingPixels(a, b), 0);
}

// Regression (S19a review): the frame read the prototype settings by reference and took the pre-pass flag before the
// transfer callback: a change in the render callback (here the sky off) reached the frame's later draws. A frame uses
// one snapshot taken after the transfer callback; the change is the next frame's
TEST_F(VulkanWorldScene, PrototypeSettingsSnapshotPerFrame)
{
    WorldRenderer world;
    WorldParameters params;
    params.terrain.size = 257;
    params.treeCount    = 0;
    Attach(world, params);
    auto& hf          = world.heightField();
    vec3 eye          = vec3(hf.worldSize() * 0.5f, 0, hf.worldSize() * 0.5f);
    eye(1)            = std::max(hf.height(eye(0), eye(2)), hf.waterLevel()) + 30;
    const vec3 target = eye + vec3(1, 0.1f, 0);
    auto a            = Capture(world, eye, target, 2);
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
    // Off in the render callback: this frame still has the sky, the next one not
    auto b = frame(noop, [&](vk::CommandBuffer) { scene->setPrototypeSettings(off); });
    EXPECT_EQ(differingPixels(a, b), 0);
    auto c = frame(noop, noop);
    EXPECT_GT(differingPixels(a, c), 100) << "the sky is not switched off";
    // On in the transfer callback: this frame has it
    PrototypeSettings on = off;
    on.sky               = true;
    auto d               = frame([&](vk::CommandBuffer) { scene->setPrototypeSettings(on); }, noop);
    EXPECT_EQ(differingPixels(a, d), 0);
}

// Regression (S19a review): a WorldRenderer attached again kept the previous attachment's statistics cursors: its
// forest (the same mesh IDs) reported the old frame's tree counts before its first frame, and the GPU times of a new
// scene (frame numbers from 1 again) were rejected until it passed the old scene's frame number
TEST_F(VulkanWorldScene, ReattachStartsFreshStatistics)
{
    WorldRenderer world;
    WorldParameters params;
    params.terrain.size = 257;
    params.treeCount    = 2000;
    Attach(world, params);
    auto& hf       = world.heightField();
    vec3 c         = vec3(hf.worldSize() * 0.5f, 0, hf.worldSize() * 0.5f);
    c(1)           = hf.height(c(0), c(2));
    const vec3 eye = c + vec3(0, 40, 120);
    SetCamera(world, eye, c);
    auto frames = [&](int n)
    {
        for (int i = 0; i < n; ++i)
        {
            RenderFrame(world);
            FrameTimes t;
            while (scene->takeGpuFrameTimes(t))
            {
            }
        }
        scene->waitIdle();
        world.update();
    };
    frames(10);
    const auto sa = world.forest().stats();
    ASSERT_GT(sa.visible[0] + sa.visible[1] + sa.visible[2], 0u);
    EXPECT_GE(world.stats().gpuMs, 0);

    // The same scene and seed without trees: no counts before its first frame, none after it
    world.destroy();
    params.treeCount = 0;
    Attach(world, params);
    auto s0 = world.forest().stats();
    EXPECT_EQ(s0.frame, 0u);
    EXPECT_EQ(s0.visible, (std::array<uint32_t, 3>{0, 0, 0}));
    SetCamera(world, eye, c);
    frames(2);
    s0 = world.forest().stats();
    EXPECT_GT(s0.frame, 0u);
    EXPECT_EQ(s0.visible, (std::array<uint32_t, 3>{0, 0, 0}));

    // A new scene: its GPU times count from its first frames
    world.destroy();
    Create(96, 64, 4);
    params.treeCount = 2000;
    Attach(world, params);
    SetCamera(world, eye, c);
    frames(4);
    EXPECT_GE(world.stats().gpuMs, 0);
    EXPECT_EQ(world.forest().stats().visible, sa.visible);
}

// S19a: the PrototypeFoliage class needs the foliage texture of setPrototypeTerrain (asserted at addMaterial and
// setMaterial, no frame needed)
TEST_F(VulkanWorldScene, FoliageClassNeedsTexture)
{
#ifdef SAIGA_ASSERTS
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    MaterialData foliage;
    foliage.pipelineClass = uint8_t(PipelineClass::PrototypeFoliage);
    EXPECT_DEATH(scene->addMaterial(1, "foliage", foliage), "needs the foliage texture of setPrototypeTerrain");
    scene->addMaterial(1, "flat", MaterialData());
    EXPECT_DEATH(scene->setMaterial(1, foliage), "needs the foliage texture of setPrototypeTerrain");
#endif
}

// Golden images of the prototype world (docs/engine_design.md §17.6): 512 x 512, clay shading, time 0, a 1 km world
// with 5000 trees, seed 1, captured after 5 frames. They gate on lavapipe only; other devices print the metrics. The
// drawn instance and terrain node counts gate on lavapipe only as well: the tree placement uses std random
// distributions and float comparisons, so other standard libraries, compilers (FMA, -march=native) or drivers may place
// or cull a few trees differently. S17: rendered through RenderScene's frame (clay mode: the hardware average resolve,
// the identity tone map). S19a: the prototype inside RenderScene (setPrototypeTerrain, the trees as instance sets of
// prototypeTreeSets, the depth pre-pass), nearPlane 0.1, fovy 60; the counts are also compared with the CPU reference
// of the culling rule (§17.5 band).
class VulkanWorldGolden : public VulkanWorldScene
{
   protected:
    static constexpr int size = 512;

    void SetUp() override
    {
        VulkanWorldScene::SetUp();
        if (IsSkipped()) return;
        Create(size, size, 4);
    }

    void InitWorld(WorldRenderer& world)
    {
        WorldParameters params;
        params.terrain.size    = 513;  // 2 m spacing: 1 km
        params.treeCount       = 5000;
        params.seed            = 1;
        world.settings.shading = WorldShading::Clay;
        Attach(world, params);
        scene->setTime(0);
    }

    // The committed counts (docs/engine_design.md §17.6)
    void ExpectCounts(WorldRenderer& world, const char* name, int lod0, int lod1, int lod2, int terrainNodes)
    {
        const auto fs  = world.forest().stats();
        const auto& st = scene->statistics();
        const auto ref = Reference(world);
        std::cout << name << " trees per lod: " << fs.visible[0] << " " << fs.visible[1] << " " << fs.visible[2]
                  << ", terrain nodes " << st.terrainNodes << " (reference " << ref.perLod[0] << " " << ref.perLod[1]
                  << " " << ref.perLod[2] << ", " << ref.ambiguous << " ambiguous)" << std::endl;
        EXPECT_EQ(fs.frame, scene->frames());
        expectForest(fs, ref, name);
        // Not checked while the goldens are regenerated: the printed counts replace the committed ones
        if (!goldenGating(deviceName()) || goldenUpdateMode()) return;
        EXPECT_EQ(int(fs.visible[0]), lod0);
        EXPECT_EQ(int(fs.visible[1]), lod1);
        EXPECT_EQ(int(fs.visible[2]), lod2);
        EXPECT_EQ(int(st.terrainNodes), terrainNodes);
    }
};

TEST_F(VulkanWorldGolden, WorldOverview)
{
    WorldRenderer world;
    InitWorld(world);
    auto& hf = world.heightField();
    // 300 m above the ground 100 m from the west edge (at the center z), looking 45 degrees down along +x: the terrain
    // and the trees fill the image, only a thin band of sky and the far edge of the world remain at the top
    vec3 c(100, 0, hf.worldSize() * 0.5f);
    c(1)     = hf.height(c(0), c(2));
    vec3 eye = c + vec3(0, 300, 0);
    vec3 dir(std::cos(radians(45.0f)), -std::sin(radians(45.0f)), 0);
    auto img = Capture(world, eye, eye + dir);
    EXPECT_EQ(img.w, 512);
    compareGolden(img, "world_overview", deviceName());
    ExpectCounts(world, "world_overview", 0, 7, 3575, 45);
}

TEST_F(VulkanWorldGolden, WorldForest)
{
    WorldRenderer world;
    InitWorld(world);
    auto& hf = world.heightField();
    // 2 m above the ground at the edge of a clearing 28 m west and 80 m north of the center (the nearest tree is 15 m
    // away), looking along +x and slightly down: a close tree with alpha-to-coverage foliage, grass, trunks, the
    // forest at mid distance and the sky
    vec3 eye(hf.worldSize() * 0.5f - 28, 0, hf.worldSize() * 0.5f - 80);
    eye(1)   = std::max(hf.height(eye(0), eye(2)), hf.waterLevel()) + 2.0f;
    auto img = Capture(world, eye, eye + vec3(1, -0.2f, 0));
    EXPECT_EQ(img.w, 512);
    compareGolden(img, "world_forest", deviceName());
    ExpectCounts(world, "world_forest", 43, 281, 677, 38);
}

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

// S19a: on RenderScene (the statistics come from the scene; the settings follow through WorldRenderer::update)
TEST_F(VulkanWorldScene, GameTools)
{
    using Json = McpServer::Json;
    WorldRenderer world;
    WorldParameters params;
    params.terrain.size = 129;
    params.treeCount    = 100;
    Attach(world, params);
    // S19b: the tools reach the app's view camera through callbacks (a tool write, no hold)
    Vulkan::WorldMcpCamera camera;
    camera.position  = Vec3(10, 50, 10);
    camera.rotation  = Quat(AngleAxis(0.3, Vec3(0, 1, 0))) * Quat(AngleAxis(-0.5, Vec3(1, 0, 0)));
    int cameraWrites = 0;
    bool walkMode    = false;
    float eyeHeight  = 1.7f;
    McpServer server("test");
    Vulkan::WorldMcpContext context;
    context.world     = &world;
    context.getCamera = [&]() { return camera; };
    context.setCamera = [&](const Vulkan::WorldMcpCamera& c)
    {
        camera = c;
        cameraWrites++;
    };
    context.walkMode  = &walkMode;
    context.eyeHeight = &eyeHeight;
    Vulkan::registerWorldMcpTools(server, context);
    const HeightField& hf = world.heightField();

    // game_get_state: the extent, statistics, settings and the camera's ground state
    Capture(world, vec3(10, 50, 10), vec3(128, 0, 128), 3);
    // The app takes the GPU times (EngineApp); here the test
    FrameTimes gpuTimes;
    bool gpuTaken = false;
    while (scene->takeGpuFrameTimes(gpuTimes)) gpuTaken = true;
    world.update();
    Json s = toolResultJson(callWorldTool(server, "game_get_state", Json::object{}));
    EXPECT_EQ(s["world"]["size_m"].number_value(), hf.worldSize()) << s.dump();
    // S19a renamed gpu_world_ms / gpu_pass_ms (the prototype's own passes) to RenderScene's frame and sections
    EXPECT_TRUE(s["stats"]["gpu_frame_ms"].is_number()) << s.dump();
    EXPECT_TRUE(s["stats"]["gpu_section_ms"].is_object()) << s.dump();
    EXPECT_TRUE(s["stats"]["gpu_world_ms"].is_null()) << s.dump();
    EXPECT_TRUE(s["stats"]["gpu_pass_ms"].is_null()) << s.dump();
    if (gpuTaken)
    {
        EXPECT_GE(s["stats"]["gpu_frame_ms"].number_value(), 0) << s.dump();
        for (const char* name : RenderScene::gpuSectionNames())
            EXPECT_TRUE(s["stats"]["gpu_section_ms"][name].is_number()) << name << " " << s.dump();
    }
    // The per-LOD tree counts and triangles from the tracked meshes, the trees and the terrain nodes
    const auto fs = world.forest().stats();
    EXPECT_EQ(s["stats"]["trees_total"].int_value(), world.forest().treeCount()) << s.dump();
    EXPECT_EQ(s["stats"]["trees_visible_lod0"].int_value(), int(fs.visible[0])) << s.dump();
    EXPECT_EQ(s["stats"]["trees_visible_lod2"].int_value(), int(fs.visible[2])) << s.dump();
    EXPECT_EQ(s["stats"]["tree_triangles"].number_value(), double(fs.triangles)) << s.dump();
    EXPECT_GT(s["stats"]["terrain_nodes"].int_value(), 0) << s.dump();
    EXPECT_TRUE(s["settings"].is_object());
    EXPECT_TRUE(s["camera"]["ground_height"].is_number());
    EXPECT_FALSE(s["camera"]["walk_mode"].bool_value());

    // game_set_settings: valid, clamped and invalid keys
    s = toolResultJson(callWorldTool(server, "game_set_settings", Json::object{{"fovy", 1000.0}, {"nope", 1}}));
    EXPECT_LT(s["settings"]["fovy"].number_value(), 1000) << s.dump();  // clamped to the GUI range
    EXPECT_EQ(s["invalid_keys"][0].string_value(), "nope") << s.dump();
    // The field of view is the app camera's (S19b)
    EXPECT_EQ(cameraWrites, 1);
    EXPECT_NEAR(camera.fovy, s["settings"]["fovy"].number_value() * pi<double>() / 180.0, 1e-6);
    // A setting reaches RenderScene with the next update (game_world calls it every frame)
    callWorldTool(server, "game_set_settings", Json::object{{"grass", false}});
    world.update();
    EXPECT_FALSE(scene->prototypeSettings().grass);

    // game_terrain_query: the field's heights, limits and errors
    const float x = hf.worldSize() * 0.25f, z = hf.worldSize() * 0.75f;
    // Nested arrays built explicitly: Json::array{Json::array{...}} is a copy, not a nesting, with Clang
    Json::array points;
    points.push_back(Json::array{x, z});
    Json q = toolResultJson(callWorldTool(server, "game_terrain_query", Json::object{{"points", points}}));
    EXPECT_NEAR(q[0]["height"].number_value(), hf.height(x, z), 0.01) << q.dump();
    Json::array badPoints;
    badPoints.push_back(Json::array{1});
    EXPECT_TRUE(callWorldTool(server, "game_terrain_query", Json::object{{"points", badPoints}})["result"]["isError"]
                    .bool_value());

    // game_set_camera: [x, z] with a height above ground keeps the view direction; walk mode on and off
    const Quat rotation = camera.rotation;
    Json c              = toolResultJson(callWorldTool(
        server, "game_set_camera", Json::object{{"position", Json::array{x, z}}, {"height_above_ground", 30.0}}));
    EXPECT_NEAR(c["height_above_ground"].number_value(), 30, 0.02) << c.dump();
    EXPECT_NEAR(c["ground_height"].number_value(), hf.height(x, z), 0.02) << c.dump();
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
    EXPECT_TRUE(
        callWorldTool(server, "game_set_camera", Json::object{{"yaw_deg", 3}})["result"]["isError"].bool_value());
}
