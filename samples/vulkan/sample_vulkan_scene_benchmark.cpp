/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

/**
 * The culling benchmark of RenderScene (S18, docs/engine_design.md §17.7, R-TEST-2): the prototype forest's 200k trees
 * (ForestRenderer::placeTrees on the default height field, one instance set per chunk) as flat meshes with the
 * prototype trees' triangle counts, LOD distances (70, 300, 30000 m) and bounds, drawn by RenderScene over the flyover
 * path of WorldBenchmark (1920 x 1080, 4x MSAA, clay mode, headless: no window, no present).
 *
 * Writes benchmark_scene_cull.json to the working directory: the GPU times of RenderScene's cull section (count, scan,
 * write) and scene section (the instance draws and the clear / MSAA store), the scene section of frames without a
 * camera (scene_empty_ms: nothing culled or drawn) and their difference (draw_ms), the transfer section, the drawn
 * instances per LOD and the draw calls (medians and p95 over the measured frames).
 *
 *   ./sample_vulkan_scene_benchmark [--instances N]
 */

#include "saiga/core/util/json11.hpp"
#include "saiga/vulkan/scene/RenderScene.h"
#include "saiga/vulkan/window/HeadlessWindow.h"
#include "saiga/vulkan/world/FoliageGenerator.h"
#include "saiga/vulkan/world/ForestRenderer.h"
#include "saiga/vulkan/world/HeightField.h"
#include "saiga/vulkan/world/WorldBenchmark.h"

#include <cmath>
#include <iostream>
#include <map>
#include <string>

using namespace Saiga;
using namespace Saiga::Vulkan;

namespace
{
constexpr int width = 1920, height = 1080;
constexpr int emptyFrames = 120;

// A flat mesh with the triangles of a prototype tree mesh (unit height): faces duplicated, face normals; the opaque
// parts brown, the leaf cards green (no alpha test: the flat model)
MeshLod flatLod(const FoliageMesh& mesh, float endDistance)
{
    MeshLod lod;
    lod.endDistance = endDistance;
    for (size_t t = 0; t + 2 < mesh.indices.size(); t += 3)
    {
        vec3 p[3];
        for (int k = 0; k < 3; ++k) p[k] = mesh.vertices[mesh.indices[t + k]].posAo.head<3>();
        vec3 n = (p[1] - p[0]).cross(p[2] - p[0]);
        if (!(n.norm() > 0)) n = vec3(0, 1, 0);
        const bool leaf     = mesh.vertices[mesh.indices[t]].uvLayer(2) >= 0;
        const ucvec4 color  = leaf ? ucvec4(70, 120, 50, 255) : ucvec4(110, 80, 60, 255);
        const uint32_t base = uint32_t(lod.vertices.size());
        for (int k = 0; k < 3; ++k)
        {
            MeshVertex v;
            v.position = p[k];
            v.normal   = packNormalOct(n.normalized());
            v.color    = packColor(color);
            lod.vertices.push_back(v);
            lod.indices.push_back(base + k);
        }
    }
    return lod;
}

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
    return make_quat(R).normalized();
}

json11::Json stat(const std::vector<double>& v)
{
    const auto s = WorldBenchmark::summarize(v);
    return json11::Json::object{{"median", s.median}, {"p95", s.p95}};
}
}  // namespace

int main(int argc, char** argv)
{
    int instanceCount = 200000;
    for (int i = 1; i + 1 < argc; ++i)
        if (std::string(argv[i]) == "--instances") instanceCount = std::stoi(argv[i + 1]);

    // The prototype's content: the default height field, the trees of seed 1, the LOD distances of WorldSettings
    HeightField hf;
    hf.generate(HeightField::Parameters());
    const uint32_t seed = 1;
    const auto species  = FoliageGenerator::treeSpecies(seed);
    const auto trees    = ForestRenderer::placeTrees(hf, instanceCount, seed, species);
    const float ends[3] = {70, 300, 30000};

    WindowParameters wp;
    wp.width  = width;
    wp.height = height;
    wp.saigaParameters.shaderDirectory.push_back(SAIGA_PROJECT_SOURCE_DIR "/shader");
    wp.imguiParameters.enable = false;
    VulkanParameters vp;
    vp.enableImgui           = false;
    vp.enableValidationLayer = false;
    RenderSceneParameters params;
    params.imgui                  = false;
    params.msaaSamples            = 4;
    params.staticInstanceCapacity = uint32_t(trees.size());
    HeadlessWindow window(wp);
    RenderScene scene(window, vp, params);

    MaterialData material;
    material.twoSided = true;  // leaf cards
    scene.addMaterial(1, "tree", material);
    int triangles[4][3] = {};
    for (size_t s = 0; s < species.size(); ++s)
    {
        MeshLods lods;
        for (int l = 0; l < 3; ++l)
        {
            lods.lods.push_back(flatLod(species[s].lods[l], ends[l]));
            triangles[s][l] = species[s].lods[l].triangles();
        }
        lods.sphereCenter  = vec3(0, species[s].bounds(0), 0);
        lods.sphereRadius  = species[s].bounds(1);
        lods.boundsPadding = 0.025f;  // the prototype's wind sway allowance (tree_cull.comp)
        scene.addMesh(MeshId(s + 1), species[s].name, lods, 1);
    }

    // One instance set per chunk (level-0 regions), region-relative positions
    std::map<std::pair<int, int>, std::vector<StaticInstance>> sets;
    for (const auto& t : trees)
    {
        const int cx = int(std::floor(t.posYaw(0) / 256.0f)), cz = int(std::floor(t.posYaw(2) / 256.0f));
        StaticInstance si;
        si.setPosition(vec3(t.posYaw(0) - cx * 256.0f, t.posYaw(1), t.posYaw(2) - cz * 256.0f));
        const float yaw = t.posYaw(3);
        si.setRotation(quat(std::cos(yaw / 2), 0, std::sin(yaw / 2), 0));
        si.scale = t.scaleSpecies(0);
        si.mesh  = MeshId(int(t.scaleSpecies(1)) + 1);
        sets[{cx, cz}].push_back(si);
    }
    for (auto& [c, v] : sets) scene.addInstanceSet(RegionId{0, 0, c.first, c.second}, v);
    std::cout << "sample_vulkan_scene_benchmark: " << trees.size() << " instances in " << sets.size() << " sets"
              << std::endl;

    const auto& names = RenderScene::gpuSectionNames();
    auto section      = [&](const char* n)
    {
        for (size_t i = 0; i < names.size(); ++i)
            if (std::string(names[i]) == n) return i;
        return size_t(0);
    };
    const size_t sCull = section("cull"), sScene = section("scene"), sTransfer = section("transfer");

    // Frames without a camera (nothing culled or drawn), then warm-up and the measured frames of the flyover path
    const int warmup = WorldBenchmark::warmupFrames, measured = WorldBenchmark::measuredFrames;
    const uint64_t firstEmpty = scene.frames() + 1, lastEmpty = firstEmpty + emptyFrames - 1;
    const uint64_t firstMeasured = lastEmpty + warmup + 1, lastMeasured = firstMeasured + measured - 1;
    std::vector<double> cullMs, sceneMs, emptyMs, transferMs, recordingMs, drawCalls;
    std::array<std::vector<double>, 4> perLod;
    auto collect = [&]()
    {
        FrameTimes t;
        while (scene.takeGpuFrameTimes(t))
        {
            if (t.frame >= firstEmpty && t.frame <= lastEmpty) emptyMs.push_back(t.gpuSectionMs[sScene]);
            if (t.frame >= firstMeasured && t.frame <= lastMeasured)
            {
                cullMs.push_back(t.gpuSectionMs[sCull]);
                sceneMs.push_back(t.gpuSectionMs[sScene]);
                transferMs.push_back(t.gpuSectionMs[sTransfer]);
            }
        }
        // Every completed frame's statistics (several frames may complete between two calls)
        RenderStatistics st;
        while (scene.takeStatistics(st))
        {
            if (st.frame < firstMeasured || st.frame > lastMeasured) continue;
            for (int l = 0; l < 4; ++l) perLod[l].push_back(st.instancesPerLod[l]);
            drawCalls.push_back(st.drawCommands);
        }
    };
    auto noop = [](vk::CommandBuffer) {};
    for (int i = 0; i < emptyFrames; ++i)
    {
        scene.renderFrame(noop, noop, nullptr);
        collect();
    }
    for (int i = 0; i < warmup + measured; ++i)
    {
        const int m                  = std::max(0, i - warmup);
        const WorldBenchmark::Pose p = WorldBenchmark::pose("flyover", hf, m);
        const Vec3 eye               = p.eye.cast<double>();
        scene.setCamera(0, eye, lookRotation(p.target.cast<double>() - eye), 60.0 * 3.14159265358979323846 / 180.0);
        scene.renderFrame(noop, noop, nullptr);
        if (i >= warmup) recordingMs.push_back(scene.lastRecordingMs());
        collect();
    }
    // The GPU times of a frame arrive when its frame slot is reused: a few more frames
    for (int i = 0; i < 8; ++i)
    {
        scene.renderFrame(noop, noop, nullptr);
        collect();
    }
    scene.waitIdle();

    const std::string gpu = scene.base().physicalDevice.getProperties().deviceName.data();
    json11::Json::object perLodJson;
    for (int l = 0; l < 3; ++l) perLodJson["lod" + std::to_string(l)] = stat(perLod[l]);
    std::vector<double> drawMs;
    const double emptyMedian = WorldBenchmark::summarize(emptyMs).median;
    for (double v : sceneMs) drawMs.push_back(v - emptyMedian);
    json11::Json::array tri;
    for (size_t s = 0; s < species.size(); ++s)
        tri.push_back(json11::Json::array{triangles[s][0], triangles[s][1], triangles[s][2]});
    json11::Json result = json11::Json::object{{"gpu", gpu},
                                               {"build", WorldBenchmark::buildInfo()},
                                               {"instances", int(trees.size())},
                                               {"sets", int(sets.size())},
                                               {"triangles_per_lod", tri},
                                               {"frames", int(cullMs.size())},
                                               {"width", scene.size()(0)},
                                               {"height", scene.size()(1)},
                                               {"msaa", int(scene.formats().samples)},
                                               {"multi_draw_indirect", scene.multiDrawIndirect()},
                                               {"cull_ms", stat(cullMs)},
                                               {"scene_ms", stat(sceneMs)},
                                               {"scene_empty_ms", stat(emptyMs)},
                                               {"draw_ms", stat(drawMs)},
                                               {"transfer_ms", stat(transferMs)},
                                               {"recording_ms", stat(recordingMs)},
                                               {"instances_per_lod", perLodJson},
                                               {"draw_calls", stat(drawCalls)}};
    const bool complete =
        int(cullMs.size()) == measured && int(emptyMs.size()) == emptyFrames && int(drawCalls.size()) == measured;
    if (!complete)
        std::cout << "Benchmark error: GPU times of " << cullMs.size() << " and statistics of " << drawCalls.size()
                  << " of " << measured << " frames" << std::endl;
    const bool written = WorldBenchmark::writeResult("benchmark_scene_cull.json", result);
    std::cout << result.dump() << std::endl;
    return complete && written ? 0 : 1;
}
