/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/scene/PrototypeWorld.h"

#include "saiga/vulkan/scene/SceneCulling.h"
#include "saiga/vulkan/world/FoliageGenerator.h"

#include <chrono>
#include <cmath>

namespace Saiga
{
namespace Vulkan
{
using Stage  = vk::PipelineStageFlagBits2;
using Access = vk::AccessFlagBits2;

namespace
{
constexpr double twoPi = 6.283185307179586;

// Towards the sun from the azimuth and elevation (degrees), as the prototype
vec3 sunDirection(const PrototypeSettings& s)
{
    float az = radians(s.sunAzimuth), el = radians(s.sunElevation);
    return vec3(std::cos(el) * std::sin(az), std::sin(el), -std::cos(el) * std::cos(az));
}
}  // namespace

uint32_t packPrototypeFoliageColor(int layer, float ao, float wind)
{
    SAIGA_ASSERT(layer < 255, "packPrototypeFoliageColor: the layer must be < 255 (255: opaque)");
    SAIGA_ASSERT(ao >= 0 && ao <= 1, "packPrototypeFoliageColor: the AO must be in [0, 1]");
    SAIGA_ASSERT(wind >= 0 && wind <= 1, "packPrototypeFoliageColor: the wind weight must be in [0, 1]");
    const uint32_t l = layer < 0 ? 0xFFu : uint32_t(layer);
    const uint32_t a = uint32_t(std::lround(std::min(std::max(ao, 0.0f), 1.0f) * 255.0f));
    const uint32_t w = uint32_t(std::lround(std::min(std::max(wind, 0.0f), 1.0f) * 255.0f));
    return l | (a << 8) | (w << 16);
}

float PrototypeWorld::windPhase(double time, double frequency)
{
    double p = std::fmod(time * frequency, twoPi);
    if (p < 0) p += twoPi;
    return float(p);
}

void PrototypeWorld::init(VulkanBase& _base, const RenderingFormats& formats,
                          std::shared_ptr<const HeightField> heights, uint32_t foliageSeed)
{
    destroy();
    SAIGA_ASSERT(heights, "setPrototypeTerrain: no height field");
    base    = &_base;
    hf      = std::move(heights);
    samples = int(formats.samples);

    uniformBuffer.init(*base, nullptr, sizeof(WorldUniforms));
    {
        auto layers = FoliageGenerator::foliageTextures(512, foliageSeed);
        std::vector<const Image*> images;
        for (auto& l : layers) images.push_back(&l);
        foliage.fromImages(*base, images, vk::ImageUsageFlagBits::eSampled, false, true);
    }
    terrain.init(*base, formats, uniformBuffer, *hf, 64.0f);
    grass.init(*base, formats, uniformBuffer, foliage, terrain);

    // Sky: a full screen triangle at the far plane (depth 0), only where nothing was drawn
    {
        DescriptorSetLayout layout(
            {{0, {0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eFragment}}});
        PipelineInfo info                       = worldPipelineInfo();
        info.rasterizationState.cullMode        = vk::CullModeFlagBits::eNone;
        info.depthStencilState.depthCompareOp   = vk::CompareOp::eEqual;
        info.depthStencilState.depthWriteEnable = VK_FALSE;
        createWorldPipeline(skyPipeline, *base, layout, 0, {"vulkan/world/sky.vert", "vulkan/world/sky.frag"}, formats,
                            info);
        skySet = skyPipeline.createDescriptorSet();
        skySet.assign(0, &uniformBuffer);
    }
    statistics  = Stats();
    frameValid  = false;
    initialized = true;
}

void PrototypeWorld::destroy()
{
    if (!initialized) return;
    terrain.destroy();
    grass.destroy();
    skyPipeline.destroy();
    foliage.destroy();
    uniformBuffer.destroy();
    hf.reset();
    initialized = false;
}

void PrototypeWorld::transfer(vk::CommandBuffer cmd, const SceneCamera& camera, const PrototypeSettings& s, double time,
                              int width, int height)
{
    statistics = Stats();
    // Nothing is drawn before the first setCamera (the node lists and uniforms of this frame are not written)
    frameValid = camera.valid;
    if (!camera.valid) return;
    const vec3 cameraPos = camera.position.cast<float>();
    // The world-space frustum of the node selection and the grass tiles (float, as the prototype): the camera-relative
    // planes n.(p - c) + w moved to n.p + (w - n.c)
    std::array<vec4, 6> frustum;
    for (int k = 0; k < 5; ++k)
    {
        const Vec4& p = camera.planes[k];
        const Vec3 n(p(0), p(1), p(2));
        frustum[k] = vec4(float(p(0)), float(p(1)), float(p(2)), float(p(3) - n.dot(camera.position)));
    }
    frustum[5] = vec4(0, 0, 0, 1);

    auto t0 = std::chrono::steady_clock::now();
    terrain.update(cmd, cameraPos, frustum, s.terrainLodRange, s.morphStart, s.water);
    auto t1 = std::chrono::steady_clock::now();
    grass.update(cameraPos, frustum, s.grass ? s.grassRadius : 0, s.grassCellSize);
    auto t2 = std::chrono::steady_clock::now();

    const vec3 sun          = sunDirection(s);
    const float aspect      = float(width) / float(std::max(height, 1));
    uniforms.viewProjRel    = camera.viewProj;
    uniforms.invViewProjRel = inverse(camera.viewProj);
    uniforms.cameraPos      = vec4(cameraPos(0), cameraPos(1), cameraPos(2), 0);
    uniforms.lodCameraPos   = vec4(cameraPos(0), cameraPos(1), cameraPos(2), 0);
    uniforms.sunDir         = vec4(sun(0), sun(1), sun(2), s.sunIntensity);
    uniforms.shading  = vec4(s.albedo, s.ambient, s.flatShading ? 1 : 0, camera.linear ? camera.preExposure : 0.0f);
    uniforms.fog      = vec4(s.fogDensity, s.fogFalloff, s.fog ? 1 : 0, hf->waterLevel());
    uniforms.fogColor = vec4(0.42f, 0.42f, 0.42f, 1);
    uniforms.skyColor = vec4(0.2f, 0.2f, 0.2f, 1);
    for (int i = 0; i < 6; ++i) uniforms.frustum[i] = frustum[i];
    uniforms.terrain     = vec4(float(hf->size()), hf->spacing(), terrain.range0(), s.morphStart);
    uniforms.terrain2    = vec4(float(terrain.maxLevel()), float(s.debugMode), s.wind, s.alphaCutoff);
    uniforms.trees       = vec4(0, 0, 0, 0);
    uniforms.grass       = vec4(s.grassRadius, s.grassCellSize, s.grassDensity, float(samples));
    uniforms.viewport    = vec4(aspect, 1, 1.0f / aspect, 1);
    uniforms.grassPhases = vec4(windPhase(time, 1.7), windPhase(time, 1.3), 0, 0);
    // In the transfer section: RenderScene's barriers order it after the previous frames' reads and before this
    // frame's draws
    uniformBuffer.update(cmd, sizeof(WorldUniforms), &uniforms);

    auto& ts                   = terrain.stats();
    statistics.terrainNodes    = ts.nodes;
    statistics.quarterNodes    = ts.quarterNodes;
    statistics.waterNodes      = ts.waterNodes;
    statistics.terrainMaxLevel = terrain.maxLevel();
    statistics.waterTriangles  = ts.waterNodes * TerrainRenderer::gridResolution * TerrainRenderer::gridResolution / 2;
    // As the prototype: with the water plane
    statistics.terrainTriangles = s.terrain ? ts.triangles : 0;
    for (int l = 0; l < 16; ++l)
    {
        statistics.nodesPerLevel[l]     = ts.nodesPerLevel[l];
        statistics.trianglesPerLevel[l] = ts.trianglesPerLevel[l];
    }
    statistics.grassTiles     = grass.stats().tiles;
    statistics.grassClumps    = grass.stats().clumps;
    statistics.grassTriangles = grass.stats().triangles;
    statistics.drawCalls      = ((s.terrain || s.water) ? terrain.drawCalls(s.terrain) : 0) +
                           (s.grass ? grass.drawCalls() : 0) + (s.sky ? 1 : 0);
    statistics.cpuTerrainSelectMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    statistics.cpuGrassCullMs     = std::chrono::duration<double, std::milli>(t2 - t1).count();
}

void PrototypeWorld::renderDepth(vk::CommandBuffer cmd, const PrototypeSettings& s)
{
    // Without terrain only the water plane
    if (frameValid && (s.terrain || s.water)) terrain.render(cmd, TerrainRenderer::Mode::Depth, s.terrain);
}

void PrototypeWorld::renderTerrain(vk::CommandBuffer cmd, const PrototypeSettings& s, bool afterPrepass)
{
    if (frameValid && (s.terrain || s.water))
        terrain.render(cmd, afterPrepass ? TerrainRenderer::Mode::Equal : TerrainRenderer::Mode::Scene, s.terrain);
}

void PrototypeWorld::renderGrass(vk::CommandBuffer cmd, const PrototypeSettings& s)
{
    if (frameValid && s.grass) grass.render(cmd);
}

void PrototypeWorld::renderSky(vk::CommandBuffer cmd, const PrototypeSettings& s)
{
    if (frameValid && s.sky && skyPipeline.bind(cmd))
    {
        skyPipeline.bindDescriptorSet(cmd, skySet);
        cmd.draw(3, 1, 0, 0);
    }
}

void PrototypeWorld::lighting(const PrototypeSettings& s, double time, SceneLighting& out) const
{
    prototypeLighting(s, time, hf->waterLevel(), out);
}

void prototypeLighting(const PrototypeSettings& s, double time, float fogBase, SceneLighting& out)
{
    const vec3 sun = sunDirection(s);
    out.sunDir     = vec4(sun(0), sun(1), sun(2), s.sunIntensity);
    out.ambient    = vec4(s.ambient, 1, s.albedo, s.alphaCutoff);
    out.fog        = vec4(s.fogDensity, s.fogFalloff, s.fog ? 1 : 0, fogBase);
    out.fogColor   = vec4(0.42f, 0.42f, 0.42f, 1);
    // The tree.vert frequencies of the sway (x, z) and the flutter
    out.wind  = vec4(PrototypeWorld::windPhase(time, 1.1), PrototypeWorld::windPhase(time, 0.9),
                     PrototypeWorld::windPhase(time, 4.0), s.wind);
    out.debug = vec4(float(s.debugMode), 0, 0, 0);
}

size_t PrototypeWorld::memoryBytes() const
{
    if (!initialized) return 0;
    // R8 array with a full mip chain (~4/3 of level 0)
    size_t foliageBytes = size_t(foliage.width) * foliage.height * foliage.layerCount * 4 / 3;
    return terrain.memoryBytes() + grass.memoryBytes() + foliageBytes + uniformBuffer.size();
}

}  // namespace Vulkan
}  // namespace Saiga
