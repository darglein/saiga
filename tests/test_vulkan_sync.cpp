/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// RenderScene's frame under the validation layer's synchronization validation (S36a2 review, blocker 1): the process
// turns it on before the first Vulkan instance (VK_LAYER_VALIDATE_SYNC, with the SPIR-V heuristic of the shader
// accesses, without which the layer does not see the descriptor reads of a draw; an environment value set by the user
// is kept). A hazard is a validation error, which aborts the test.
// Sync validation is turned on only with a Khronos validation layer of at least kMinSyncValidationLayer (its
// specVersion, enumerated before the first instance); with an older layer or none the tests only render (they are not
// skipped: CI fails on skipped tests). Ubuntu 24.04's layer (1.3.275, CI) does run sync validation, but without the
// shader-access heuristic, and reports the scene pass's SAMPLE_ZERO depth resolve at vkCmdEndRendering as a
// read-after-write hazard against the pass's own depth writes: one dynamic-rendering pass, where no barrier can be
// placed and the resolve is ordered by the pass itself (a false positive; 1.4.341 accepts it).
// SAIGA_TEST_SYNC_VALIDATION = 1 / 0 forces sync validation on / off. A scene per test (RenderSceneTest), skipped
// without a Vulkan 1.3 device.

#include "saiga/core/math/math.h"
#include "saiga/vulkan/scene/CloudLayer.h"
#include "saiga/vulkan/scene/GrassPass.h"
#include "saiga/vulkan/scene/LightingDebug.h"
#include "saiga/vulkan/scene/LightingTypes.h"
#include "saiga/vulkan/scene/PrototypeWorld.h"
#include "saiga/vulkan/scene/RenderScene.h"
#include "saiga/vulkan/world/WorldCommon.h"

#include "gtest/gtest.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "render_scene_test_context.h"
#include "terrain_test_hooks.h"

using namespace Saiga;
using namespace Saiga::Vulkan;
using namespace TerrainTestHooks;

namespace
{
// Before main: the instance of the first test reads the layer settings from the environment
void setEnv(const char* name, const char* value)
{
#if defined(_WIN32)
    if (!std::getenv(name)) _putenv_s(name, value);
#else
    setenv(name, value, 0);
#endif
}
// The oldest layer used with sync validation: 1.4.328 knows the shader-access heuristic
// (syncval_shader_accesses_heuristic in its sync_settings.h) and is newer than the 1.3.275 of the false positive; the
// local layer (1.4.341) is clean
constexpr uint32_t kMinSyncValidationLayer = VK_MAKE_API_VERSION(0, 1, 4, 328);

bool syncValidationWanted()
{
    if (const char* forced = std::getenv("SAIGA_TEST_SYNC_VALIDATION")) return std::string(forced) != "0";
    uint32_t count = 0;
    if (vkEnumerateInstanceLayerProperties(&count, nullptr) != VK_SUCCESS) return false;
    std::vector<VkLayerProperties> layers(count);
    if (vkEnumerateInstanceLayerProperties(&count, layers.data()) != VK_SUCCESS) return false;
    for (uint32_t i = 0; i < count; ++i)
        if (std::string(layers[i].layerName) == "VK_LAYER_KHRONOS_validation")
            return layers[i].specVersion >= kMinSyncValidationLayer;
    return false;
}

const bool syncValidationOn = []()
{
    const bool on = syncValidationWanted();
    if (on)
    {
        setEnv("VK_LAYER_VALIDATE_SYNC", "1");
        setEnv("VK_LAYER_SYNCVAL_SHADER_ACCESSES_HEURISTIC", "1");
    }
    std::cout << "test_vulkan_sync: synchronization validation " << (on ? "on" : "off") << std::endl;
    return on;
}();

GrassMaskData fullMask(const TerrainLayout& layout)
{
    GrassMaskData m;
    m.size    = layout.grassMaskSize();
    m.density = std::make_shared<std::vector<uint8_t>>(size_t(m.size) * size_t(m.size), uint8_t(255));
    return m;
}

class SyncValidation : public RenderSceneTest
{
};
}  // namespace

// Regression (S36a2 review, blocker 1): the grass samples the cascades and the horizon maps in its vertex shader, but
// the barriers after the cascades' depth pass and the horizon map's rows named only the fragment and compute stages
// (a read-after-write hazard of the vertex stage: the grass read a layer still being rasterized, in the depth
// attachment layout). A frame with terrain, grass, the cascades and the horizon map in clay (the studio sun) and in the
// physical path (the atmosphere LUTs, also sampled per vertex), the sun moving every frame (the horizon map recomputes,
// the sky-view LUT is rewritten). Small images (256 texel cascades, a 128 texel horizon map): test speed
TEST_F(SyncValidation, ShadowedGrassFrame)
{
    RecordProperty("sync_validation", syncValidationOn ? "on" : "off");
    const TerrainLayout layout = testLayout(16, 2);
    scene->setFoliageTexture(1);
    const Vec3 cam(300.5, 3, 300.5);
    // A small terrain range and grass radius (lavapipe's vertex stage)
    fillTestTerrain(*scene, layout, slopeRidgeHook, 1, std::nullopt, cam, false, 0, 100);
    PrototypeSettings ps = scene->prototypeSettings();
    ps.grassRadius       = 10;
    scene->setPrototypeSettings(ps);
    for (int32_t z = 0; z < 3; ++z)
        for (int32_t x = 0; x < 3; ++x)
        {
            ChunkId c;
            c.x = x;
            c.z = z;
            scene->setGrassMask(c, fullMask(layout));
        }
    LightingParameters p;
    p.cascadeResolution = 256;
    p.horizonResolution = 128;
    p.shadowDistance    = 100;
    scene->setLightingParameters(p);
    const Vec3 forward(0.6, -1.2, 0.7);
    SetCamera(0, cam, viewRotation(lookAt(cam.cast<float>(), (cam + forward).cast<float>(), vec3(0, 1, 0))), 1.0);
    renderUntilUploaded(*scene);
    for (int frame = 0; frame < 2; ++frame)
    {
        const double el = (25 + 2 * frame) * kPi / 180;
        StudioEnvironment s;
        s.sunDir = vec3(float(std::cos(el)), float(std::sin(el)), 0.2f).normalized();
        scene->setStudioEnvironment(s);
        RenderFrame();
    }
    scene->waitIdle();
    EXPECT_TRUE(scene->lightingStatistics().shadows);
    EXPECT_GT(scene->grassPass().stats().blocks, 0);
    EXPECT_GT(scene->lightingStatistics().horizon.completed, 0u);

    // The physical path: an environment, ACES
    scene->setStudioEnvironment(std::nullopt);
    scene->setToneMap(ToneMap::Aces);
    for (int frame = 0; frame < 2; ++frame)
    {
        const float el = float((30 + 2 * frame) * kPi / 180);
        EnvironmentState e;
        e.sunDir = vec3(std::cos(el), std::sin(el), 0.2f).normalized();
        scene->setEnvironment(e);
        RenderFrame();
    }
    scene->waitIdle();
    EXPECT_TRUE(scene->lightingStatistics().shadows);
}

// Round 2 (finding 3): the frames above wait for every frame (a screenshot each), so two frames are never in flight
// and a hazard against the previous frame (a write-after-read, a descriptor set updated while a pending command buffer
// has it bound) cannot occur. Here 8 frames are recorded back to back without a wait (2 frames in flight), each
// changing something that rewrites resources of the frame before: the fovy (the projection), the cloud seed in
// consecutive frames (finding 11: the noise and the coverage texture regenerated while the previous frame reads them),
// the environment map's resolution, the horizon map's resolution and the cascade count; a capture and a probe on the
// way. Small images (test speed): 32/64 texel environment maps, 256 texel cascades, 128/256 texel horizon maps
TEST_F(SyncValidation, FramesInFlightWithChanges)
{
    RecordProperty("sync_validation", syncValidationOn ? "on" : "off");
    const TerrainLayout layout = testLayout(16, 2);
    const Vec3 cam(300.5, 12, 300.5);
    fillTestTerrain(*scene, layout, flatHook, 1, std::nullopt, cam, false, 0, 60);
    LightingParameters p;
    p.cascadeResolution     = 256;
    p.horizonResolution     = 128;
    p.environmentResolution = 32;
    p.shadowDistance        = 60;
    scene->setLightingParameters(p);
    EnvironmentState e;
    e.cloudiness = 0.5f;
    scene->setEnvironment(e);
    scene->setToneMap(ToneMap::Aces);
    const Quat rot = viewRotation(lookAt(cam.cast<float>(), vec3(300.5f, 0, 299.5f), vec3(0, 1, 0)));
    SetCamera(0, cam, rot, 1.0);
    renderUntilUploaded(*scene);
    bool captured = false, probed = false;
    for (int frame = 0; frame < 8; ++frame)
    {
        // Every frame: another projection
        SetCamera(0, cam, rot, 1.0 + 0.02 * frame);
        // Consecutive frames: another cloud seed
        if (frame == 1 || frame == 2)
        {
            e.clouds.seed = uint32_t(10 + frame);
            scene->setEnvironment(e);
        }
        if (frame == 3)
        {
            p.environmentResolution = 64;
            p.horizonResolution     = 256;
            scene->setLightingParameters(p);
        }
        if (frame == 4)
        {
            p.cascadeCount = 2;
            scene->setLightingParameters(p);
            scene->lightingDebugImages().request(LightingImage::Luminance, PreviewMapping(),
                                                 [&](const LightingDebug::Capture&) { captured = true; });
        }
        if (frame == 5) scene->requestProbe(ivec2(width / 2, height / 2), [&](const LightingProbe&) { probed = true; });
        scene->renderFrame([](vk::CommandBuffer) {}, [](vk::CommandBuffer) {}, nullptr);
    }
    // (the probe's sequence takes frames; render until it answered)
    for (int frame = 0; frame < 32 && !probed; ++frame)
        scene->renderFrame([](vk::CommandBuffer) {}, [](vk::CommandBuffer) {}, nullptr);
    scene->waitIdle();
    EXPECT_TRUE(captured);
    EXPECT_TRUE(probed);
    EXPECT_GE(scene->cloudLayer().generations(), 2u);
}
