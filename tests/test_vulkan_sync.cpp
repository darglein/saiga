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
void setEnv(const char* name, const char* value, bool overwrite)
{
#if defined(_WIN32)
    if (overwrite || !std::getenv(name)) _putenv_s(name, value);
#else
    setenv(name, value, overwrite ? 1 : 0);
#endif
}
// The oldest layer used with sync validation: 1.4.328 knows the shader-access heuristic
// (syncval_shader_accesses_heuristic in its sync_settings.h) and is newer than the 1.3.275 of the false positive; the
// local layer (1.4.341) is clean
constexpr uint32_t kMinSyncValidationLayer = VK_MAKE_API_VERSION(0, 1, 4, 328);

// The spec version of the installed Khronos validation layer (0: none)
uint32_t validationLayerVersion()
{
    uint32_t count = 0;
    if (vkEnumerateInstanceLayerProperties(&count, nullptr) != VK_SUCCESS) return 0;
    std::vector<VkLayerProperties> layers(count);
    if (vkEnumerateInstanceLayerProperties(&count, layers.data()) != VK_SUCCESS) return 0;
    for (uint32_t i = 0; i < count; ++i)
        if (std::string(layers[i].layerName) == "VK_LAYER_KHRONOS_validation") return layers[i].specVersion;
    return 0;
}

// Whether the instance offers VK_EXT_debug_utils (Instance::create enables the validation layer only with it)
bool hasDebugUtils()
{
    uint32_t count = 0;
    if (vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr) != VK_SUCCESS) return false;
    std::vector<VkExtensionProperties> extensions(count);
    if (vkEnumerateInstanceExtensionProperties(nullptr, &count, extensions.data()) != VK_SUCCESS) return false;
    for (uint32_t i = 0; i < count; ++i)
        if (std::string(extensions[i].extensionName) == VK_EXT_DEBUG_UTILS_EXTENSION_NAME) return true;
    return false;
}

// Whether the frames are checked by sync validation: requested by SAIGA_TEST_SYNC_VALIDATION=1/0 (it wins over
// VK_LAYER_VALIDATE_SYNC), else by VK_LAYER_VALIDATE_SYNC, else on with a layer of at least kMinSyncValidationLayer;
// VK_LAYER_VALIDATE_SYNC is set to the request, and the frames are checked only with a validation layer installed and
// VK_EXT_debug_utils (as Instance::create; CodeRabbit #85 round 2)
const bool syncValidationOn = []()
{
    const uint32_t layer = validationLayerVersion();
    bool requested       = layer >= kMinSyncValidationLayer;
    const char* source   = "the layer version";
    if (const char* forced = std::getenv("SAIGA_TEST_SYNC_VALIDATION"))
    {
        requested = std::string(forced) != "0";
        source    = "SAIGA_TEST_SYNC_VALIDATION";
    }
    else if (const char* env = std::getenv("VK_LAYER_VALIDATE_SYNC"))
    {
        requested = std::string(env) != "0" && std::string(env) != "false";
        source    = "VK_LAYER_VALIDATE_SYNC";
    }
    setEnv("VK_LAYER_VALIDATE_SYNC", requested ? "1" : "0", true);
    if (requested) setEnv("VK_LAYER_SYNCVAL_SHADER_ACCESSES_HEURISTIC", "1", false);
    const bool debugUtils = hasDebugUtils();
    const bool on         = requested && layer > 0 && debugUtils;
    std::cout << "test_vulkan_sync: synchronization validation " << (on ? "on" : "off") << " (requested "
              << (requested ? "on" : "off") << " by " << source << "; validation layer "
              << (layer > 0 ? std::to_string(VK_API_VERSION_MAJOR(layer)) + "." +
                                  std::to_string(VK_API_VERSION_MINOR(layer)) + "." +
                                  std::to_string(VK_API_VERSION_PATCH(layer))
                            : std::string("not installed"))
              << (debugUtils ? "" : "; no VK_EXT_debug_utils") << ")" << std::endl;
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

namespace
{
MeshLods syncBox(const vec3& half)
{
    MeshLod lod;
    lod.endDistance = 2000;
    for (int f = 0; f < 6; ++f)
    {
        const int axis      = f / 2;
        const float s       = (f % 2) ? -1.0f : 1.0f;
        vec3 n              = vec3::Zero();
        n(axis)             = s;
        const vec3 u        = axis == 1 ? vec3(1, 0, 0) : vec3(0, 1, 0);
        const vec3 v        = n.cross(u);
        const uint32_t base = uint32_t(lod.vertices.size());
        for (const vec2 c : {vec2(-1, -1), vec2(1, -1), vec2(1, 1), vec2(-1, 1)})
        {
            MeshVertex m;
            m.position = (n + c(0) * u + c(1) * v).cwiseProduct(half);
            m.normal   = packNormalOct(n);
            m.color    = packColor(ucvec4(200, 200, 200, 255));
            lod.vertices.push_back(m);
        }
        for (uint32_t i : {0u, 1u, 2u, 0u, 2u, 3u}) lod.indices.push_back(base + i);
    }
    MeshLods m;
    m.lods.push_back(lod);
    m.sphereRadius = half.norm();
    return m;
}
}  // namespace

// S36b round 1 (findings 3, 14; the review's request for a hazard check of the local lights): a frame with shadowed
// point and spot lights over terrain (the atlas's terrain path), a moving dynamic instance (the dynamic layer), the
// clusters; the atlas is created inside the frame's command buffer (finding 14: no blocking submit) and recreated
// (a resolution change), the cluster grid grows between frames (finding 3: the buffers before the lighting set binds
// them), a light changes its kind (finding 1)
TEST_F(SyncValidation, LocalLightsAndAtlasFrame)
{
    ASSERT_TRUE(syncValidationOn);
    const TerrainLayout layout = testLayout(16, 2);
    const Vec3 cam(300.5, 12, 300.5);
    fillTestTerrain(*scene, layout, flatHook, 1, std::nullopt, cam, false, 0, 60);
    MaterialData m;
    scene->addMaterial(MaterialId(900), "sync_box", m);
    scene->addMesh(MeshId(900), "sync_box", syncBox(vec3(0.5f, 0.5f, 0.5f)), MaterialId(900));
    const double ground = 0;  // flatHook
    AddSet({[&]()
            {
                StaticInstance s;
                s.setPosition(vec3(301.5f, float(ground) + 0.5f, 300.5f));
                s.mesh = MeshId(900);
                return s;
            }()});
    LightingParameters p;
    p.cascadeResolution = 256;
    p.horizonResolution = 128;
    p.sunShadows        = false;
    p.horizonShadows    = false;
    p.localLights       = true;
    p.atlasResolution   = 512;
    p.atlasTile         = 128;
    p.facesPerFrame     = 11;
    scene->setLightingParameters(p);
    EnvironmentState e;
    e.sunIlluminance  = 0;
    e.moonIlluminance = 0;
    scene->setEnvironment(e);
    scene->setToneMap(ToneMap::Aces);
    LightData point;
    point.kind          = LightKind::Point;
    point.position      = vec3(300.5f, float(ground) + 3, 300.5f);
    point.lumen         = 2000;
    point.range         = 10;
    point.castShadows   = true;
    const LightHandle a = scene->addLight(ChunkId(), point);
    LightData spot      = point;
    spot.kind           = LightKind::Spot;
    spot.position       = vec3(303.5f, float(ground) + 4, 300.5f);
    spot.direction      = vec3(0, -1, 0);
    spot.cosInner       = 0.9f;
    spot.cosOuter       = 0.8f;
    scene->addLight(ChunkId(), spot);
    const InstanceHandle d = scene->addDynamic(MeshId(900));
    SetCamera(0, cam, viewRotation(lookAt(cam.cast<float>(), vec3(300.5f, float(ground), 299.5f), vec3(0, 1, 0))), 1.0);
    renderUntilUploaded(*scene);
    for (int frame = 0; frame < 4; ++frame)
    {
        DynamicPose prev, curr;
        prev.position = Vec3(299.5 + 0.1 * frame, ground + 1.5, 300.5);
        curr.position = prev.position + Vec3(0.1, 0, 0);
        scene->setDynamic(d, ChunkId(), prev, curr);
        if (frame == 1)
        {
            p.clusterZ            = 48;
            p.maxLightsPerCluster = 256;
            p.atlasResolution     = 1024;
            scene->setLightingParameters(p);
        }
        if (frame == 2) scene->setLight(a, ChunkId(), spot);
        RenderFrame();
    }
    scene->waitIdle();
    EXPECT_TRUE(scene->lightingStatistics().atlas.on);
    EXPECT_EQ(scene->lightingStatistics().atlas.resolution, 1024);
    EXPECT_GT(scene->lightingStatistics().lights.visible, 0u);
    scene->removeDynamic(d);
}

// Round 2 (finding 3): the frames above wait for every frame (a screenshot each), so two frames are never in flight
// and a hazard against the previous frame (a write-after-read, a descriptor set updated while a pending command buffer
// has it bound) cannot occur. Here 8 frames are recorded back to back without a wait (2 frames in flight), each
// changing something that rewrites resources of the frame before: the fovy (the projection), the cloud seed in
// consecutive frames (finding 11: the noise and the coverage texture regenerated while the previous frame reads them),
// the environment map's resolution, the horizon map's resolution and the cascade count; a capture and a probe after
// eight frames. Small images (test speed): 32/64 texel environment maps, 256 texel cascades, 128/256 texel horizon maps
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
        }
        scene->renderFrame([](vk::CommandBuffer) {}, [](vk::CommandBuffer) {}, nullptr);
    }
    // The blocking readbacks after the eight frames (CodeRabbit #85 round 2: a capture waits for its frame and the
    // probe holds the camera, so the frames after them neither overlapped nor changed the projection): a capture and
    // a probe on frames in flight; the probe's sequence takes frames, render until it answered
    scene->lightingDebugImages().request(LightingImage::Luminance, PreviewMapping(),
                                         [&](const LightingDebug::Capture&) { captured = true; });
    scene->renderFrame([](vk::CommandBuffer) {}, [](vk::CommandBuffer) {}, nullptr);
    scene->requestProbe(ivec2(width / 2, height / 2), [&](const LightingProbe&) { probed = true; });
    for (int frame = 0; frame < 32 && !probed; ++frame)
        scene->renderFrame([](vk::CommandBuffer) {}, [](vk::CommandBuffer) {}, nullptr);
    scene->waitIdle();
    EXPECT_TRUE(captured);
    EXPECT_TRUE(probed);
    EXPECT_GE(scene->cloudLayer().generations(), 2u);
}
