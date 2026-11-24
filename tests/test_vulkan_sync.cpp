/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// RenderScene's frame under the validation layer's synchronization validation (S36a2 review, blocker 1): the process
// turns it on before the first Vulkan instance (VK_LAYER_VALIDATE_SYNC, with the SPIR-V heuristic of the shader
// accesses, without which the layer does not see the descriptor reads of a draw; an environment value set by the user
// is kept). A hazard is a validation error, which aborts the test. Without the validation layer, or with a layer that
// does not know the heuristic (e.g. Ubuntu 24.04's 1.3.275), the test only renders. A scene per test
// (RenderSceneTest), skipped without a Vulkan 1.3 device.

#include "saiga/core/math/math.h"
#include "saiga/vulkan/scene/GrassPass.h"
#include "saiga/vulkan/scene/LightingTypes.h"
#include "saiga/vulkan/scene/PrototypeWorld.h"
#include "saiga/vulkan/scene/RenderScene.h"
#include "saiga/vulkan/world/WorldCommon.h"

#include "gtest/gtest.h"

#include <cmath>
#include <cstdlib>
#include <memory>
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
const bool syncValidationOn = []()
{
    setEnv("VK_LAYER_VALIDATE_SYNC", "1");
    setEnv("VK_LAYER_SYNCVAL_SHADER_ACCESSES_HEURISTIC", "1");
    return true;
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
    ASSERT_TRUE(syncValidationOn);
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
