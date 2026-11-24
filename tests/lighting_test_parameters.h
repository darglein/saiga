/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vulkan/scene/LightingTypes.h"

namespace Saiga
{
// The lighting of the other tests (S36a2 review, test speed): the cascades at 256 texels and the horizon map at 128
// texels over 4 km (32 m texels as at the defaults), S36b the shadow atlas at 1024 / 128, where the parameters have
// the defaults' sizes (the goldens and the tests of the sizes keep theirs); the scene fixtures
// (render_scene_test_context.h, test_vulkan_scene.cpp) and the apps of the tests use it. The default 4 x 2048^2
// cascades and the 1024^2 horizon map cost lavapipe most of a test's frame time
inline Vulkan::LightingParameters testLightingParameters(Vulkan::LightingParameters l = {})
{
    const Vulkan::LightingParameters d;
    if (l.cascadeResolution == d.cascadeResolution) l.cascadeResolution = 256;
    if (l.horizonResolution == d.horizonResolution && l.horizonExtent == d.horizonExtent)
    {
        l.horizonResolution = 128;
        l.horizonExtent     = 4096;
    }
    // S36b round 1, finding 16: the shadow atlas at 1024 texels in tiles of 128 (64 tiles as at the defaults' 4096 /
    // 512: the same budget of 10 point lights; a 4096^2 x 2 D16 image costs lavapipe its clear and the faces'
    // rasterizer work)
    if (l.atlasResolution == d.atlasResolution && l.atlasTile == d.atlasTile)
    {
        l.atlasResolution = 1024;
        l.atlasTile       = 128;
    }
    return l;
}

}  // namespace Saiga
