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
// texels over 4 km (32 m texels as at the defaults) where the parameters have the defaults' sizes (the goldens and the
// tests of the sizes keep theirs); the scene fixtures (render_scene_test_context.h, test_vulkan_scene.cpp) and the apps
// of the tests use it. The default 4 x 2048^2 cascades and the 1024^2 horizon map cost lavapipe most of a test's frame
// time
inline Vulkan::LightingParameters testLightingParameters(Vulkan::LightingParameters l = {})
{
    const Vulkan::LightingParameters d;
    if (l.cascadeResolution == d.cascadeResolution) l.cascadeResolution = 256;
    if (l.horizonResolution == d.horizonResolution && l.horizonExtent == d.horizonExtent)
    {
        l.horizonResolution = 128;
        l.horizonExtent     = 4096;
    }
    return l;
}

}  // namespace Saiga
