/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/math.h"


namespace Saiga
{
/**
 * Procedural Skybox Base class.
 *
 * The implementations for OpenGL and Vulkan are provided in
 * the corresponding modules.
 */
class SAIGA_CORE_API ProceduralSkyboxBase
{
   public:
    float horizonHeight = 0;
    float distance      = 1024;
    float sunIntensity  = 1;
    float sunSize       = 1;
    vec3 sunDir         = vec3(0, -1, 0);
    void imgui();
};

}  // namespace Saiga
