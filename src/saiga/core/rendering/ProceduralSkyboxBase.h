/**
 * Copyright (c) 2021 Darius Rückert
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
    float sunSize       = 0.1;
    vec3 sunDir         = vec3(0, -1, 0);
    vec3 sunColor       = vec3(220, 155, 45) / 255.0f;
    vec3 highSkyColor   = vec3(43, 99, 192) / 255.0f;
    vec3 lowSkyColor    = vec3(97, 161, 248) / 255.0f;

    void imgui();
};

}  // namespace Saiga
