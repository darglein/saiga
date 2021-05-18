/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/math/math.h"

namespace Saiga
{
class SAIGA_CORE_API Cone
{
   public:
    vec3 position;
    vec3 direction = vec3(0, 1, 0);
    float radius   = 1.0f;
    float height   = 1.0f;


    Cone(void) {}

    Cone(const vec3& position, const vec3& direction, float radius, float height)
        : position(position), direction(direction), radius(radius), height(height)
    {
    }
    ~Cone(void) {}
};

}  // namespace Saiga
