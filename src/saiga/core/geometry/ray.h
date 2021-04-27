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
class SAIGA_CORE_API Ray
{
   public:
    vec3 direction;
    vec3 origin;

    HD Ray() : direction(make_vec3(0)), origin(make_vec3(0)) {}
    HD Ray(const vec3& dir, const vec3& ori) : direction(dir), origin(ori) {}
    HD vec3 positionOnRay(float t) const { return origin + t * direction; }

    SAIGA_CORE_API friend std::ostream& operator<<(std::ostream& os, const Ray& dt);
};

}  // namespace Saiga
