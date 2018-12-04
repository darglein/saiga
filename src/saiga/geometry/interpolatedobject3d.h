/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/geometry/object3d.h"

namespace Saiga
{
class SAIGA_GLOBAL InterpolatedObject3D : public Object3D
{
   public:
    mat4 interpolatedmodel = mat4(1);

    quat oldrot, interpolatedrot;
    vec4 oldscale = vec4(1), interpolatedscale = vec4(1);
    vec4 oldposition = vec4(0), interpolatedposition = vec4(0);


    void interpolate(float alpha);
    void update();
};

}  // namespace Saiga
