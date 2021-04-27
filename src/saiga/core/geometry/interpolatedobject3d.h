/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "object3d.h"

namespace Saiga
{
class SAIGA_CORE_API InterpolatedObject3D : public Object3D
{
   public:
    mat4 interpolatedmodel = mat4::Identity();

    quat oldrot, interpolatedrot;
    vec4 oldscale = make_vec4(1), interpolatedscale = make_vec4(1);
    vec4 oldposition = make_vec4(0), interpolatedposition = make_vec4(0);


    void interpolate(float alpha);
    void update();
};

}  // namespace Saiga
