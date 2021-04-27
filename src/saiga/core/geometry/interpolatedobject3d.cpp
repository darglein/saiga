/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "interpolatedobject3d.h"

#include "internal/noGraphicsAPI.h"

namespace Saiga
{
void InterpolatedObject3D::interpolate(float alpha)
{
    //    interpolatedrot = mix(oldrot,rot,alpha);

    interpolatedrot = slerp(oldrot, rot, alpha);
    interpolatedrot = (interpolatedrot).normalized();


    interpolatedscale    = mix(oldscale, scale, alpha);
    interpolatedposition = mix(oldposition, position, alpha);

    interpolatedmodel = createTRSmatrix(interpolatedposition, interpolatedrot, interpolatedscale);
    //    interpolatedmodel = mat4_cast(interpolatedrot)*scale(mat4::Identity(),interpolatedscale);
    //    interpolatedmodel[3] = vec4(interpolatedposition,1);
}

void InterpolatedObject3D::update()
{
    oldrot      = rot;
    oldscale    = scale;
    oldposition = position;
}

}  // namespace Saiga
