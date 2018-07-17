/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "interpolatedobject3d.h"
#include "internal/noGraphicsAPI.h"

namespace Saiga {

void InterpolatedObject3D::interpolate(float alpha)
{
//    interpolatedrot = glm::mix(oldrot,rot,alpha);

    interpolatedrot = glm::slerp(oldrot,rot,alpha);
    interpolatedrot = glm::normalize(interpolatedrot);


    interpolatedscale = glm::mix(oldscale,scale,alpha);
    interpolatedposition = glm::mix(oldposition,position,alpha);

    interpolatedmodel = createTRSmatrix(interpolatedposition,interpolatedrot,interpolatedscale);
//    interpolatedmodel = mat4_cast(interpolatedrot)*glm::scale(mat4(1),interpolatedscale);
//    interpolatedmodel[3] = vec4(interpolatedposition,1);


}

void InterpolatedObject3D::update()
{
    oldrot = rot;
    oldscale = scale;
    oldposition = position;
}

}
