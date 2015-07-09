#include "libhello/rendering/interpolatedobject3d.h"






void InterpolatedObject3D::interpolate(float alpha)
{
//    interpolatedrot = glm::mix(oldrot,rot,alpha);

    interpolatedrot = glm::slerp(oldrot,rot,alpha);
    interpolatedrot = glm::normalize(interpolatedrot);


    interpolatedscale = glm::mix(oldscale,scale,alpha);
    interpolatedposition = glm::mix(oldposition,position,alpha);

    interpolatedmodel = glm::mat4_cast(interpolatedrot)*glm::scale(mat4(),interpolatedscale);
    interpolatedmodel[3] = vec4(interpolatedposition,1);

}

void InterpolatedObject3D::update()
{
    oldrot = rot;
    oldscale = scale;
    oldposition = position;
}
