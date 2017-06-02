#include "saiga/geometry/obb.h"

void OBB::setOrientationScale(vec3 x, vec3 y, vec3 z)
{
    orientationScale[0] = x;
    orientationScale[1] = y;
    orientationScale[2] = z;
}

void OBB::fitToPoints(int axis, vec3 *points, int count)
{
    float xMin = 234235125, xMax = -34853690;

    vec3 dir = orientationScale[axis];
    dir = glm::normalize(dir);

    for(int i = 0 ; i < count ; ++i){
        float x = dot(dir,points[i]);
        xMin = glm::min(xMin,x); xMax = glm::max(xMax,x);
    }

    orientationScale[axis] = 0.5f * dir * (xMax - xMin);

    //translate center along axis
    float centerAxis = 0.5f * (xMax + xMin);
    float d = dot(dir,center);
    center += (centerAxis - d) * dir;
}

void OBB::normalize()
{
    orientationScale[0] = glm::normalize( orientationScale[0] );
    orientationScale[1] = glm::normalize( orientationScale[1] );
    orientationScale[2] = glm::normalize( orientationScale[2] );
}
