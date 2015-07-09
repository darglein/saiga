#pragma once

#include "libhello/rendering/object3d.h"

class SAIGA_GLOBAL InterpolatedObject3D : public Object3D{
public:
    mat4 interpolatedmodel;

    glm::quat oldrot, interpolatedrot;
    vec3 oldscale = vec3(1), interpolatedscale = vec3(1);
    vec3 oldposition = vec3(0), interpolatedposition = vec3(0);


    void interpolate(float alpha);
    void update();

};


