/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "object3d.h"

#include "saiga/core/math/String.h"

#include "internal/noGraphicsAPI.h"

namespace Saiga
{
void Object3D::setSimpleDirection(const vec3& dir)
{
    this->rot = getSimpleDirectionQuat(dir);
}



void Object3D::turn(float angleX, float angleY)
{
    rotateGlobal(vec3(0, 1, 0), angleX);
    mat4 modeltmp = model;
    rotateLocal(vec3(1, 0, 0), angleY);
    if (col(model, 1)[1] < 0)
    {
        model = modeltmp;
    }
}

void Object3D::turnLocal(float angleX, float angleY)
{
    rotateLocal(vec3(0, 1, 0), angleX);
    rotateLocal(vec3(1, 0, 0), angleY);
}

void Object3D::rotateAroundPoint(const vec3& point, const vec3& axis, float angle)
{
    rotateLocal(axis, angle);

    translateGlobal(vec3(-point));
    quat qrot = angleAxis(radians(angle), axis);
    //    position = vec3(qrot*vec4(position,1));
    position = make_vec4(qrot * make_vec3(position), 1);
    translateGlobal(point);
}

quat Object3D::getSimpleDirectionQuat(const vec3& dir)
{
    mat4 rotmat   = identityMat4();
    rotmat.col(0) = make_vec4(normalize(cross(dir, vec3(0, 1, 0))), 0);
    rotmat.col(1) = make_vec4(0, 1, 0, 0);
    rotmat.col(2) = make_vec4(-dir, 0);

    return normalize(make_quat(rotmat));
}

Object3D Object3D::interpolate(const Object3D& a, const Object3D& b, float alpha)
{
    Object3D res;
    res.rot      = normalize(slerp(a.rot, b.rot, alpha));
    res.scale    = mix(a.scale, b.scale, alpha);
    res.position = mix(a.position, b.position, alpha);
    return res;
}

std::ostream& operator<<(std::ostream& os, const Saiga::Object3D& o)
{
    os << "Object3D (P/R/S): " << o.position.transpose() << " " << o.rot.coeffs().transpose() << " "
       << o.scale.transpose();
    return os;
}

}  // namespace Saiga
