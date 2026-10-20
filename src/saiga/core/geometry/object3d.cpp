/**
 * Copyright (c) 2021 Darius Rückert
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



void Object3D::turn(float angleX, float angleY, const vec3& up)
{
    rotateGlobal(up, angleX);
    quat rottmp   = rot;
    float old_dot = make_vec3(getUpVector()).dot(up);
    rotateLocal(vec3(1, 0, 0), angleY);
    // Don't turn upside down. An object that is already upside down may still pitch back towards upright.
    float new_dot = make_vec3(getUpVector()).dot(up);
    if (new_dot < 0 && new_dot < old_dot)
    {
        rot = rottmp;
    }
}

void Object3D::turnLocal(float angleX, float angleY)
{
    rotateLocal(vec3(0, 1, 0), angleX);
    rotateLocal(vec3(1, 0, 0), angleY);
}

void Object3D::rotateAroundPoint(const vec3& point, const vec3& axis, float angle)
{
    // The position is rotated around the global axis, so the orientation must be too.
    rotateGlobal(axis, angle);

    translateGlobal(vec3(-point));
    quat qrot = angleAxis(radians(angle), axis);
    //    position = vec3(qrot*vec4(position,1));
    position = make_vec4(qrot * make_vec3(position), 1);
    translateGlobal(point);
}

quat Object3D::getSimpleDirectionQuat(const vec3& dir)
{
    // Orthonormal basis looking along -z. The up vector is only (0,1,0) if dir is horizontal.
    vec3 z = -dir.normalized();
    vec3 x = cross(dir, vec3(0, 1, 0));
    if (x.squaredNorm() < 1e-12f * dir.squaredNorm())
    {
        // dir is vertical: any horizontal right vector is valid.
        x = cross(dir, vec3(0, 0, -1));
    }
    x      = normalize(x);
    vec3 y = cross(z, x);
    mat3 rotmat;
    rotmat.col(0) = x;
    rotmat.col(1) = y;
    rotmat.col(2) = z;
    return (make_quat(rotmat)).normalized();
}

Object3D Object3D::interpolate(const Object3D& a, const Object3D& b, float alpha)
{
    Object3D res;
    res.rot      = (slerp(a.rot, b.rot, alpha)).normalized();
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

void Saiga::Object3D::setModelMatrix(const mat4& _model)
{
    // this is the inverse of createTRSmatrix
    model    = _model;
    position = model.col(3);
    mat3 R   = make_mat3(model);
    scale[0] = length(vec3(R.col(0)));
    scale[1] = length(vec3(R.col(1)));
    scale[2] = length(vec3(R.col(2)));
    R.col(0) = R.col(0) / scale[0];
    R.col(1) = R.col(1) / scale[1];
    R.col(2) = R.col(2) / scale[2];
    rot      = quat_cast(R);
}

}  // namespace Saiga
