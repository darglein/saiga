/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

#include <Eigen/Core>
#include <Eigen/Geometry>


using quat = Eigen::Quaternionf;

using vec4 = Eigen::Vector4f;
using vec3 = Eigen::Vector3f;
using vec2 = Eigen::Vector2f;

using mat4 = Eigen::Matrix4f;
using mat3 = Eigen::Matrix3f;


#define IDENTITY_QUATERNION quat::Identity()



SAIGA_GLOBAL inline mat4 createTRSmatrix(const vec4& t, const quat& r, const vec4& s)
{
    // Equivalent to:
    //    mat4 T = translate(mat4(1),vec3(t));
    //    mat4 R = mat4_cast(r);
    //    mat4 S = scale(mat4(1),vec3(s));
    //    return T * R * S;


    return mat4::Identity();
}
