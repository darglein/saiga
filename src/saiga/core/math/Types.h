/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <Eigen/Core>


namespace Saiga
{
// ===== Double Precision (Capital Letter) ======
using Vec7 = Eigen::Matrix<double, 7, 1>;
using Vec6 = Eigen::Matrix<double, 6, 1>;
using Vec4 = Eigen::Matrix<double, 4, 1>;
using Vec3 = Eigen::Matrix<double, 3, 1>;
using Vec2 = Eigen::Matrix<double, 2, 1>;

using Mat4 = Eigen::Matrix4d;
using Mat3 = Eigen::Matrix3d;


// ===== Single Precision  ======

using Vec4f = Eigen::Matrix<float, 4, 1>;
using Vec3f = Eigen::Matrix<float, 3, 1>;
using Vec2f = Eigen::Matrix<float, 2, 1>;

using vec4 = Eigen::Vector4f;
using vec3 = Eigen::Vector3f;
using vec2 = Eigen::Vector2f;

using mat4 = Eigen::Matrix4f;
using mat3 = Eigen::Matrix3f;

// ===== Non-floating point types. Used for example in image processing  ======

using uvec3 = Eigen::Matrix<unsigned int, 3, 1>;

using ivec2 = Eigen::Matrix<int, 2, 1>;
using ivec3 = Eigen::Matrix<int, 3, 1>;
using ivec4 = Eigen::Matrix<int, 4, 1>;

using cvec2 = Eigen::Matrix<char, 2, 1>;
using cvec3 = Eigen::Matrix<char, 3, 1>;
using cvec4 = Eigen::Matrix<char, 4, 1>;

using ucvec2 = Eigen::Matrix<unsigned char, 2, 1>;
using ucvec3 = Eigen::Matrix<unsigned char, 3, 1>;
using ucvec4 = Eigen::Matrix<unsigned char, 4, 1>;

using usvec2 = Eigen::Matrix<unsigned short, 2, 1>;
using usvec3 = Eigen::Matrix<unsigned short, 3, 1>;
using usvec4 = Eigen::Matrix<unsigned short, 4, 1>;

}  // namespace Saiga
