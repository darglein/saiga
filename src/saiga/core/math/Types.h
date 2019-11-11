/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <Eigen/Core>


namespace Saiga
{
// All vector types are formed from this typedef.
// -> A vector is a Nx1 Eigen::Matrix.
template <typename Scalar, int Size>
using Vector = Eigen::Matrix<Scalar, Size, 1, Eigen::ColMajor>;

// All 2D fixed size matrices are formed from this typedef.
// They are all stored in column major order.
template <typename Scalar, int Rows, int Cols>
using Matrix = Eigen::Matrix<Scalar, Rows, Cols, Eigen::ColMajor>;

// ===== Double Precision (Capital Letter) ======
using Vec7 = Vector<double, 7>;
using Vec6 = Vector<double, 6>;
using Vec4 = Vector<double, 4>;
using Vec3 = Vector<double, 3>;
using Vec2 = Vector<double, 2>;

using Mat4 = Matrix<double, 4, 4>;
using Mat3 = Matrix<double, 3, 3>;


// ===== Single Precision  ======

using Vec4f = Vector<float, 4>;
using Vec3f = Vector<float, 3>;
using Vec2f = Vector<float, 2>;

using vec4 = Vec4f;
using vec3 = Vec3f;
using vec2 = Vec2f;

using mat4 = Matrix<float, 4, 4>;
using mat3 = Matrix<float, 3, 3>;

// ===== Non-floating point types. Used for example in image processing  ======

using uvec2 = Vector<unsigned int, 2>;
using uvec3 = Vector<unsigned int, 3>;
using uvec4 = Vector<unsigned int, 4>;

using ivec2 = Vector<int, 2>;
using ivec3 = Vector<int, 3>;
using ivec4 = Vector<int, 4>;

using cvec2 = Vector<char, 2>;
using cvec3 = Vector<char, 3>;
using cvec4 = Vector<char, 4>;

using ucvec2 = Vector<unsigned char, 2>;
using ucvec3 = Vector<unsigned char, 3>;
using ucvec4 = Vector<unsigned char, 4>;

using svec2 = Vector<short, 2>;
using svec3 = Vector<short, 3>;
using svec4 = Vector<short, 4>;

using usvec2 = Vector<unsigned short, 2>;
using usvec3 = Vector<unsigned short, 3>;
using usvec4 = Vector<unsigned short, 4>;

}  // namespace Saiga
