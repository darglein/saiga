/**
 * Copyright (c) 2021 Darius Rückert
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
using Vec2 = Vector<double, 2>;
using Vec3 = Vector<double, 3>;
using Vec4 = Vector<double, 4>;
using Vec5 = Vector<double, 5>;
using Vec6 = Vector<double, 6>;
using Vec7 = Vector<double, 7>;
using Vec8 = Vector<double, 8>;
using Vec9 = Vector<double, 9>;

using Mat2 = Matrix<double, 2, 2>;
using Mat3 = Matrix<double, 3, 3>;
using Mat4 = Matrix<double, 4, 4>;


// ===== Single Precision  ======

using vec2 = Vector<float, 2>;
using vec3 = Vector<float, 3>;
using vec4 = Vector<float, 4>;
using vec5 = Vector<float, 5>;
using vec6 = Vector<float, 6>;
using vec7 = Vector<float, 7>;
using vec8 = Vector<float, 8>;
using vec9 = Vector<float, 9>;

using mat2 = Matrix<float, 2, 2>;
using mat3 = Matrix<float, 3, 3>;
using mat4 = Matrix<float, 4, 4>;

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
