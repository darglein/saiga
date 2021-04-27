/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/assert.h"
#include "saiga/core/util/tostring.h"

#include "Quaternion.h"
#include "Types.h"


namespace Saiga
{
template <typename _Scalar, int _Rows, int _Cols>
std::string toIniString(const Eigen::Matrix<_Scalar, _Rows, _Cols>& M)
{
    std::string str;
    // Add entries to string, separated with ' ' in row major order.
    for (int i = 0; i < M.rows(); ++i)
        for (int j = 0; j < M.cols(); ++j) str += Saiga::to_string(M(i, j), 12) + " ";

    return str;
}


template <typename _Scalar, int _Rows, int _Cols>
void fromIniString(const std::string& str, Eigen::Matrix<_Scalar, _Rows, _Cols>& M)
{
    auto arr = Saiga::split(str, ' ');
    SAIGA_ASSERT((int)arr.size() == M.rows() * M.cols());

    for (int i = 0; i < M.rows(); ++i)
        for (int j = 0; j < M.cols(); ++j) M(i, j) = FromStringConverter<_Scalar>::convert(arr[i * M.cols() + j]);
}


template <typename T, int N>
inline std::istream& operator>>(std::istream& is, Saiga::Vector<T, N>& v)
{
    //    is >> v(0) >> v(1);
    for (int i = 0; i < N; ++i)
    {
        is >> v(i);
    }
    return is;
}

}  // namespace Saiga
