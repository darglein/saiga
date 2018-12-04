/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/util/assert.h"
#include "saiga/util/tostring.h"
#include "saiga/vision/VisionTypes.h"

namespace Saiga
{
template <typename _Scalar>
std::string toIniString(const Sophus::SE3<_Scalar>& M)
{
    const _Scalar* data = M.data();
    std::string str;
    for (int i = 0; i < M.num_parameters; ++i) str += Saiga::to_string(data[i], 12) + " ";
    return str;
}

template <typename _Scalar>
void fromIniString(const std::string& str, Sophus::SE3<_Scalar>& M)
{
    _Scalar* data = M.data();
    auto arr      = Saiga::split(str, ' ');
    SAIGA_ASSERT((int)arr.size() == M.num_parameters);

    for (int i = 0; i < M.num_parameters; ++i) data[i] = FromStringConverter<_Scalar>::convert(arr[i]);
}



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


template <typename _Scalar>
std::string toIniString(const Intrinsics4Base<_Scalar>& I)
{
    return toIniString(I.coeffs());
}

template <typename _Scalar>
void fromIniString(const std::string& str, Intrinsics4Base<_Scalar>& I)
{
    typename Intrinsics4Base<_Scalar>::Vec4 v;
    fromIniString(str, v);
    I.coeffs(v);
}


}  // namespace Saiga
