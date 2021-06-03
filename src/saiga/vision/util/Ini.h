/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/String.h"
#include "saiga/core/util/assert.h"
#include "saiga/core/util/tostring.h"
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



template <typename _Scalar>
std::string toIniString(const IntrinsicsPinhole<_Scalar>& I)
{
    return toIniString(I.coeffs());
}

template <typename _Scalar>
std::string toIniString(const StereoCamera4Base<_Scalar>& I)
{
    return toIniString(I.coeffs());
}

template <typename _Scalar>
void fromIniString(const std::string& str, IntrinsicsPinhole<_Scalar>& I)
{
    typename IntrinsicsPinhole<_Scalar>::Vec5 v;
    fromIniString(str, v);
    I.coeffs(v);
}

template <typename _Scalar>
void fromIniString(const std::string& str, StereoCamera4Base<_Scalar>& I)
{
    typename StereoCamera4Base<_Scalar>::Vec5 v;
    fromIniString(str, v);
    I.coeffs(v);
}


}  // namespace Saiga
