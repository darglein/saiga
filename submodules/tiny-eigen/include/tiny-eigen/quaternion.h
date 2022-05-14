/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "config.h"

namespace Eigen
{

template <class Derived>
class QuaternionBase
{
   public:
};

template <typename _Scalar>
class Quaternion : public QuaternionBase<Quaternion<_Scalar>>
{
   public:
};

using Quaternionf = Quaternion<float>;
using Quaterniond = Quaternion<double>;

}  // namespace Eigen