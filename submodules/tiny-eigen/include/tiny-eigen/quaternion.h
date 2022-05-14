/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "config.h"
#include "matrix.h"
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
    using Scalar     = _Scalar;
    using SameObject = Quaternion<_Scalar>;

    static SameObject Identity() { return SameObject(1, 0, 0, 0); }

    static SameObject FromTwoVectors(Matrix<Scalar, 3, 1> v1, Matrix<Scalar, 3, 1> v2) { return {}; }
    static SameObject FromAngleAxis(Scalar angle, Matrix<Scalar, 3, 1> v1) { return {}; }

    Quaternion() {}
    Quaternion(_Scalar w, _Scalar x, _Scalar y, _Scalar z)
    {
        _data[0] = x;
        _data[1] = y;
        _data[2] = z;
        _data[3] = w;
    }

    Quaternion(const Matrix<_Scalar, 3, 3>& m) {}

    Matrix<_Scalar, 3, 3> matrix() const
    {
        Matrix<_Scalar, 3, 3> result;
        return result;
    }

    SameObject inverse() const { return SameObject(); }
    SameObject normalized() const { return SameObject(); }

    SameObject slerp(Scalar alpha, SameObject other) const { return SameObject(); }

    Scalar& x() { return _data[0]; }
    Scalar& y() { return _data[1]; }
    Scalar& z() { return _data[2]; }
    Scalar& w() { return _data[3]; }
    const Scalar& x() const { return _data[0]; }
    const Scalar& y() const { return _data[1]; }
    const Scalar& z() const { return _data[2]; }
    const Scalar& w() const { return _data[3]; }

   private:
    _Scalar _data[4];
};

template <typename _Scalar, int _Options>
Matrix<_Scalar, 3, 1, _Options> operator*(const Quaternion<_Scalar> quat, const Matrix<_Scalar, 3, 1, _Options>& v)
{
    Matrix<_Scalar, 3, 1, _Options> result;
    throw 1;
    return result;
}

template <typename _Scalar>
 Quaternion<_Scalar> operator*(const Quaternion<_Scalar> quat, const Quaternion<_Scalar>& v)
{
    Quaternion<_Scalar>  result;
    throw 1;
    return result;
}

using Quaternionf = Quaternion<float>;
using Quaterniond = Quaternion<double>;

}  // namespace Eigen