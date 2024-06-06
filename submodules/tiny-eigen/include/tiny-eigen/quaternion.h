/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "config.h"
#include "matrix.h"

#include <limits>

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

    HD static SameObject Identity() { return SameObject(1, 0, 0, 0); }

    HD static SameObject FromTwoVectors(Matrix<Scalar, 3, 1> a, Matrix<Scalar, 3, 1> b)
    {
        Matrix<Scalar, 3, 1> v0 = a.normalized();
        Matrix<Scalar, 3, 1> v1 = b.normalized();
        Scalar c                = v1.dot(v0);


        Matrix<Scalar, 3, 1> axis = v0.cross(v1);
        Scalar s                  = sqrt((Scalar(1) + c) * Scalar(2));
        Scalar invs               = Scalar(1) / s;

        axis = axis * invs;
        SameObject result;
        result.x() = axis.x();
        result.y() = axis.y();
        result.z() = axis.z();
        result.w() = s * Scalar(0.5);
        return result;
    }
    HD static SameObject FromAngleAxis(Scalar angle, Matrix<Scalar, 3, 1> v1)
    {
        Matrix<Scalar, 3, 1> vn = v1.normalized();

        angle *= 0.5f;
        float sinAngle = sin(angle);

        return SameObject(cos(angle), vn.x() * sinAngle, vn.y() * sinAngle, vn.z() * sinAngle);
    }

    HD Quaternion() {}
    HD Quaternion(_Scalar w, _Scalar x, _Scalar y, _Scalar z)
    {
        _data[0] = x;
        _data[1] = y;
        _data[2] = z;
        _data[3] = w;
    }

    HD Quaternion(const Matrix<Scalar, 4, 1>& coeff) { _data = coeff; }

    HD Quaternion(const Matrix<_Scalar, 3, 3>& rm)
    {
        Scalar t = rm(0, 0) + rm(1, 1) + rm(2, 2);
        if (t > 0)
        {
            Scalar s = 0.5 / std::sqrt(t + 1);
            (*this)  = Quaternion<Scalar>(0.25 / s, (rm(2, 1) - rm(1, 2)) * s, (rm(0, 2) - rm(2, 0)) * s,
                                         (rm(1, 0) - rm(0, 1)) * s);
        }
        else
        {
            if (rm(0, 0) > rm(1, 1) && rm(0, 0) > rm(2, 2))
            {
                Scalar s = 2.0 * std::sqrt(1.0 + rm(0, 0) - rm(1, 1) - rm(2, 2));
                (*this)  = Quaternion<Scalar>((rm(2, 1) - rm(1, 2)) / s, 0.25 * s, (rm(0, 1) + rm(1, 0)) / s,
                                             (rm(0, 2) + rm(2, 0)) / s);
            }
            else if (rm(1, 1) > rm(2, 2))
            {
                Scalar s = 2.0 * std::sqrt(1.0 + rm(1, 1) - rm(0, 0) - rm(2, 2));
                (*this)  = Quaternion<Scalar>((rm(0, 2) - rm(2, 0)) / s, (rm(0, 1) + rm(1, 0)) / s, 0.25 * s,
                                             (rm(1, 2) + rm(2, 1)) / s);
            }
            else
            {
                Scalar s = 2.0 * std::sqrt(1.0 + rm(2, 2) - rm(0, 0) - rm(1, 1));
                (*this)  = Quaternion<Scalar>((rm(1, 0) - rm(0, 1)) / s, (rm(0, 2) + rm(2, 0)) / s,
                                             (rm(1, 2) + rm(2, 1)) / s, 0.25 * s);
            }
        }
    }


    template <typename G>
    HD Quaternion<G> cast() const
    {
        return Quaternion<G>(w(), x(), y(), z());
    }


    HD Matrix<_Scalar, 3, 3> matrix() const
    {
        Matrix<_Scalar, 3, 3> result;


        Scalar s = w();
        Scalar x = _data[0];
        Scalar y = _data[1];
        Scalar z = _data[2];

        result(0, 0) = 1 - 2 * y * y - 2 * z * z;
        result(0, 1) = 2 * x * y - 2 * s * z;
        result(0, 2) = 2 * x * z + 2 * s * y;
        result(1, 0) = 2 * x * y + 2 * s * z;
        result(1, 1) = 1 - 2 * x * x - 2 * z * z;
        result(1, 2) = 2 * y * z - 2 * s * x;
        result(2, 0) = 2 * x * z - 2 * s * y;
        result(2, 1) = 2 * y * z + 2 * s * x;
        result(2, 2) = 1 - 2 * x * x - 2 * y * y;

        return result;
    }

    HD Matrix<_Scalar, 4, 1> coeffs() const { return _data; }

    HD Matrix<_Scalar, 4, 1>& coeffs() { return _data; }

    HD SameObject inverse() const { return SameObject(w(), -x(), -y(), -z()); }
    HD SameObject normalized() const
    {
        Scalar scale = 1 / norm();
        SameObject result;
        result.w() = w() * scale;
        result.x() = x() * scale;
        result.y() = y() * scale;
        result.z() = z() * scale;
        return result;
    }
    HD void normalize()
    {
        Scalar scale = 1 / norm();
        w() *= scale;
        x() *= scale;
        y() *= scale;
        z() *= scale;
    }
    HD Scalar norm() const { return coeffs().norm(); }

    HD SameObject slerp(Scalar alpha, SameObject other) const
    {
#if 1
        // Eigen implentation
        const Scalar one = Scalar(1) - std::numeric_limits<Scalar>::epsilon();
        Scalar d         = this->dot(other);
        Scalar absD      = std::abs(d);

        Scalar scale0;
        Scalar scale1;

        if (absD >= one)
        {
            scale0 = Scalar(1) - alpha;
            scale1 = alpha;
        }
        else
        {
            // theta is the angle between the 2 quaternions
            Scalar theta    = acos(absD);
            Scalar sinTheta = sin(theta);

            scale0 = sin((Scalar(1) - alpha) * theta) / sinTheta;
            scale1 = sin((alpha * theta)) / sinTheta;
        }
        if (d < Scalar(0)) scale1 = -scale1;

        return Quaternion<Scalar>(scale0 * coeffs() + scale1 * other.coeffs());
#endif
        Scalar cosHalfTheta = this->dot(other);
        if (std::abs(cosHalfTheta) >= 1.0)
        {
            return *this;
        }
        // Calculate temporary values.
        SameObject result;
        double halfTheta    = acos(cosHalfTheta);
        double sinHalfTheta = sqrt(1.0 - cosHalfTheta * cosHalfTheta);
        // if theta = 180 degrees then result is not fully defined
        // we could rotate around any axis normal to qa or qb
        if (fabs(sinHalfTheta) < 0.001)
        {  // fabs is floating point absolute
            result.w() = (this->w() * 0.5 + other.w() * 0.5);
            result.x() = (this->x() * 0.5 + other.x() * 0.5);
            result.y() = (this->y() * 0.5 + other.y() * 0.5);
            result.z() = (this->z() * 0.5 + other.z() * 0.5);
            return result;
        }
        double ratioA = sin((1 - alpha) * halfTheta) / sinHalfTheta;
        double ratioB = sin(alpha * halfTheta) / sinHalfTheta;
        // calculate Quaternion.
        result.w() = (this->w() * ratioA + other.w() * ratioB);
        result.x() = (this->x() * ratioA + other.x() * ratioB);
        result.y() = (this->y() * ratioA + other.y() * ratioB);
        result.z() = (this->z() * ratioA + other.z() * ratioB);
        return result;
    }

    Matrix<Scalar, 3, 1> vec() const { return Matrix<Scalar, 3, 1>(x(), y(), z()); }

    HD Scalar dot(const SameObject& other) const { return this->coeffs().dot(other.coeffs()); }

    HD Scalar& x() { return _data[0]; }
    HD Scalar& y() { return _data[1]; }
    HD Scalar& z() { return _data[2]; }
    HD Scalar& w() { return _data[3]; }
    HD const Scalar& x() const { return _data[0]; }
    HD const Scalar& y() const { return _data[1]; }
    HD const Scalar& z() const { return _data[2]; }
    HD const Scalar& w() const { return _data[3]; }

   private:
    Matrix<Scalar, 4, 1> _data;
};

template <typename _Scalar, int _Options>
HD Matrix<_Scalar, 3, 1, _Options> operator*(const Quaternion<_Scalar> quat, const Matrix<_Scalar, 3, 1, _Options>& v)
{
    Matrix<_Scalar, 3, 1, _Options> result;
    return quat.matrix() * v;
}

template <typename _Scalar>
HD Quaternion<_Scalar> operator*(const Quaternion<_Scalar> q1, const Quaternion<_Scalar>& q2)
{
    Quaternion<_Scalar> q;

    q.w() = q1.w() * q2.w() - q1.x() * q2.x() - q1.y() * q2.y() - q1.z() * q2.z();
    q.x() = q1.w() * q2.x() + q1.x() * q2.w() + q1.y() * q2.z() - q1.z() * q2.y();
    q.y() = q1.w() * q2.y() - q1.x() * q2.z() + q1.y() * q2.w() + q1.z() * q2.x();
    q.z() = q1.w() * q2.z() + q1.x() * q2.y() - q1.y() * q2.x() + q1.z() * q2.w();

    return q;
}

using Quaternionf = Quaternion<float>;
using Quaterniond = Quaternion<double>;

}  // namespace Eigen