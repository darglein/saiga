/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "config.h"
#include "matrix.h"
#include "quaternion.h"

namespace Sophus
{
template <typename T>
class SO3
{
    using Vec3   = Eigen::Matrix<T, 3, 1>;
    using Quat   = Eigen::Quaternion<T>;
    using Scalar = T;

   public:
    HD SO3() : q(Quat::Identity()) {}
    HD SO3(const Quat& q) : q(q.normalized()) {}
    HD SO3(const Eigen::Matrix<T, 3, 3>& R) : q(R) { q = q.normalized(); }
    HD Quat& unit_quaternion() { return q; }
    HD const Quat& unit_quaternion() const { return q; }

    HD SO3<Scalar> inverse() const { return SO3<Scalar>(q.inverse()); }


    template <typename G>
    HD SO3<G> cast()
    {
        return SO3<G>(q.template cast<G>());
    }

    HD static SO3<T> exp(Vec3 const& omega)
    {
        using std::abs;
        using std::cos;
        using std::sin;
        using std::sqrt;
        Scalar theta_sq   = omega.squaredNorm();
        Scalar theta      = sqrt(theta_sq);
        Scalar half_theta = Scalar(0.5) * (theta);

        Scalar imag_factor;
        Scalar real_factor;
        if ((theta) < std::numeric_limits<T>::epsilon())
        {
            Scalar theta_po4 = theta_sq * theta_sq;
            imag_factor      = Scalar(0.5) - Scalar(1.0 / 48.0) * theta_sq + Scalar(1.0 / 3840.0) * theta_po4;
            real_factor      = Scalar(1) - Scalar(1.0 / 8.0) * theta_sq + Scalar(1.0 / 384.0) * theta_po4;
        }
        else
        {
            Scalar sin_half_theta = sin(half_theta);
            imag_factor           = sin_half_theta / (theta);
            real_factor           = cos(half_theta);
        }

        SO3 q;
        q.unit_quaternion() =
            Quat(real_factor, imag_factor * omega.x(), imag_factor * omega.y(), imag_factor * omega.z());
        return q;
    }

    HD Vec3 log() const
    {
        using std::abs;
        using std::atan;
        using std::sqrt;
        Scalar squared_n = unit_quaternion().vec().squaredNorm();
        Scalar n         = sqrt(squared_n);
        Scalar w         = unit_quaternion().w();

        Scalar two_atan_nbyw_by_n;

        /// Atan-based log thanks to
        ///
        /// C. Hertzberg et al.:
        /// "Integrating Generic Sensor Fusion Algorithms with Sound State
        /// Representation through Encapsulation of Manifolds"
        /// Information Fusion, 2011

        if (n < std::numeric_limits<T>::epsilon())
        {
            /// If quaternion is normalized and n=0, then w should be 1;
            /// w=0 should never happen here!
            Scalar squared_w   = w * w;
            two_atan_nbyw_by_n = Scalar(2) / w - Scalar(2) * (squared_n) / (w * squared_w);
        }
        else
        {
            if (abs(w) < std::numeric_limits<T>::epsilon())
            {
                if (w > Scalar(0))
                {
                    two_atan_nbyw_by_n = 3.14159265358979323846 / n;
                }
                else
                {
                    two_atan_nbyw_by_n = -3.14159265358979323846 / n;
                }
            }
            else
            {
                two_atan_nbyw_by_n = Scalar(2) * atan(n / w) / n;
            }
        }

        return two_atan_nbyw_by_n * unit_quaternion().vec();
    }

    HD Eigen::Matrix<T, 3, 3> matrix() const { return q.matrix(); }


   private:
    Quat q;
};


template <typename T>
class alignas(sizeof(T) * 8) SE3
{
    using Vec3   = Eigen::Matrix<T, 3, 1>;
    using Quat   = Eigen::Quaternion<T>;
    using Scalar = T;


   public:
    HD SE3() : t(0, 0, 0) {}
    HD SE3(const Quat& q, const Vec3& v) : _so3(q), t(v) {}
    HD SE3(const SO3<T>& q, const Vec3& v) : _so3(q), t(v) {}
    HD SE3(const Eigen::Matrix<T, 4, 4>& tra) : _so3(tra.template block<3, 3>(0, 0)), t(tra.template block<3, 1>(0, 3))
    {
    }


    HD static SE3<T> fitToSE3(const Eigen::Matrix<T, 4, 4>& tra) { return SE3(tra); }

    template <typename G>
    HD SE3<G> cast()
    {
        return SE3<G>(_so3.template cast<G>(), t.template cast<G>());
    }


    HD void setQuaternion(const Quat& q) { so3() = SO3<Scalar>(q); }

    HD Vec3& translation() { return t; }
    HD Quat& unit_quaternion() { return so3().unit_quaternion(); }
    HD const Vec3& translation() const { return t; }
    HD const Quat& unit_quaternion() const { return so3().unit_quaternion(); }

    HD SE3<Scalar> inverse() const
    {
        SO3<Scalar> invR = so3().inverse();
        return SE3<Scalar>(invR, invR * (translation() * Scalar(-1)));
    }

    HD SO3<T>& so3() { return _so3; }
    HD const SO3<T>& so3() const { return _so3; }

    HD const Scalar* data() const { return (Scalar*)(this); }
    HD Scalar* data() { return (Scalar*)(this); }

    HD Eigen::Matrix<T, 4, 4> matrix() const
    {
        Eigen::Matrix<T, 4, 4> result     = Eigen::Matrix<T, 4, 4>::Identity();
        result.template block<3, 3>(0, 0) = unit_quaternion().matrix();
        result.template block<3, 1>(0, 3) = t;
        return result;
    }


    HD Eigen::Matrix<T, 7, 1> params() const
    {
        Eigen::Matrix<T, 7, 1> data;
        data.template head<4>() = unit_quaternion().coeffs();
        data(4)                 = t(0);
        data(5)                 = t(1);
        data(6)                 = t(2);
        return data;
    }

   private:
    SO3<T> _so3;
    Vec3 t;
};

template <typename T>
class Sim3
{
   public:
    Sim3() {}
};

template <typename T>
HD Eigen::Matrix<T, 3, 1> operator*(const SO3<T>& a, const Eigen::Matrix<T, 3, 1>& v)
{
    return a.unit_quaternion() * v;
}


template <typename T>
HD Eigen::Matrix<T, 3, 1> operator*(const SE3<T>& a, const Eigen::Matrix<T, 3, 1>& v)
{
    return a.unit_quaternion() * v + a.translation();
}

template <typename T>
HD SO3<T> operator*(const SO3<T>& a, const SO3<T>& b)
{
    return SO3<T>(a.unit_quaternion() * b.unit_quaternion());
}

template <typename T>
HD SE3<T> operator*(const SE3<T>& a, const SE3<T>& b)
{
    //  so3() * other.so3(), translation() + so3() * other.translation()
    return SE3<T>(a.so3() * b.so3(), a.translation() + a.so3() * b.translation());
}



using SE3d = SE3<double>;
using SO3d = SO3<double>;
using SE3f = SE3<float>;
using SO3f = SO3<float>;
}  // namespace Sophus