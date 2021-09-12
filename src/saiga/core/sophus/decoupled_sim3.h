/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once


#include "SophusSelector.h"


namespace Sophus
{
// Decoupled implementation of the Sophus::Sim3.
//
// In the original Sophus::Sim3, the scale is stored in the quaternion with
// sqrt(scale) = |sim3.quaternion|
//
// DSim3 decouples the scale and quaternion by storing the scale separately. This simplifies a lot of operations and
// makes autodiff on DSim3 more robust.
template <typename T>
struct DSim3
{
    static constexpr int DoF = 7;
    using Scalar             = T;
    using Tangent            = Eigen::Matrix<T, DoF, 1>;

    DSim3()
    {
        se3()   = SE3<T>();
        scale() = 1.0;

        static_assert(sizeof(DSim3<T>) == 8 * sizeof(T), "DSim size is incorrect!");
    }
    DSim3(const SE3<T>& se3, T scale = T(1))
    {
        this->se3()   = se3;
        this->scale() = scale;
    }

    DSim3(const Sim3<T>& sim3)
    {
        this->se3()   = SE3<T>(sim3.rxso3().quaternion().normalized(), sim3.translation());
        this->scale() = sim3.scale();
    }

    DSim3(const Eigen::Matrix<T, 8, 1>& data) : data_(data) {}

    template <class OtherDerived, typename PointDerived>
    DSim3(SO3Base<OtherDerived> const& so3, const Eigen::MatrixBase<PointDerived>& t, T scale)
    {
        this->se3()   = SE3<T>(so3, t);
        this->scale() = scale;
    }

    /// Group multiplication, which is rotation concatenation.
    ///
    SOPHUS_FUNC DSim3<T> operator*(DSim3<T> const& other) const
    {
        return {se3().so3() * other.se3().so3(),
                se3().translation() + scale() * (se3().so3() * other.se3().translation()), scale() * other.scale()};
    }

    /// Group action on 3-points.
    ///
    /// This function rotates and translates a three dimensional point ``p`` by
    /// the SE(3) element ``bar_T_foo = (bar_R_foo, t_bar)`` (= rigid body
    /// transformation):
    ///
    ///   ``p_bar = bar_R_foo * p_foo + t_bar``.
    ///
    template <typename PointDerived,
              typename = typename std::enable_if<IsFixedSizeVector<PointDerived, 3>::value>::type>
    SOPHUS_FUNC auto operator*(Eigen::MatrixBase<PointDerived> const& p) const
    {
        return (scale() * (se3().so3() * p) + se3().translation()).eval();
    }

    SOPHUS_FUNC DSim3<T> inverse() const
    {
        SE3<T> t_inv = se3().inverse();
        T s_inv      = T(1.0) / scale();
        t_inv.translation() *= s_inv;
        return {t_inv, s_inv};
    }

    T* data() { return data_.data(); }
    Eigen::Matrix<T, 8, 1> params() const { return data_; }

    template <typename G>
    DSim3<G> cast() const
    {
        return DSim3<G>(data_.template cast<G>());
    }

    SE3<T>& se3() { return *((SE3<T>*)data_.data()); }
    T& scale() { return data_(7); }
    const SE3<T>& se3() const { return *((const SE3<T>*)data_.data()); }
    T scale() const { return data_(7); }

   protected:
    Eigen::Matrix<T, 8, 1> data_;
};

using DSim3d = DSim3<double>;


template <typename T>
inline std::ostream& operator<<(std::ostream& os, const Sophus::DSim3<T>& sim3)
{
    os << "Sim3(" << sim3.se3() << " Scale=" << sim3.scale() << ")";
    return os;
}

template <typename T>
inline DSim3<T> PointerToDSim3(const T* ptr)
{
    Eigen::Map<const Vector<T, 8>> const x(ptr);
    DSim3<T> result(x);
    return result;
}


template <typename T>
inline void DSim3ToPointer(const DSim3<T>& sim3, T* ptr)
{
    Eigen::Map<Vector<T, 8>> x(ptr);
    x = sim3.params();
}

}  // namespace Sophus
