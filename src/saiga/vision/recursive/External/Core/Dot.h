/**
 * This file is part of the Eigen Recursive Matrix Extension (ERME).
 *
 * Copyright (c) 2019 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "MatrixScalar.h"

namespace Eigen::Recursive
{
template <typename T>
struct DotImpl
{
    using Scalar    = typename T::Scalar;
    using ChildType = DotImpl<Scalar>;
    using BaseType  = typename ChildType::BaseType;

    // This is the actual recursive spezialization
    static BaseType get(const T& a, const T& b)
    {
        BaseType sum = BaseType(0);
        for (int i = 0; i < a.rows(); ++i)
        {
            sum += ChildType::get(a(i), b(i));
        }
        return sum;
    }
};

template <>
struct DotImpl<double>
{
    using BaseType = double;
    static double get(double a, double b) { return a * b; }
};

template <>
struct DotImpl<float>
{
    using BaseType = float;
    static float get(float a, float b) { return a * b; }
};

template <typename G>
struct DotImpl<MatrixScalar<G>>
{
    using Scalar    = G;
    using ChildType = DotImpl<Scalar>;
    using BaseType  = typename ChildType::BaseType;
    static BaseType get(const MatrixScalar<G>& a, const MatrixScalar<G>& b) { return ChildType::get(a.get(), b.get()); }
};


template <typename T>
auto dot(const T& a, const T& b)
{
    return DotImpl<T>::get(a, b);
}

}  // namespace Eigen::Recursive
