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
struct SquaredNormImpl
{
    using Scalar    = typename T::Scalar;
    using ChildType = SquaredNormImpl<Scalar>;
    using BaseType  = typename ChildType::BaseType;

    // This is the actual recursive spezialization
    static BaseType get(const T& A)
    {
        BaseType sum = BaseType(0);
        for (int i = 0; i < A.outerSize(); ++i)
        {
            for (typename T::InnerIterator it(A, i); it; ++it)
            {
                sum += ChildType::get(it.value());
            }
        }
        return sum;
    }
};

template <>
struct SquaredNormImpl<double>
{
    using BaseType = double;
    static double get(double d) { return d * d; }
};

template <>
struct SquaredNormImpl<float>
{
    using BaseType = float;
    static float get(float d) { return d * d; }
};

template <typename G>
struct SquaredNormImpl<MatrixScalar<G>>
{
    using Scalar    = G;
    using ChildType = SquaredNormImpl<Scalar>;
    using BaseType  = typename ChildType::BaseType;
    static BaseType get(const MatrixScalar<G>& m) { return ChildType::get(m.get()); }
};


template <typename T>
auto squaredNorm(const T& v)
{
    return SquaredNormImpl<T>::get(v);
}

}  // namespace Eigen::Recursive
