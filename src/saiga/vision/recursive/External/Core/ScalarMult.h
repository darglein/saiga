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
struct ScalarMultImpl
{
    using Scalar    = typename T::Scalar;
    using ChildType = ScalarMultImpl<Scalar>;
    using BaseType  = typename ChildType::BaseType;

    // This is the actual recursive spezialization
    static T get(const T& a, BaseType b)
    {
        T result;
        result.resize(a.rows());

        for (int i = 0; i < a.rows(); ++i)
        {
            result(i) = ChildType::get(a(i), b);
        }
        return result;
    }
};

template <>
struct ScalarMultImpl<double>
{
    using BaseType = double;
    static double get(double a, double b) { return a * b; }
};

template <>
struct ScalarMultImpl<float>
{
    using BaseType = float;
    static float get(float a, float b) { return a * b; }
};

template <typename G>
struct ScalarMultImpl<MatrixScalar<G>>
{
    using Scalar    = G;
    using ChildType = ScalarMultImpl<Scalar>;
    using BaseType  = typename ChildType::BaseType;
    static MatrixScalar<G> get(const MatrixScalar<G>& a, const BaseType& b) { return {ChildType::get(a.get(), b)}; }
};


template <typename T, typename G>
auto scalarMult(const T& a, const G& b)
{
    return ScalarMultImpl<T>::get(a, b);
}

}  // namespace Eigen::Recursive
