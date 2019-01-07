/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/util/assert.h"
#include "saiga/vision/MatrixScalar.h"

namespace Saiga
{
/**
 * Multiplicative neutral element e
 *
 * A * e = A
 */
template <typename T>
struct MultiplicativeNeutral
{
};

template <>
struct MultiplicativeNeutral<double>
{
    static double get() { return 1.0; }
};

template <typename G>
struct MultiplicativeNeutral<MatrixScalar<G>>
{
    static MatrixScalar<G> get()
    {
        static_assert(G::RowsAtCompileTime == G::ColsAtCompileTime,
                      "The Multiplicative Neutral Element is only defined for square matrices!");
        return MatrixScalar<G>(G::Identity());
    }
};

/**
 * Additive neutral element e
 *
 * A + e = A
 */
template <typename T>
struct AdditiveNeutral
{
};

template <>
struct AdditiveNeutral<double>
{
    static double get() { return 0.0; }
};

template <typename G>
struct AdditiveNeutral<MatrixScalar<G>>
{
    static MatrixScalar<G> get() { return MatrixScalar<G>(G::Zero()); }
};

}  // namespace Saiga
