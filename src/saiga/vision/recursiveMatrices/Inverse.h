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
struct InverseSymmetric
{
};

template <>
struct InverseSymmetric<double>
{
    static double get(double d) { return 1.0 / d; }
};

template <typename G>
struct InverseSymmetric<MatrixScalar<G>>
{
    static MatrixScalar<G> get(const MatrixScalar<G>& m)
    {
        static_assert(G::RowsAtCompileTime == G::ColsAtCompileTime,
                      "The Symmetric Inverse is only defined for square matrices!");
        return MatrixScalar<G>(m.get().inverse());
    }
};
}  // namespace Saiga
