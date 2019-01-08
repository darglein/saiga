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
template <typename T>
struct InverseImpl
{
    static T get(const T& m)
    {
        static_assert(T::RowsAtCompileTime == T::ColsAtCompileTime,
                      "The Symmetric Inverse is only defined for square matrices!");
        return m.inverse();
    }
};

template <>
struct InverseImpl<double>
{
    static double get(double d) { return 1.0 / d; }
};

template <typename G>
struct InverseImpl<MatrixScalar<G>>
{
    static MatrixScalar<G> get(const MatrixScalar<G>& m) { return MatrixScalar<G>(InverseImpl<G>::get(m.get())); }
};

template <typename T>
auto inverse(const T& v)
{
    return InverseImpl<T>::get(v);
}



}  // namespace Saiga
