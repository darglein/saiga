/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/util/assert.h"
#include "saiga/vision/recursiveMatrices/MatrixScalar.h"

namespace Saiga
{
template <typename T>
struct Transpose
{
    static auto get(const T& m) { return (m.transpose().eval()); }
};

template <>
struct Transpose<double>
{
    static double get(double d) { return d; }
};

template <>
struct Transpose<float>
{
    static float get(float d) { return d; }
};

template <typename G>
struct Transpose<MatrixScalar<G>>
{
    //    static auto get(const MatrixScalar<G>& m) { return m.transpose(); }
    static auto get(const MatrixScalar<G>& m) { return makeMatrixScalar(Transpose<G>::get(m.get())); }
};


template <typename T>
auto transpose(const T& v)
{
    return Transpose<T>::get(v);
}

}  // namespace Saiga
