/**
 * This file is part of the Eigen Recursive Matrix Extension (ERME).
 *
 * Copyright (c) 2019 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "../Core.h"

namespace Eigen::Recursive
{
//
// A symmetric 2x2 Matrix, where each element has a different type.
// Layout:
//
// | U  W |
// | WT V |
//
//
template <typename U, typename V, typename W>
struct SymmetricMixedMatrix2
{
    using UType = U;
    using VType = V;
    using WType = W;

    UType u;
    VType v;
    WType w;

    void resize(int n, int m)
    {
        u.resize(n);
        v.resize(m);
        w.resize(n, m);
    }
};


//
// A simple 2x1 vector with two different types.
//
// | U |
// | V |
//
template <typename U, typename V>
struct MixedVector2
{
    using UType = U;
    using VType = V;

    UType u;
    VType v;

    void resize(int n, int m)
    {
        u.resize(n);
        v.resize(m);
    }

    void setZero()
    {
        u.setZero();
        v.setZero();
    }
};

}  // namespace Eigen::Recursive
