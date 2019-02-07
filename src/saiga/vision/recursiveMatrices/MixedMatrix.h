/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/assert.h"
#include "saiga/vision/recursiveMatrices/MatrixScalar.h"

namespace Saiga
{
/**
 * A symmetric 2x2 Matrix, where each element has a different type.
 * Layout:
 *
 * | U  W |
 * | WT V |
 *
 */
template <typename U, typename V, typename W, typename WT>
struct SymmetricMixedMatrix22
{
    using UType  = U;
    using VType  = V;
    using WType  = W;
    using WTType = WT;

    UType u;
    VType v;
    WType w;
    WTType wt;
};

template <typename U, typename V, typename W>
struct SymmetricMixedMatrix2
{
    using UType = U;
    using VType = V;
    using WType = W;

    UType u;
    VType v;
    WType w;
};


/**
 * A simple 2x1 vector with two different types.
 *
 * | U |
 * | V |
 */
template <typename U, typename V>
struct MixedVector2
{
    using UType = U;
    using VType = V;

    UType u;
    VType v;
};

}  // namespace Saiga
