/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/util/assert.h"
#include "saiga/vision/MatrixScalar.h"
#include "saiga/vision/recursiveMatrices/Expand.h"



namespace Saiga
{
template <typename LHS, typename RHS>
auto multSparseInner(const LHS& lhs, const RHS& rhs)
{
    // TODO: build
    //    cout << expand(lhs) << endl << endl;
    auto res = (lhs * rhs).eval();
    //    cout << expand(res) << endl << endl;
    return res;
}

}  // namespace Saiga
