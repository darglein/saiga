/**
 * This file contains (modified) code from the Eigen library.
 * Eigen License:
 *
 * Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
 * Copyright (C) 2007-2011 Benoit Jacob <jacob.benoit.1@gmail.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla
 * Public License v. 2.0. If a copy of the MPL was not distributed
 * with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * ======================
 *
 * The modifications are part of the Eigen Recursive Matrix Extension (ERME).
 * ERME License:
 *
 * Copyright (c) 2019 Darius Rückert
 * Licensed under the MIT License.
 */

#pragma once

#include "MatrixScalar.h"

#include <numeric>
namespace Eigen
{
namespace Recursive
{
template <typename T, typename T2>
inline void squaredNorm_omp_local(const T& v, T2& result)
{
    // using Scalar = typename BaseScalar<T>::type;

    result = 0;

#pragma omp for
    for (int i = 0; i < v.rows(); ++i)
    {
        result += v(i).get().squaredNorm();
    }
}

template <typename T, typename T2>
inline void dot_omp_local(const T& a, const T& b, T2& result)
{
    // using Scalar = typename BaseScalar<T>::type;

    result = 0;

#pragma omp for
    for (int i = 0; i < a.rows(); ++i)
    {
        result += a(i).get().dot(b(i).get());
    }
}

template <typename SparseLhsType, typename DenseRhsType, typename DenseResType>
inline void sparse_mv_omp(const SparseLhsType& lhs, const DenseRhsType& rhs, DenseResType& res)
{
    typedef typename internal::remove_all<SparseLhsType>::type Lhs;
    typedef Eigen::internal::evaluator<Lhs> LhsEval;
    typedef typename Eigen::internal::evaluator<Lhs>::InnerIterator LhsInnerIterator;

    //#pragma omp single
    {
        LhsEval lhsEval(lhs);
        Index n = lhs.outerSize();

        //        for (Index c = 0; c < rhs.cols(); ++c)
        {
#pragma omp for
            for (Index i = 0; i < n; ++i)
            {
                res.coeffRef(i).get().setZero();
                for (LhsInnerIterator it(lhs, i); it; ++it)
                {
                    auto& vlhs = it.value().get();
                    auto& vrhs = rhs.coeff(it.index()).get();
                    res.coeffRef(i).get() += vlhs * vrhs;
                }
            }
        }
    }
}


}  // namespace Recursive
}  // namespace Eigen
