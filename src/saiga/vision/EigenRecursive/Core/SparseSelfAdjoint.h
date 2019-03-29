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

/***************************************************************************
 * Implementation of sparse self-adjoint time dense matrix
 ***************************************************************************/
namespace Eigen
{
namespace internal
{
template <int Mode, typename SparseLhsType, typename DenseRhsType, typename DenseResType>
inline void sparse_selfadjoint_time_dense_product_recursive(const SparseLhsType& lhs, const DenseRhsType& rhs,
                                                            DenseResType& res)
{
    typedef typename internal::nested_eval<SparseLhsType, DenseRhsType::MaxColsAtCompileTime>::type SparseLhsTypeNested;
    typedef typename internal::remove_all<SparseLhsTypeNested>::type SparseLhsTypeNestedCleaned;
    typedef evaluator<SparseLhsTypeNestedCleaned> LhsEval;
    typedef typename LhsEval::InnerIterator LhsIterator;
    typedef typename SparseLhsType::Scalar LhsScalar;

    enum
    {
        LhsIsRowMajor    = (LhsEval::Flags & RowMajorBit) == RowMajorBit,
        ProcessFirstHalf = ((Mode & (Upper | Lower)) == (Upper | Lower)) || ((Mode & Upper) && !LhsIsRowMajor) ||
                           ((Mode & Lower) && LhsIsRowMajor),
        ProcessSecondHalf = !ProcessFirstHalf
    };

    SparseLhsTypeNested lhs_nested(lhs);
    LhsEval lhsEval(lhs_nested);

    // work on one column at once
    for (Index k = 0; k < rhs.cols(); ++k)
    {
        for (Index j = 0; j < lhs.outerSize(); ++j)
        {
            LhsIterator i(lhsEval, j);
            // handle diagonal coeff
            if (ProcessSecondHalf)
            {
                while (i && i.index() < j) ++i;
                if (i && i.index() == j)
                {
                    res.coeffRef(j, k) += i.value() * rhs.coeff(j, k);
                    ++i;
                }
            }

            // premultiplied rhs for scatters
            auto rhs_j = (rhs(j, k));
            // accumulator for partial scalar product
            typename DenseResType::Scalar res_j(0);
            for (; (ProcessFirstHalf ? i && i.index() < j : i); ++i)
            {
                LhsScalar lhs_ij = i.value();
                if (!LhsIsRowMajor) lhs_ij = transpose(lhs_ij);
                res_j += lhs_ij * rhs.coeff(i.index(), k);
                res(i.index(), k) += transpose(lhs_ij) * rhs_j;
            }
            res.coeffRef(j, k) += res_j;

            // handle diagonal coeff
            if (ProcessFirstHalf && i && (i.index() == j)) res.coeffRef(j, k) += i.value() * rhs.coeff(j, k);
        }
    }
}

template <typename T, int _Options, unsigned int _Mode, typename Rhs, int ProductType>
struct generic_product_impl<
    Eigen::SparseSelfAdjointView<typename Eigen::SparseMatrix<Eigen::Recursive::MatrixScalar<T>, _Options>, _Mode>, Rhs,
    SparseSelfAdjointShape, DenseShape, ProductType>
    : generic_product_impl_base<
          Eigen::SparseSelfAdjointView<typename Eigen::SparseMatrix<Eigen::Recursive::MatrixScalar<T>, _Options>,
                                       _Mode>,
          Rhs,
          generic_product_impl<Eigen::SparseSelfAdjointView<
                                   typename Eigen::SparseMatrix<Eigen::Recursive::MatrixScalar<T>, _Options>, _Mode>,
                               Rhs, SparseSelfAdjointShape, DenseShape, ProductType> >
{
    using LhsView =
        Eigen::SparseSelfAdjointView<typename Eigen::SparseMatrix<Eigen::Recursive::MatrixScalar<T>, _Options>, _Mode>;
    template <typename Dest>
    static void scaleAndAddTo(Dest& dst, const LhsView& lhsView, const Rhs& rhs, const typename Dest::Scalar&)
    {
        typedef typename LhsView::_MatrixTypeNested Lhs;
        typedef typename nested_eval<Lhs, Dynamic>::type LhsNested;
        typedef typename nested_eval<Rhs, Dynamic>::type RhsNested;
        LhsNested lhsNested(lhsView.matrix());
        RhsNested rhsNested(rhs);

        internal::sparse_selfadjoint_time_dense_product_recursive<LhsView::Mode>(lhsNested, rhsNested, dst);
    }
};

#if 0
template <typename Lhs, typename RhsView, int ProductType>
struct generic_product_impl<Lhs, RhsView, DenseShape, SparseSelfAdjointShape, ProductType>
    : generic_product_impl_base<Lhs, RhsView,
                                generic_product_impl<Lhs, RhsView, DenseShape, SparseSelfAdjointShape, ProductType> >
{
    template <typename Dest>
    static void scaleAndAddTo(Dest& dst, const Lhs& lhs, const RhsView& rhsView, const typename Dest::Scalar& alpha)
    {
        typedef typename RhsView::_MatrixTypeNested Rhs;
        typedef typename nested_eval<Lhs, Dynamic>::type LhsNested;
        typedef typename nested_eval<Rhs, Dynamic>::type RhsNested;
        LhsNested lhsNested(lhs);
        RhsNested rhsNested(rhsView.matrix());

        // transpose everything
        Transpose<Dest> dstT(dst);
        internal::sparse_selfadjoint_time_dense_product<RhsView::TransposeMode>(rhsNested.transpose(),
                                                                                lhsNested.transpose(), dstT, alpha);
    }
};
#endif

}  // namespace internal
}  // namespace Eigen
