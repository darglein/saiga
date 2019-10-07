/**
 * This file is part of the Eigen Recursive Matrix Extension (ERME).
 *
 * Copyright (c) 2019 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "MatrixScalar.h"
#include "NeutralElements.h"

namespace Eigen::internal
{
template <typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols, typename Rhs>
struct generic_product_impl<Matrix<Recursive::MatrixScalar<_Scalar>, _Rows, _Cols, _Options, _MaxRows, _MaxCols>, Rhs,
                            DenseShape, DenseShape, GemmProduct>
    : generic_product_impl_base<
          Matrix<Recursive::MatrixScalar<_Scalar>, _Rows, _Cols, _Options, _MaxRows, _MaxCols>, Rhs,
          generic_product_impl<Matrix<Recursive::MatrixScalar<_Scalar>, _Rows, _Cols, _Options, _MaxRows, _MaxCols>,
                               Rhs, DenseShape, DenseShape, GemmProduct>>
{
    using Lhs = Matrix<Recursive::MatrixScalar<_Scalar>, _Rows, _Cols, _Options, _MaxRows, _MaxCols>;
    using BS  = typename Recursive::BaseScalar<Lhs>::type;
    typedef typename Product<Lhs, Rhs>::Scalar Scalar;
    typedef typename Lhs::Scalar LhsScalar;
    typedef typename Rhs::Scalar RhsScalar;

    typedef internal::blas_traits<Lhs> LhsBlasTraits;
    typedef typename LhsBlasTraits::DirectLinearAccessType ActualLhsType;
    typedef typename internal::remove_all<ActualLhsType>::type ActualLhsTypeCleaned;

    typedef internal::blas_traits<Rhs> RhsBlasTraits;
    typedef typename RhsBlasTraits::DirectLinearAccessType ActualRhsType;
    typedef typename internal::remove_all<ActualRhsType>::type ActualRhsTypeCleaned;

    enum
    {
        MaxDepthAtCompileTime = EIGEN_SIZE_MIN_PREFER_FIXED(Lhs::MaxColsAtCompileTime, Rhs::MaxRowsAtCompileTime)
    };

    typedef generic_product_impl<Lhs, Rhs, DenseShape, DenseShape, CoeffBasedProductMode> lazyproduct;

    template <typename Dst>
    static void evalTo(Dst& dst, const Lhs& lhs, const Rhs& rhs)
    {
        // See http://eigen.tuxfamily.org/bz/show_bug.cgi?id=404 for a discussion and helper program
        // to determine the following heuristic.
        // EIGEN_GEMM_TO_COEFFBASED_THRESHOLD is typically defined to 20 in GeneralProduct.h,
        // unless it has been specialized by the user or for a given architecture.
        // Note that the condition rhs.rows()>0 was required because lazy product is (was?) not happy with empty inputs.
        // I'm not sure it is still required.
//        if ((rhs.rows() + dst.rows() + dst.cols()) < EIGEN_GEMM_TO_COEFFBASED_THRESHOLD && rhs.rows() > 0)
//        {
//            std::cout << "lazy eval" << std::endl;
//            lazyproduct::eval_dynamic(dst, lhs, rhs, internal::assign_op<typename Dst::Scalar, Scalar>());
//        }
//        else
        {
            dst.setZero();
            scaleAndAddTo(dst, lhs, rhs, BS(1));
        }
    }

    template <typename Dst>
    static void addTo(Dst& dst, const Lhs& lhs, const Rhs& rhs)
    {
//        if ((rhs.rows() + dst.rows() + dst.cols()) < EIGEN_GEMM_TO_COEFFBASED_THRESHOLD && rhs.rows() > 0)
//        {
//            std::cout << "lazy eval" << std::endl;
//            lazyproduct::eval_dynamic(dst, lhs, rhs, internal::add_assign_op<typename Dst::Scalar, Scalar>());
//        }
//        else
            scaleAndAddTo(dst, lhs, rhs, BS(1));
    }

    template <typename Dst>
    static void subTo(Dst& dst, const Lhs& lhs, const Rhs& rhs)
    {
//        if ((rhs.rows() + dst.rows() + dst.cols()) < EIGEN_GEMM_TO_COEFFBASED_THRESHOLD && rhs.rows() > 0)
//        {
//            std::cout << "lazy eval" << std::endl;
//            lazyproduct::eval_dynamic(dst, lhs, rhs, internal::sub_assign_op<typename Dst::Scalar, Scalar>());
//        }
//        else
            scaleAndAddTo(dst, lhs, rhs, BS(-1));
    }

    template <typename Dest>
    static void scaleAndAddTo(Dest& dst, const Lhs& a_lhs, const Rhs& a_rhs, const BS& alpha)
    {
        eigen_assert(dst.rows() == a_lhs.rows() && dst.cols() == a_rhs.cols());
        if (a_lhs.cols() == 0 || a_lhs.rows() == 0 || a_rhs.cols() == 0) return;

        // Fallback to GEMV if either the lhs or rhs is a runtime vector
        //        if (dst.cols() == 1)
        //        {
        //            typename Dest::ColXpr dst_vec(dst.col(0));
        //            return internal::generic_product_impl<Lhs, typename Rhs::ConstColXpr, DenseShape, DenseShape,
        //                                                  GemvProduct>::scaleAndAddTo(dst_vec, a_lhs, a_rhs.col(0),
        //                                                  alpha);
        //        }
        //        else if (dst.rows() == 1)
        //        {
        //            typename Dest::RowXpr dst_vec(dst.row(0));
        //            return internal::generic_product_impl<typename Lhs::ConstRowXpr, Rhs, DenseShape, DenseShape,
        //                                                  GemvProduct>::scaleAndAddTo(dst_vec, a_lhs.row(0), a_rhs,
        //                                                  alpha);
        //        }


        typename internal::add_const_on_value_type<ActualLhsType>::type lhs = LhsBlasTraits::extract(a_lhs);
        typename internal::add_const_on_value_type<ActualRhsType>::type rhs = RhsBlasTraits::extract(a_rhs);

//        BS actualAlpha =
//            alpha * LhsBlasTraits::extractScalarFactor(a_lhs) * RhsBlasTraits::extractScalarFactor(a_rhs);
#if 0

        typedef internal::gemm_blocking_space<(Dest::Flags & RowMajorBit) ? RowMajor : ColMajor, LhsScalar, RhsScalar,
                                              Dest::MaxRowsAtCompileTime, Dest::MaxColsAtCompileTime,
                                              MaxDepthAtCompileTime>
            BlockingType;

//        internal::gebp_kernel<LhsScalar, RhsScalar, Index, ResMapper, Traits::mr, Traits::nr, ConjugateLhs, ConjugateRhs> gebp;

        typedef internal::gemm_functor<
            Scalar, Index,
            internal::general_matrix_matrix_product<
                Index, LhsScalar, (ActualLhsTypeCleaned::Flags & RowMajorBit) ? RowMajor : ColMajor,
                bool(LhsBlasTraits::NeedToConjugate), RhsScalar,
                (ActualRhsTypeCleaned::Flags & RowMajorBit) ? RowMajor : ColMajor, bool(RhsBlasTraits::NeedToConjugate),
                (Dest::Flags & RowMajorBit) ? RowMajor : ColMajor>,
            ActualLhsTypeCleaned, ActualRhsTypeCleaned, Dest, BlockingType>
            GemmFunctor;

        BlockingType blocking(dst.rows(), dst.cols(), lhs.cols(), 1, true);
        internal::parallelize_gemm<(Dest::MaxRowsAtCompileTime > 32 || Dest::MaxRowsAtCompileTime == Dynamic)>(
            GemmFunctor(lhs, rhs, dst, actualAlpha, blocking), a_lhs.rows(), a_rhs.cols(), a_lhs.cols(),
            Dest::Flags & RowMajorBit);
    }
#else
//        std::cout << "test" << std::endl;

        for (int i = 0; i < dst.rows(); i++)

        {

            for (int j = 0; j < dst.cols(); j++)

            {

//                Eigen::Recursive::setZero(dst(i,j));

                //                C[i][j] = 0;

                for (int k = 0; k < lhs.cols(); k++)

                {

                    //                    C[i][j] += A[i][k] * B[k][j];
                    dst(i,j) += alpha * lhs(i,k) * rhs(k,j);

                }

            }

        }
    }
#endif
    };

}  // namespace Eigen::internal
