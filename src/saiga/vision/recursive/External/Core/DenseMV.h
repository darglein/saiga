/**
 * This file is part of the Eigen Recursive Matrix Extension (ERME).
 *
 * Copyright (c) 2019 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "MatrixScalar.h"


namespace Eigen::Recursive
{
template <typename T>
struct ExtractScale
{
};


template <>
struct ExtractScale<double>
{
    static double set(double value) { return value; }
};

template <>
struct ExtractScale<float>
{
    static float set(float value) { return value; }
};

template <typename _Matrix>
struct ExtractScale<MatrixScalar<_Matrix>>
{
    using ChildExpansion = ExtractScale<typename _Matrix::Scalar>;
    using MatrixType     = MatrixScalar<_Matrix>;
    using BS             = typename BaseScalar<_Matrix>::type;

    static BS set(const MatrixType& A) { return ChildExpansion::set(A(0, 0)); }
};

template <typename T>
auto extractScale(const T& A)
{
    return ExtractScale<T>::set(A);
}

}  // namespace Eigen::Recursive
namespace Eigen::internal
{
/* Optimized col-major matrix * vector product:
 * This algorithm processes the matrix per vertical panels,
 * which are then processed horizontaly per chunck of 8*PacketSize x 1 vertical segments.
 *
 * Mixing type logic: C += alpha * A * B
 *  |  A  |  B  |alpha| comments
 *  |real |cplx |cplx | no vectorization
 *  |real |cplx |real | alpha is converted to a cplx when calling the run function, no vectorization
 *  |cplx |real |cplx | invalid, the caller has to do tmp: = A * B; C += alpha*tmp
 *  |cplx |real |real | optimal case, vectorization possible via real-cplx mul
 *
 * The same reasoning apply for the transposed case.
 */
template <typename Index, typename LhsScalar2, typename LhsMapper, bool ConjugateLhs, typename RhsScalar,
          typename RhsMapper, bool ConjugateRhs, int Version>
struct general_matrix_vector_product<Index, Recursive::MatrixScalar<LhsScalar2>, LhsMapper, ColMajor, ConjugateLhs,
                                     RhsScalar, RhsMapper, ConjugateRhs, Version>
{
    using LhsScalar = Recursive::MatrixScalar<LhsScalar2>;
    typedef typename ScalarBinaryOpTraits<LhsScalar, RhsScalar>::ReturnType ResScalar;

    using BS = typename Recursive::BaseScalar<LhsScalar>::type;


    EIGEN_DEVICE_FUNC EIGEN_DONT_INLINE static void run(Index rows, Index cols, const LhsMapper& alhs,
                                                        const RhsMapper& rhs, ResScalar* res, Index resIncr, BS alpha)
    {
        // using LhsScalar = Recursive::MatrixScalar<LhsScalar2>;
        EIGEN_UNUSED_VARIABLE(resIncr);
        eigen_internal_assert(resIncr == 1);

        // The following copy tells the compiler that lhs's attributes are not modified outside this function
        // This helps GCC to generate propoer code.
        LhsMapper lhs(alhs);

        for (int i = 0; i < rows; ++i)
        {
            res[i] = 0;
        }

        for (Index j2 = 0; j2 < cols; j2 += 1)
        {
            for (int i = 0; i < rows; ++i)
            {
                res[i] += alpha * (lhs(i, j2) * rhs(j2, 0));
            }
        }
    }
};

// template <typename Index, typename LhsScalar2, typename LhsMapper, bool ConjugateLhs, typename RhsScalar,
//          typename RhsMapper, bool ConjugateRhs, int Version>
// EIGEN_DEVICE_FUNC EIGEN_DONT_INLINE void
// general_matrix_vector_product<Index, Recursive::MatrixScalar<LhsScalar2>, LhsMapper, ColMajor, ConjugateLhs,
// RhsScalar,
//                              RhsMapper, ConjugateRhs, Version>::run(Index rows, Index cols, const LhsMapper& alhs,
//                                                                     const RhsMapper& rhs, ResScalar* res,
//                                                                     Index resIncr, RhsScalar alpha)
//{

//}

/* Optimized row-major matrix * vector product:
 * This algorithm processes 4 rows at once that allows to both reduce
 * the number of load/stores of the result by a factor 4 and to reduce
 * the instruction dependency. Moreover, we know that all bands have the
 * same alignment pattern.
 *
 * Mixing type logic:
 *  - alpha is always a complex (or converted to a complex)
 *  - no vectorization
 */
template <typename Index, typename LhsScalar2, typename LhsMapper, bool ConjugateLhs, typename RhsScalar,
          typename RhsMapper, bool ConjugateRhs, int Version>
struct general_matrix_vector_product<Index, Recursive::MatrixScalar<LhsScalar2>, LhsMapper, RowMajor, ConjugateLhs,
                                     RhsScalar, RhsMapper, ConjugateRhs, Version>
{
    using LhsScalar = Recursive::MatrixScalar<LhsScalar2>;
    typedef typename ScalarBinaryOpTraits<LhsScalar, RhsScalar>::ReturnType ResScalar;

    enum
    {
        Vectorizable = packet_traits<LhsScalar>::Vectorizable && packet_traits<RhsScalar>::Vectorizable &&
                       int(packet_traits<LhsScalar>::size) == int(packet_traits<RhsScalar>::size),
        LhsPacketSize = Vectorizable ? packet_traits<LhsScalar>::size : 1,
        RhsPacketSize = Vectorizable ? packet_traits<RhsScalar>::size : 1,
        ResPacketSize = Vectorizable ? packet_traits<ResScalar>::size : 1
    };

    typedef typename packet_traits<LhsScalar>::type _LhsPacket;
    typedef typename packet_traits<RhsScalar>::type _RhsPacket;
    typedef typename packet_traits<ResScalar>::type _ResPacket;

    typedef typename conditional<Vectorizable, _LhsPacket, LhsScalar>::type LhsPacket;
    typedef typename conditional<Vectorizable, _RhsPacket, RhsScalar>::type RhsPacket;
    typedef typename conditional<Vectorizable, _ResPacket, ResScalar>::type ResPacket;

    EIGEN_DEVICE_FUNC EIGEN_DONT_INLINE static void run(Index rows, Index cols, const LhsMapper& lhs,
                                                        const RhsMapper& rhs, ResScalar* res, Index resIncr,
                                                        ResScalar alpha);
};

template <typename Index, typename LhsScalar2, typename LhsMapper, bool ConjugateLhs, typename RhsScalar,
          typename RhsMapper, bool ConjugateRhs, int Version>
EIGEN_DEVICE_FUNC EIGEN_DONT_INLINE void
general_matrix_vector_product<Index, Recursive::MatrixScalar<LhsScalar2>, LhsMapper, RowMajor, ConjugateLhs, RhsScalar,
                              RhsMapper, ConjugateRhs, Version>::run(Index rows, Index cols, const LhsMapper& alhs,
                                                                     const RhsMapper& rhs, ResScalar* res,
                                                                     Index resIncr, ResScalar alpha)
{
    using LhsScalar = Recursive::MatrixScalar<LhsScalar2>;
    // The following copy tells the compiler that lhs's attributes are not modified outside this function
    // This helps GCC to generate propoer code.
    LhsMapper lhs(alhs);

    eigen_internal_assert(rhs.stride() == 1);
    conj_helper<LhsScalar, RhsScalar, ConjugateLhs, ConjugateRhs> cj;
    conj_helper<LhsPacket, RhsPacket, ConjugateLhs, ConjugateRhs> pcj;

    // TODO: fine tune the following heuristic. The rationale is that if the matrix is very large,
    //       processing 8 rows at once might be counter productive wrt cache.
    const Index n8 = lhs.stride() * sizeof(LhsScalar) > 32000 ? 0 : rows - 7;
    const Index n4 = rows - 3;
    const Index n2 = rows - 1;

    // TODO: for padded aligned inputs, we could enable aligned reads
    enum
    {
        LhsAlignment = Unaligned
    };

    Index i = 0;
    for (; i < n8; i += 8)
    {
        ResPacket c0 = pset1<ResPacket>(ResScalar(0)), c1 = pset1<ResPacket>(ResScalar(0)),
                  c2 = pset1<ResPacket>(ResScalar(0)), c3 = pset1<ResPacket>(ResScalar(0)),
                  c4 = pset1<ResPacket>(ResScalar(0)), c5 = pset1<ResPacket>(ResScalar(0)),
                  c6 = pset1<ResPacket>(ResScalar(0)), c7 = pset1<ResPacket>(ResScalar(0));

        Index j = 0;
        for (; j + LhsPacketSize <= cols; j += LhsPacketSize)
        {
            RhsPacket b0 = rhs.template load<RhsPacket, Unaligned>(j, 0);

            c0 = pcj.pmadd(lhs.template load<LhsPacket, LhsAlignment>(i + 0, j), b0, c0);
            c1 = pcj.pmadd(lhs.template load<LhsPacket, LhsAlignment>(i + 1, j), b0, c1);
            c2 = pcj.pmadd(lhs.template load<LhsPacket, LhsAlignment>(i + 2, j), b0, c2);
            c3 = pcj.pmadd(lhs.template load<LhsPacket, LhsAlignment>(i + 3, j), b0, c3);
            c4 = pcj.pmadd(lhs.template load<LhsPacket, LhsAlignment>(i + 4, j), b0, c4);
            c5 = pcj.pmadd(lhs.template load<LhsPacket, LhsAlignment>(i + 5, j), b0, c5);
            c6 = pcj.pmadd(lhs.template load<LhsPacket, LhsAlignment>(i + 6, j), b0, c6);
            c7 = pcj.pmadd(lhs.template load<LhsPacket, LhsAlignment>(i + 7, j), b0, c7);
        }
        ResScalar cc0 = predux(c0);
        ResScalar cc1 = predux(c1);
        ResScalar cc2 = predux(c2);
        ResScalar cc3 = predux(c3);
        ResScalar cc4 = predux(c4);
        ResScalar cc5 = predux(c5);
        ResScalar cc6 = predux(c6);
        ResScalar cc7 = predux(c7);
        for (; j < cols; ++j)
        {
            RhsScalar b0 = rhs(j, 0);

            cc0 += cj.pmul(lhs(i + 0, j), b0);
            cc1 += cj.pmul(lhs(i + 1, j), b0);
            cc2 += cj.pmul(lhs(i + 2, j), b0);
            cc3 += cj.pmul(lhs(i + 3, j), b0);
            cc4 += cj.pmul(lhs(i + 4, j), b0);
            cc5 += cj.pmul(lhs(i + 5, j), b0);
            cc6 += cj.pmul(lhs(i + 6, j), b0);
            cc7 += cj.pmul(lhs(i + 7, j), b0);
        }
        res[(i + 0) * resIncr] += alpha * cc0;
        res[(i + 1) * resIncr] += alpha * cc1;
        res[(i + 2) * resIncr] += alpha * cc2;
        res[(i + 3) * resIncr] += alpha * cc3;
        res[(i + 4) * resIncr] += alpha * cc4;
        res[(i + 5) * resIncr] += alpha * cc5;
        res[(i + 6) * resIncr] += alpha * cc6;
        res[(i + 7) * resIncr] += alpha * cc7;
    }
    for (; i < n4; i += 4)
    {
        ResPacket c0 = pset1<ResPacket>(ResScalar(0)), c1 = pset1<ResPacket>(ResScalar(0)),
                  c2 = pset1<ResPacket>(ResScalar(0)), c3 = pset1<ResPacket>(ResScalar(0));

        Index j = 0;
        for (; j + LhsPacketSize <= cols; j += LhsPacketSize)
        {
            RhsPacket b0 = rhs.template load<RhsPacket, Unaligned>(j, 0);

            c0 = pcj.pmadd(lhs.template load<LhsPacket, LhsAlignment>(i + 0, j), b0, c0);
            c1 = pcj.pmadd(lhs.template load<LhsPacket, LhsAlignment>(i + 1, j), b0, c1);
            c2 = pcj.pmadd(lhs.template load<LhsPacket, LhsAlignment>(i + 2, j), b0, c2);
            c3 = pcj.pmadd(lhs.template load<LhsPacket, LhsAlignment>(i + 3, j), b0, c3);
        }
        ResScalar cc0 = predux(c0);
        ResScalar cc1 = predux(c1);
        ResScalar cc2 = predux(c2);
        ResScalar cc3 = predux(c3);
        for (; j < cols; ++j)
        {
            RhsScalar b0 = rhs(j, 0);

            cc0 += cj.pmul(lhs(i + 0, j), b0);
            cc1 += cj.pmul(lhs(i + 1, j), b0);
            cc2 += cj.pmul(lhs(i + 2, j), b0);
            cc3 += cj.pmul(lhs(i + 3, j), b0);
        }
        res[(i + 0) * resIncr] += alpha * cc0;
        res[(i + 1) * resIncr] += alpha * cc1;
        res[(i + 2) * resIncr] += alpha * cc2;
        res[(i + 3) * resIncr] += alpha * cc3;
    }
    for (; i < n2; i += 2)
    {
        ResPacket c0 = pset1<ResPacket>(ResScalar(0)), c1 = pset1<ResPacket>(ResScalar(0));

        Index j = 0;
        for (; j + LhsPacketSize <= cols; j += LhsPacketSize)
        {
            RhsPacket b0 = rhs.template load<RhsPacket, Unaligned>(j, 0);

            c0 = pcj.pmadd(lhs.template load<LhsPacket, LhsAlignment>(i + 0, j), b0, c0);
            c1 = pcj.pmadd(lhs.template load<LhsPacket, LhsAlignment>(i + 1, j), b0, c1);
        }
        ResScalar cc0 = predux(c0);
        ResScalar cc1 = predux(c1);
        for (; j < cols; ++j)
        {
            RhsScalar b0 = rhs(j, 0);

            cc0 += cj.pmul(lhs(i + 0, j), b0);
            cc1 += cj.pmul(lhs(i + 1, j), b0);
        }
        res[(i + 0) * resIncr] += alpha * cc0;
        res[(i + 1) * resIncr] += alpha * cc1;
    }
    for (; i < rows; ++i)
    {
        ResPacket c0 = pset1<ResPacket>(ResScalar(0));
        Index j      = 0;
        for (; j + LhsPacketSize <= cols; j += LhsPacketSize)
        {
            RhsPacket b0 = rhs.template load<RhsPacket, Unaligned>(j, 0);
            c0           = pcj.pmadd(lhs.template load<LhsPacket, LhsAlignment>(i, j), b0, c0);
        }
        ResScalar cc0 = predux(c0);
        for (; j < cols; ++j)
        {
            cc0 += cj.pmul(lhs(i, j), rhs(j, 0));
        }
        res[i * resIncr] += alpha * cc0;
    }
}


template <int Side, int StorageOrder, bool BlasCompatible>
struct gemv_dense_selector2;

template <>
struct gemv_dense_selector2<OnTheRight, ColMajor, true>
{
    template <typename Lhs, typename Rhs, typename Dest>
    static inline void run(const Lhs& lhs, const Rhs& rhs, Dest& dest,
                           const typename Recursive::BaseScalar<Lhs>::type& alpha)
    {
        using BS = typename Recursive::BaseScalar<Lhs>::type;

        typedef typename Lhs::Scalar LhsScalar;
        typedef typename Rhs::Scalar RhsScalar;
        typedef typename Dest::Scalar ResScalar;
        //        typedef typename Dest::RealScalar RealScalar;

        typedef internal::blas_traits<Lhs> LhsBlasTraits;
        typedef typename LhsBlasTraits::DirectLinearAccessType ActualLhsType;
        typedef internal::blas_traits<Rhs> RhsBlasTraits;
        typedef typename RhsBlasTraits::DirectLinearAccessType ActualRhsType;

        typedef Map<Matrix<ResScalar, Dynamic, 1>,
                    EIGEN_PLAIN_ENUM_MIN(AlignedMax, internal::packet_traits<ResScalar>::size)>
            MappedDest;

        ActualLhsType actualLhs = LhsBlasTraits::extract(lhs);
        ActualRhsType actualRhs = RhsBlasTraits::extract(rhs);



        //        std::cout << "test: " << expand(LhsBlasTraits::extractScalarFactor(lhs)) << std::endl;
        //        std::cout << "test: " << expand(RhsBlasTraits::extractScalarFactor(rhs)) << std::endl;



        BS actualAlpha = alpha * Recursive::extractScale(LhsBlasTraits::extractScalarFactor(lhs)) *
                         Recursive::extractScale(RhsBlasTraits::extractScalarFactor(rhs));

        //        std::cout << actualAlpha << std::endl;
        //        BS actualAlpha = alpha;

        // make sure Dest is a compile-time vector type (bug 1166)
        typedef typename conditional<Dest::IsVectorAtCompileTime, Dest, typename Dest::ColXpr>::type ActualDest;

        enum
        {
            // FIXME find a way to allow an inner stride on the result if packet_traits<Scalar>::size==1
            // on, the other hand it is good for the cache to pack the vector anyways...
            EvalToDestAtCompileTime = (ActualDest::InnerStrideAtCompileTime == 1),
            ComplexByReal           = (NumTraits<LhsScalar>::IsComplex) && (!NumTraits<RhsScalar>::IsComplex),
            MightCannotUseDest      = (!EvalToDestAtCompileTime) || ComplexByReal
        };

        typedef const_blas_data_mapper<LhsScalar, Index, ColMajor> LhsMapper;
        typedef const_blas_data_mapper<RhsScalar, Index, RowMajor> RhsMapper;
        //        RhsScalar compatibleAlpha = get_factor<ResScalar, RhsScalar>::run(actualAlpha);
        BS compatibleAlpha = actualAlpha;

        if (!MightCannotUseDest)
        {
            // shortcut if we are sure to be able to use dest directly,
            // this ease the compiler to generate cleaner and more optimzized code for most common cases
            general_matrix_vector_product<
                Index, LhsScalar, LhsMapper, ColMajor, LhsBlasTraits::NeedToConjugate, RhsScalar, RhsMapper,
                RhsBlasTraits::NeedToConjugate>::run(actualLhs.rows(), actualLhs.cols(),
                                                     LhsMapper(actualLhs.data(), actualLhs.outerStride()),
                                                     RhsMapper(actualRhs.data(), actualRhs.innerStride()), dest.data(),
                                                     1, compatibleAlpha);
        }
        else
        {
            gemv_static_vector_if<ResScalar, ActualDest::SizeAtCompileTime, ActualDest::MaxSizeAtCompileTime,
                                  MightCannotUseDest>
                static_dest;

            const bool alphaIsCompatible = (!ComplexByReal) || (numext::imag(actualAlpha) == BS(0));
            const bool evalToDest        = EvalToDestAtCompileTime && alphaIsCompatible;

            static_assert(alphaIsCompatible, "invalid alpha");

            ei_declare_aligned_stack_constructed_variable(ResScalar, actualDestPtr, dest.size(),
                                                          evalToDest ? dest.data() : static_dest.data());

            if (!evalToDest)
            {
#ifdef EIGEN_DENSE_STORAGE_CTOR_PLUGIN
                Index size = dest.size();
                EIGEN_DENSE_STORAGE_CTOR_PLUGIN
#endif
                if (!alphaIsCompatible)
                {
                    MappedDest(actualDestPtr, dest.size()).setZero();
                    compatibleAlpha = BS(1);
                }
                else
                    MappedDest(actualDestPtr, dest.size()) = dest;
            }

            general_matrix_vector_product<
                Index, LhsScalar, LhsMapper, ColMajor, LhsBlasTraits::NeedToConjugate, RhsScalar, RhsMapper,
                RhsBlasTraits::NeedToConjugate>::run(actualLhs.rows(), actualLhs.cols(),
                                                     LhsMapper(actualLhs.data(), actualLhs.outerStride()),
                                                     RhsMapper(actualRhs.data(), actualRhs.innerStride()),
                                                     actualDestPtr, 1, compatibleAlpha);

            if (!evalToDest)
            {
                //                if (!alphaIsCompatible)
                //                    dest.matrix() += actualAlpha * MappedDest(actualDestPtr, dest.size());
                //                else
                dest = MappedDest(actualDestPtr, dest.size());
            }
        }
    }
};

template <typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols, typename Rhs>
struct generic_product_impl<Matrix<Recursive::MatrixScalar<_Scalar>, _Rows, _Cols, _Options, _MaxRows, _MaxCols>, Rhs,
                            DenseShape, DenseShape, GemvProduct>
    : generic_product_impl_base<
          Matrix<Recursive::MatrixScalar<_Scalar>, _Rows, _Cols, _Options, _MaxRows, _MaxCols>, Rhs,
          generic_product_impl<Matrix<Recursive::MatrixScalar<_Scalar>, _Rows, _Cols, _Options, _MaxRows, _MaxCols>,
                               Rhs, DenseShape, DenseShape, GemvProduct>>
{
    using Lhs = Matrix<Recursive::MatrixScalar<_Scalar>, _Rows, _Cols, _Options, _MaxRows, _MaxCols>;

    typedef typename nested_eval<Lhs, 1>::type LhsNested;
    typedef typename nested_eval<Rhs, 1>::type RhsNested;
    typedef typename Product<Lhs, Rhs>::Scalar Scalar;
    using BS = typename Recursive::BaseScalar<Lhs>::type;
    enum
    {
        Side = Lhs::IsVectorAtCompileTime ? OnTheLeft : OnTheRight
    };
    typedef typename internal::remove_all<
        typename internal::conditional<int(Side) == OnTheRight, LhsNested, RhsNested>::type>::type MatrixType;

    template <typename Dest>
    static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void scaleAndAddTo(Dest& dst, const Lhs& lhs, const Rhs& rhs,
                                                                    const BS& alpha)
    {
        LhsNested actual_lhs(lhs);
        RhsNested actual_rhs(rhs);
        internal::gemv_dense_selector2<Side, (int(MatrixType::Flags) & RowMajorBit) ? RowMajor : ColMajor,
                                       bool(internal::blas_traits<MatrixType>::HasUsableDirectAccess)>::run(actual_lhs,
                                                                                                            actual_rhs,
                                                                                                            dst, alpha);
    }
};


// This base class provides default implementations for evalTo, addTo, subTo, in terms of scaleAndAddTo
template <typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols, typename Rhs,
          typename Derived>
struct generic_product_impl_base<Matrix<Recursive::MatrixScalar<_Scalar>, _Rows, _Cols, _Options, _MaxRows, _MaxCols>,
                                 Rhs, Derived>
{
    using Lhs = Matrix<Recursive::MatrixScalar<_Scalar>, _Rows, _Cols, _Options, _MaxRows, _MaxCols>;
    typedef typename Product<Lhs, Rhs>::Scalar Scalar;
    using BS = typename Recursive::BaseScalar<Lhs>::type;

    template <typename Dst>
    static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void evalTo(Dst& dst, const Lhs& lhs, const Rhs& rhs)
    {
        dst.setZero();
        scaleAndAddTo(dst, lhs, rhs, BS(1));
    }

    template <typename Dst>
    static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void addTo(Dst& dst, const Lhs& lhs, const Rhs& rhs)
    {
        scaleAndAddTo(dst, lhs, rhs, BS(1));
    }

    template <typename Dst>
    static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void subTo(Dst& dst, const Lhs& lhs, const Rhs& rhs)
    {
        scaleAndAddTo(dst, lhs, rhs, BS(-1));
    }

    template <typename Dst>
    static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void scaleAndAddTo(Dst& dst, const Lhs& lhs, const Rhs& rhs,
                                                                    const BS& alpha)
    {
        Derived::scaleAndAddTo(dst, lhs, rhs, alpha);
    }
};

}  // namespace Eigen::internal
