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

#include "../Core.h"

#include <iostream>

template <typename Lhs, typename LhsDiag, typename Rhs>
void solveTriangularForwardLeftLowerCol(const Lhs& lhs, const LhsDiag& lhsDiag, Eigen::MatrixBase<Rhs>& other)
{
    //    std::cout << std::string("solveTriangularForwardLeftLowerCol 2") << std::endl;

    using Index       = int;
    using RScalar     = typename Rhs::Scalar;
    using LhsIterator = typename Lhs::InnerIterator;

    //    Rhs cpy = other.eval();

    for (Index col = 0; col < other.cols(); ++col)
    {
        for (Index i = 0; i < lhs.cols(); ++i)
        {
            RScalar& tmp = other.coeffRef(i, col);

            {
                auto& diag = lhsDiag[i];
                auto test  = Eigen::Recursive::forwardSubstituteDiagOne(diag.get(), tmp.get());
                tmp        = test;
                //                ++it;
            }

            // it points now to the diagonal element
            LhsIterator it(lhs, i);
            while (it && it.index() < i) ++it;
            eigen_assert(!it || it.index() != i);


            //            if (it && it.index() == i)


            //                if (it && it.index() == i)

            for (; it; ++it)
            {
                other.coeffRef(it.index(), col).get() -= it.value().get() * tmp.get();
            }
        }
    }

#if 0
    std::cout << cpy.rows() << "x" << cpy.cols() << std::endl;
    // Test if (Ax-b)==0
    auto lcpy       = lhs.toDense().eval();
    lcpy.diagonal() = lhsDiag;

    auto lcpyE = expand(lcpy);
    lcpyE.template triangularView<Eigen::Upper>().setZero();
    lcpyE.diagonal().setOnes();

    auto otherE = expand(other);
    auto cpyE   = expand(cpy);


    //    std::cout << lcpyE << std::endl << std::endl;
    //    std::cout << "result: " << otherE.transpose() << std::endl;

    double test = (lcpyE.template triangularView<Eigen::Lower>() * otherE - cpyE).squaredNorm();
    std::cout << "error solveTriangularForwardLeftLowerCol: " << test << std::endl;
    eigen_assert(test < 1e-10);
#endif
}


template <typename Lhs, typename LhsDiag, typename Rhs>
void solveTriangularBackwardsLeftLowerCol(const Lhs& lhs, const LhsDiag& lhsDiag, Eigen::MatrixBase<Rhs>& other)
{
    //    std::cout << std::string("solveTriangularBackwardsLeftLowerCol 2") << std::endl;

    using Index       = int;
    using RScalar     = typename Rhs::Scalar;
    using LhsIterator = typename Lhs::InnerIterator;

    //    Rhs cpy = other.eval();

    for (Index col = 0; col < other.cols(); ++col)
    {
        for (Index i = lhs.rows() - 1; i >= 0; --i)
        {
            RScalar tmp = other.coeff(i, col);
            LhsIterator it(lhs, i);
            while (it && it.index() < i) ++it;
            eigen_assert(!it || it.index() != i);

            for (; it; ++it)
            {
                tmp.get() -= it.value().get().transpose() * other.coeff(it.index(), col).get();
            }



            auto test = Eigen::Recursive::backwardSubstituteDiagOneTranspose(lhsDiag[i].get(), (tmp.get()).eval());
            other.coeffRef(i, col) = test;

            //            other.coeffRef(i, col) = tmp;
        }
    }

#if 0
    std::cout << cpy.rows() << "x" << cpy.cols() << std::endl;
    // Test if (Ax-b)==0
    auto lcpy       = lhs.toDense().eval();
    lcpy.diagonal() = lhsDiag;

    auto lcpyE = expand(lcpy);
    lcpyE.template triangularView<Eigen::Upper>().setZero();
    lcpyE.diagonal().setOnes();

    auto otherE = expand(other);
    auto cpyE   = expand(cpy);


    //    std::cout << lcpyE << std::endl << std::endl;
    //    std::cout << "result: " << otherE.transpose() << std::endl;

    double test = (lcpyE.template triangularView<Eigen::Lower>().transpose() * otherE - cpyE).squaredNorm();
    std::cout << "error solveTriangularBackwardsLeftLowerCol: " << test << std::endl;
    eigen_assert(test < 1e-10);
#endif
}

namespace Eigen
{
namespace internal
{
}  // end namespace internal

/** \ingroup SparseCholesky_Module
 * \brief A base class for direct sparse Cholesky factorizations
 *
 * This is a base class for LL^T and LDL^T Cholesky factorizations of sparse matrices that are
 * selfadjoint and positive definite. These factorizations allow for solving A.X = B where
 * X and B can be either dense or sparse.
 *
 * In order to reduce the fill-in, a symmetric permutation P is applied prior to the factorization
 * such that the factorized matrix is P A P^-1.
 *
 * \tparam Derived the type of the derived class, that is the actual factorization type.
 *
 */
template <typename Derived>
class RecursiveSimplicialCholesky3Base2 : public SparseSolverBase<Derived>
{
    typedef SparseSolverBase<Derived> Base;
    using Base::m_isInitialized;

   public:
    typedef typename internal::traits<Derived>::MatrixType MatrixType;
    typedef typename internal::traits<Derived>::OrderingType OrderingType;
    enum
    {
        UpLo = internal::traits<Derived>::UpLo
    };
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::RealScalar RealScalar;
    typedef typename MatrixType::StorageIndex StorageIndex;
    typedef SparseMatrix<Scalar, ColMajor, StorageIndex> CholMatrixType;
    typedef CholMatrixType const* ConstCholMatrixPtr;
    typedef Matrix<Scalar, Dynamic, 1> VectorType;
    typedef std::vector<DiagonalMatrix<typename Scalar::Scalar, Dynamic>> DiagonalType;
    typedef Matrix<StorageIndex, Dynamic, 1> VectorI;

    enum
    {
        ColsAtCompileTime    = MatrixType::ColsAtCompileTime,
        MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
    };

   public:
    using Base::derived;

    /** Default constructor */
    RecursiveSimplicialCholesky3Base2()
        : m_info(Success), m_factorizationIsOk(false), m_analysisIsOk(false), m_shiftOffset(0), m_shiftScale(1)
    {
    }

    explicit RecursiveSimplicialCholesky3Base2(const MatrixType& matrix)
        : m_info(Success), m_factorizationIsOk(false), m_analysisIsOk(false), m_shiftOffset(0), m_shiftScale(1)
    {
        derived().compute(matrix);
    }

    ~RecursiveSimplicialCholesky3Base2() {}

    Derived& derived() { return *static_cast<Derived*>(this); }
    const Derived& derived() const { return *static_cast<const Derived*>(this); }

    inline Index cols() const { return m_matrix.cols(); }
    inline Index rows() const { return m_matrix.rows(); }

    /** \brief Reports whether previous computation was successful.
     *
     * \returns \c Success if computation was successful,
     *          \c NumericalIssue if the matrix.appears to be negative.
     */
    ComputationInfo info() const
    {
        eigen_assert(m_isInitialized && "Decomposition is not initialized.");
        return m_info;
    }

    /** \returns the permutation P
     * \sa permutationPinv() */
    const PermutationMatrix<Dynamic, Dynamic, StorageIndex>& permutationP() const { return m_P; }

    /** \returns the inverse P^-1 of the permutation P
     * \sa permutationP() */
    const PermutationMatrix<Dynamic, Dynamic, StorageIndex>& permutationPinv() const { return m_Pinv; }

    /** Sets the shift parameters that will be used to adjust the diagonal coefficients during the numerical
     * factorization.
     *
     * During the numerical factorization, the diagonal coefficients are transformed by the following linear model:\n
     * \c d_ii = \a offset + \a scale * \c d_ii
     *
     * The default is the identity transformation with \a offset=0, and \a scale=1.
     *
     * \returns a reference to \c *this.
     */
    Derived& setShift(const RealScalar& offset, const RealScalar& scale = 1)
    {
        m_shiftOffset = offset;
        m_shiftScale  = scale;
        return derived();
    }

#ifndef EIGEN_PARSED_BY_DOXYGEN
    /** \internal */
    template <typename Stream>
    void dumpMemory(Stream& s)
    {
        int total = 0;
        s << "  L:        "
          << ((total += (m_matrix.cols() + 1) * sizeof(int) + m_matrix.nonZeros() * (sizeof(int) + sizeof(Scalar))) >>
              20)
          << "Mb"
          << "\n";
        s << "  diag:     " << ((total += m_diag.size() * sizeof(Scalar)) >> 20) << "Mb"
          << "\n";
        s << "  tree:     " << ((total += m_parent.size() * sizeof(int)) >> 20) << "Mb"
          << "\n";
        s << "  nonzeros: " << ((total += m_nonZerosPerCol.size() * sizeof(int)) >> 20) << "Mb"
          << "\n";
        s << "  perm:     " << ((total += m_P.size() * sizeof(int)) >> 20) << "Mb"
          << "\n";
        s << "  perm^-1:  " << ((total += m_Pinv.size() * sizeof(int)) >> 20) << "Mb"
          << "\n";
        s << "  TOTAL:    " << (total >> 20) << "Mb"
          << "\n";
    }

    /** \internal */
    template <typename Rhs, typename Dest>
    void _solve_impl(const MatrixBase<Rhs>& b, MatrixBase<Dest>& dest) const
    {
        eigen_assert(m_factorizationIsOk &&
                     "The decomposition is not in a valid state for solving, you must first call either compute() or "
                     "symbolic()/numeric()");
        eigen_assert(m_matrix.rows() == b.rows());


        if (m_info != Success) return;



#    if 1
        if (m_P.size() > 0)
            dest = m_P * b;
        else
            dest = b;
        //        std::cout << "solve impl" << std::endl;

        //        if (m_matrix.nonZeros() > 0)
        {
            //            derived().matrixL().solveInPlace(dest);
            solveTriangularForwardLeftLowerCol(m_matrix, m_diagL, dest);
        }


        //        std::cout << "solve L dest: " << expand(dest).transpose() << std::endl;

        //        multDiagVector2(m_diag_inv.asDiagonal(), dest);

        for (int k = 0; k < dest.rows(); ++k)
        {
            dest(k).get() = m_diag_inv[k] * dest(k).get();
        }

        //        std::cout << "solve diag inv: " << expand(dest).transpose() << std::endl;

        //        if (m_matrix.nonZeros() > 0)  // otherwise U==I
        {
            solveTriangularBackwardsLeftLowerCol(m_matrix, m_diagL, dest);
            //            derived().matrixU().solveInPlace(dest);
        }

        //        std::cout << "solve L2 dest: " << expand(dest).transpose() << std::endl;

//        std::cout << "x2: " << expand(dest).transpose() << std::endl;

//        std::cout << "solve done" << std::endl;
#    endif

#    if 0
        if (m_P.size() > 0)
            dest = m_P * b;
        else
            dest = b;

        // brute force solution

        // build L and add the diag elements
        auto denseL = m_matrix.toDense();
        for (int i = 0; i < denseL.rows(); ++i)
        {
            denseL(i, i) = m_diagL(i);
        }
        auto L = expand(denseL);
        L.template triangularView<Eigen::Upper>().setZero();
        L.diagonal().setOnes();

        //        const int block_size = 3;
        const int block_size = Rhs::Scalar::M::RowsAtCompileTime;

        Eigen::Matrix<double, -1, 1> dinv(L.rows());
        for (int i = 0; i < denseL.rows(); ++i)
        {
            dinv.segment<block_size>(i * block_size) = m_diag_inv[i].get().diagonal();
        }

        Eigen::Matrix<double, -1, -1> x = expand(dest);

        std::cout << L << std::endl << std::endl;
        //        std::cout << dinv.transpose() << std::endl;
        //        std::cout << x.transpose() << std::endl;
        //        std::cout << std::endl;
        L.template triangularView<Eigen::Lower>().solveInPlace(x);

        std::cout << "solve L dest: " << x.transpose() << std::endl;

        x.array() = dinv.array() * x.array();

        std::cout << "solve diag inv: " << x.transpose() << std::endl;

        L.template triangularView<Eigen::Lower>().transpose().solveInPlace(x);

        std::cout << "solve L2 dest: " << x.transpose() << std::endl;

        dest.setZero();
        for (int i = 0; i < dest.rows(); ++i)
        {
            //            dest[i].get() = x.block(i * 2, 1, 2, 1);
            dest[i].get() = x.block<block_size, 1>(i * block_size, 0);
            //            std::cout << i << std::endl << dest[i].get() << std::endl;
        }

        //        std::cout << "x1: " << x.transpose() << std::endl;
        //        std::cout << "x2: " << expand(dest).transpose() << std::endl;

//        exit(0);
#    endif

        if (m_P.size() > 0) dest = m_Pinv * dest;
    }

    template <typename Rhs, typename Dest>
    void _solve_impl(const SparseMatrixBase<Rhs>& b, SparseMatrixBase<Dest>& dest) const
    {
        internal::solve_sparse_through_dense_panels(derived(), b, dest);
    }

#endif  // EIGEN_PARSED_BY_DOXYGEN

   public:
    /** Computes the sparse Cholesky decomposition of \a matrix */
    template <bool DoLDLT>
    void compute(const MatrixType& matrix)
    {
        eigen_assert(matrix.rows() == matrix.cols());
        Index size = matrix.cols();
        CholMatrixType tmp(size, size);
        ConstCholMatrixPtr pmat;
        ordering(matrix, pmat, tmp);
        analyzePattern_preordered(*pmat, DoLDLT);
        factorize_preordered<DoLDLT>(*pmat);
    }

    template <bool DoLDLT>
    void factorize(const MatrixType& a)
    {
        eigen_assert(a.rows() == a.cols());
        Index size = a.cols();
        CholMatrixType tmp(size, size);
        ConstCholMatrixPtr pmat;

        if (m_P.size() == 0 && (UpLo & Upper) == Upper)
        {
            eigen_assert(0);
            // If there is no ordering, try to directly use the input matrix without any copy
            internal::simplicial_cholesky_grab_input<CholMatrixType, MatrixType>::run(a, pmat, tmp);
        }
        else
        {
            tmp.template selfadjointView<Upper>() = a.template selfadjointView<UpLo>().twistedBy(m_P);
            pmat                                  = &tmp;
        }

        factorize_preordered<DoLDLT>(*pmat);
    }

    template <bool DoLDLT>
    void factorize_preordered(const CholMatrixType& a);

    void analyzePattern(const MatrixType& a, bool doLDLT)
    {
        eigen_assert(a.rows() == a.cols());
        Index size = a.cols();
        CholMatrixType tmp(size, size);
        ConstCholMatrixPtr pmat;
        ordering(a, pmat, tmp);
        analyzePattern_preordered(*pmat, doLDLT);
    }
    void analyzePattern_preordered(const CholMatrixType& a, bool doLDLT);

    void ordering(const MatrixType& a, ConstCholMatrixPtr& pmat, CholMatrixType& ap);

    /** keeps off-diagonal entries; drops diagonal entries */
    struct keep_diag
    {
        inline bool operator()(const Index& row, const Index& col, const Scalar&) const { return row != col; }
    };

    mutable ComputationInfo m_info;
    bool m_factorizationIsOk;
    bool m_analysisIsOk;

    CholMatrixType m_matrix;
    VectorType m_diagL;

    //    VectorType m_diag;  // the diagonal coefficients (LDLT mode)
    //    VectorType m_diag_inv;

    DiagonalType m_diag, m_diag_inv;

    VectorI m_parent;  // elimination tree
    VectorI m_nonZerosPerCol;
    PermutationMatrix<Dynamic, Dynamic, StorageIndex> m_P;     // the permutation
    PermutationMatrix<Dynamic, Dynamic, StorageIndex> m_Pinv;  // the inverse permutation

    RealScalar m_shiftOffset;
    RealScalar m_shiftScale;

    Eigen::Recursive::DenseLDLT<typename CholMatrixType::Scalar::M> ldlt;
    Eigen::LDLT<typename CholMatrixType::Scalar::M> eldlt;
    std::vector<Scalar> rowCache;
    std::vector<StorageIndex> pattern;
    std::vector<StorageIndex> tags;
};

template <typename _MatrixType, int _UpLo = Lower, typename _Ordering = AMDOrdering<typename _MatrixType::StorageIndex>>
class RecursiveSimplicialLDLT2;
template <typename _MatrixType, int _UpLo = Lower, typename _Ordering = AMDOrdering<typename _MatrixType::StorageIndex>>
class RecursiveSimplicialCholesky3;

namespace internal
{
template <typename _MatrixType, int _UpLo, typename _Ordering>
struct traits<RecursiveSimplicialLDLT2<_MatrixType, _UpLo, _Ordering>>
{
    typedef _MatrixType MatrixType;
    typedef _Ordering OrderingType;
    enum
    {
        UpLo = _UpLo
    };
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::StorageIndex StorageIndex;
    typedef SparseMatrix<Scalar, ColMajor, StorageIndex> CholMatrixType;
    typedef TriangularView<const CholMatrixType, Eigen::UnitLower> MatrixL;
    typedef TriangularView<const typename CholMatrixType::AdjointReturnType, Eigen::UnitUpper> MatrixU;
    //    static inline MatrixL getL(const MatrixType& m) { return MatrixL(m); }
    //    static inline MatrixU getU(const MatrixType& m) { return MatrixU(m.adjoint()); }
    static inline MatrixL getL(const CholMatrixType& m) { return MatrixL(m); }
    static inline MatrixU getU(const CholMatrixType& m) { return MatrixU(m.adjoint()); }
};

template <typename _MatrixType, int _UpLo, typename _Ordering>
struct traits<RecursiveSimplicialCholesky3<_MatrixType, _UpLo, _Ordering>>
{
    typedef _MatrixType MatrixType;
    typedef _Ordering OrderingType;
    enum
    {
        UpLo = _UpLo
    };
};

}  // namespace internal

/** \ingroup SparseCholesky_Module
 * \class SimplicialLLT
 * \brief A direct sparse LLT Cholesky factorizations
 *
 * This class provides a LL^T Cholesky factorizations of sparse matrices that are
 * selfadjoint and positive definite. The factorization allows for solving A.X = B where
 * X and B can be either dense or sparse.
 *
 * In order to reduce the fill-in, a symmetric permutation P is applied prior to the factorization
 * such that the factorized matrix is P A P^-1.
 *
 * \tparam _MatrixType the type of the sparse matrix A, it must be a SparseMatrix<>
 * \tparam _UpLo the triangular part that will be used for the computations. It can be Lower
 *               or Upper. Default is Lower.
 * \tparam _Ordering The ordering method to use, either AMDOrdering<> or NaturalOrdering<>. Default is AMDOrdering<>
 *
 * \implsparsesolverconcept
 *
 * \sa class SimplicialLDLT2, class AMDOrdering, class NaturalOrdering
 */

template <typename _MatrixType, int _UpLo, typename _Ordering>
class RecursiveSimplicialLDLT2
    : public RecursiveSimplicialCholesky3Base2<RecursiveSimplicialLDLT2<_MatrixType, _UpLo, _Ordering>>
{
   public:
    typedef _MatrixType MatrixType;
    enum
    {
        UpLo = _UpLo
    };
    typedef RecursiveSimplicialCholesky3Base2<RecursiveSimplicialLDLT2> Base;
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::RealScalar RealScalar;
    typedef typename MatrixType::StorageIndex StorageIndex;
    typedef SparseMatrix<Scalar, ColMajor, StorageIndex> CholMatrixType;
    typedef Matrix<Scalar, Dynamic, 1> VectorType;
    typedef internal::traits<RecursiveSimplicialLDLT2> Traits;
    typedef typename Traits::MatrixL MatrixL;
    typedef typename Traits::MatrixU MatrixU;

   public:
    /** Default constructor */
    RecursiveSimplicialLDLT2() : Base() {}

    /** Constructs and performs the LLT factorization of \a matrix */
    explicit RecursiveSimplicialLDLT2(const MatrixType& matrix) : Base(matrix) {}

    /** \returns a vector expression of the diagonal D */
    inline const VectorType vectorD() const
    {
        eigen_assert(Base::m_factorizationIsOk && "Simplicial LDLT not factorized");
        return Base::m_diag;
    }
    /** \returns an expression of the factor L */
    inline const MatrixL matrixL() const
    {
        eigen_assert(Base::m_factorizationIsOk && "Simplicial LDLT not factorized");
        return Traits::getL(Base::m_matrix);
    }

    /** \returns an expression of the factor U (= L^*) */
    inline const MatrixU matrixU() const
    {
        eigen_assert(Base::m_factorizationIsOk && "Simplicial LDLT not factorized");
        return Traits::getU(Base::m_matrix);
    }

    /** Computes the sparse Cholesky decomposition of \a matrix */
    RecursiveSimplicialLDLT2& compute(const MatrixType& matrix)
    {
        Base::template compute<true>(matrix);
        return *this;
    }

    /** Performs a symbolic decomposition on the sparcity of \a matrix.
     *
     * This function is particularly useful when solving for several problems having the same structure.
     *
     * \sa factorize()
     */
    void analyzePattern(const MatrixType& a) { Base::analyzePattern(a, true); }

    /** Performs a numeric decomposition of \a matrix
     *
     * The given matrix must has the same sparcity than the matrix on which the symbolic decomposition has been
     * performed.
     *
     * \sa analyzePattern()
     */
    void factorize(const MatrixType& a) { Base::template factorize<true>(a); }

    /** \returns the determinant of the underlying matrix from the current factorization */
    Scalar determinant() const { return Base::m_diag.prod(); }
};

/** \deprecated use SimplicialLDLT2 or class SimplicialLLT
 * \ingroup SparseCholesky_Module
 * \class SimplicialCholesky3
 *
 * \sa class SimplicialLDLT2, class SimplicialLLT
 */
template <typename _MatrixType, int _UpLo, typename _Ordering>
class SimplicialCholesky3 : public RecursiveSimplicialCholesky3Base2<SimplicialCholesky3<_MatrixType, _UpLo, _Ordering>>
{
   public:
    typedef _MatrixType MatrixType;
    enum
    {
        UpLo = _UpLo
    };
    typedef RecursiveSimplicialCholesky3Base2<SimplicialCholesky3> Base;
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::RealScalar RealScalar;
    typedef typename MatrixType::StorageIndex StorageIndex;
    typedef SparseMatrix<Scalar, ColMajor, StorageIndex> CholMatrixType;
    typedef Matrix<Scalar, Dynamic, 1> VectorType;
    typedef internal::traits<SimplicialCholesky3> Traits;
    typedef internal::traits<RecursiveSimplicialLDLT2<MatrixType, UpLo>> LDLTTraits;
    typedef internal::traits<SimplicialLLT<MatrixType, UpLo>> LLTTraits;

   public:
    SimplicialCholesky3() : Base(), m_LDLT(true) {}

    explicit SimplicialCholesky3(const MatrixType& matrix) : Base(), m_LDLT(true) { compute(matrix); }

    SimplicialCholesky3& setMode(SimplicialCholeskyMode mode)
    {
        switch (mode)
        {
            case SimplicialCholeskyLLT:
                m_LDLT = false;
                break;
            case SimplicialCholeskyLDLT:
                m_LDLT = true;
                break;
            default:
                break;
        }

        return *this;
    }

    inline const VectorType vectorD() const
    {
        eigen_assert(Base::m_factorizationIsOk && "Simplicial Cholesky not factorized");
        return Base::m_diag;
    }
    inline const CholMatrixType rawMatrix() const
    {
        eigen_assert(Base::m_factorizationIsOk && "Simplicial Cholesky not factorized");
        return Base::m_matrix;
    }

    /** Computes the sparse Cholesky decomposition of \a matrix */
    SimplicialCholesky3& compute(const MatrixType& matrix)
    {
        if (m_LDLT)
            Base::template compute<true>(matrix);
        else
            Base::template compute<false>(matrix);
        return *this;
    }

    /** Performs a symbolic decomposition on the sparcity of \a matrix.
     *
     * This function is particularly useful when solving for several problems having the same structure.
     *
     * \sa factorize()
     */
    void analyzePattern(const MatrixType& a) { Base::analyzePattern(a, m_LDLT); }

    /** Performs a numeric decomposition of \a matrix
     *
     * The given matrix must has the same sparcity than the matrix on which the symbolic decomposition has been
     * performed.
     *
     * \sa analyzePattern()
     */
    void factorize(const MatrixType& a)
    {
        if (m_LDLT)
            Base::template factorize<true>(a);
        else
            Base::template factorize<false>(a);
    }

    /** \internal */
    template <typename Rhs, typename Dest>
    void _solve_impl(const MatrixBase<Rhs>& b, MatrixBase<Dest>& dest) const
    {
        eigen_assert(Base::m_factorizationIsOk &&
                     "The decomposition is not in a valid state for solving, you must first call either compute() or "
                     "symbolic()/numeric()");
        eigen_assert(Base::m_matrix.rows() == b.rows());

        eigen_assert(0);
#if 0
        if (Base::m_info != Success) return;

        if (Base::m_P.size() > 0)
            dest = Base::m_P * b;
        else
            dest = b;

        if (Base::m_matrix.nonZeros() > 0)  // otherwise L==I
        {
            if (m_LDLT)
                LDLTTraits::getL(Base::m_matrix).solveInPlace(dest);
            else
                LLTTraits::getL(Base::m_matrix).solveInPlace(dest);
        }

                if (Base::m_diag.size() > 0) dest = Base::m_diag.asDiagonal().inverse() * dest;

        if (Base::m_matrix.nonZeros() > 0)  // otherwise I==I
        {
            if (m_LDLT)
                LDLTTraits::getU(Base::m_matrix).solveInPlace(dest);
            else
                LLTTraits::getU(Base::m_matrix).solveInPlace(dest);
        }

        if (Base::m_P.size() > 0) dest = Base::m_Pinv * dest;
#endif
    }

    /** \internal */
    template <typename Rhs, typename Dest>
    void _solve_impl(const SparseMatrixBase<Rhs>& b, SparseMatrixBase<Dest>& dest) const
    {
#if 0
        internal::solve_sparse_through_dense_panels(*this, b, dest);
#endif
    }

    Scalar determinant() const
    {
        if (m_LDLT)
        {
            return Base::m_diag.prod();
        }
        else
        {
            Scalar detL = Diagonal<const CholMatrixType>(Base::m_matrix).prod();
            return numext::abs2(detL);
        }
    }

   protected:
    bool m_LDLT;
};

template <typename Derived>
void RecursiveSimplicialCholesky3Base2<Derived>::ordering(const MatrixType& a, ConstCholMatrixPtr& pmat,
                                                          CholMatrixType& ap)
{
    eigen_assert(a.rows() == a.cols());
    const Index size = a.rows();
    pmat             = &ap;
    // Note that ordering methods compute the inverse permutation
    if (!internal::is_same<OrderingType, NaturalOrdering<Index>>::value)
    {
        if (m_Pinv.size() > 0)
        {
            // The ordering is already set use it!
            //            std::cout << "Using preset permutation!" << std::endl;
            m_P = m_Pinv.inverse();
        }
        else
        {
            {
                CholMatrixType C;
                C = a.template selfadjointView<UpLo>();

                OrderingType ordering;
                ordering(C, m_Pinv);
            }

            if (m_Pinv.size() > 0)
                m_P = m_Pinv.inverse();
            else
                m_P.resize(0);
        }

        ap.resize(size, size);
        ap.template selfadjointView<Upper>() = a.template selfadjointView<UpLo>().twistedBy(m_P);
    }
    else
    {
        eigen_assert(0);
        m_Pinv.resize(0);
        m_P.resize(0);
        if (int(UpLo) == int(Lower) || MatrixType::IsRowMajor)
        {
            // we have to transpose the lower part to to the upper one
            ap.resize(size, size);
            ap.template selfadjointView<Upper>() = a.template selfadjointView<UpLo>();
        }
        else
            internal::simplicial_cholesky_grab_input<CholMatrixType, MatrixType>::run(a, pmat, ap);
    }
}

}  // end namespace Eigen



#include "RecursiveSimplicialCholesky_impl2.h"
