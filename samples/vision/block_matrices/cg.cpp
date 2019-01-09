/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#define EIGEN_DONT_PARALLELIZE

#include "saiga/time/timer.h"
#include "saiga/util/random.h"
#include "saiga/vision/BlockRecursiveBATemplates.h"
#include "saiga/vision/MatrixScalar.h"
#include "saiga/vision/SparseHelper.h"
#include "saiga/vision/VisionIncludes.h"
#include "saiga/vision/recursiveMatrices/Cholesky.h"
#include "saiga/vision/recursiveMatrices/Dot.h"
#include "saiga/vision/recursiveMatrices/ForwardBackwardSubs.h"
#include "saiga/vision/recursiveMatrices/ForwardBackwardSubs_Sparse.h"
#include "saiga/vision/recursiveMatrices/Inverse.h"
#include "saiga/vision/recursiveMatrices/NeutralElements.h"
#include "saiga/vision/recursiveMatrices/Norm.h"
#include "saiga/vision/recursiveMatrices/ScalarMult.h"
#include "saiga/vision/recursiveMatrices/SparseCholesky.h"
#include "saiga/vision/recursiveMatrices/Transpose.h"

#include "cholesky.h"

using Scalar = float;
const int bn = 4;
const int bm = 4;
using Block  = Eigen::Matrix<Scalar, bn, bm>;
using Vector = Eigen::Matrix<Scalar, bn, 1>;

using namespace Saiga;



template <typename BinaryOp>
struct Eigen::ScalarBinaryOpTraits<MatrixScalar<Block>, MatrixScalar<Vector>, BinaryOp>
{
    typedef MatrixScalar<Vector> ReturnType;
};


template <typename SparseLhsType, typename DenseRhsType>
struct Eigen::internal::sparse_time_dense_product_impl<SparseLhsType, DenseRhsType,
                                                       Eigen::Matrix<MatrixScalar<Vector>, -1, 1>, MatrixScalar<Vector>,
                                                       Eigen::RowMajor, true>
{
    typedef typename internal::remove_all<SparseLhsType>::type Lhs;
    typedef typename internal::remove_all<DenseRhsType>::type Rhs;
    using DenseResType = DenseRhsType;
    typedef typename internal::remove_all<DenseResType>::type Res;
    typedef typename evaluator<Lhs>::InnerIterator LhsInnerIterator;
    typedef evaluator<Lhs> LhsEval;
    static void run(const SparseLhsType& lhs, const DenseRhsType& rhs, DenseResType& res,
                    const typename Res::Scalar& alpha)
    {
        LhsEval lhsEval(lhs);

        Index n = lhs.outerSize();


        for (Index c = 0; c < rhs.cols(); ++c)
        {
            {
                for (Index i = 0; i < n; ++i) processRow(lhsEval, rhs, res, i, c);
            }
        }
    }

    static void processRow(const LhsEval& lhsEval, const DenseRhsType& rhs, DenseResType& res, Index i, Index col)
    {
        typename Res::Scalar tmp(0);
        for (LhsInnerIterator it(lhsEval, i); it; ++it) tmp += it.value() * rhs.coeff(it.index(), col);
        res.coeffRef(i, col) += tmp;
    }
};


namespace Eigen
{
template <typename _Scalar>
class RecursiveDiagonalPreconditioner
{
    typedef _Scalar Scalar;
    typedef Matrix<Scalar, Dynamic, 1> Vector;

   public:
    typedef typename Vector::StorageIndex StorageIndex;
    enum
    {
        ColsAtCompileTime    = Dynamic,
        MaxColsAtCompileTime = Dynamic
    };

    RecursiveDiagonalPreconditioner() : m_isInitialized(false) {}

    template <typename MatType>
    explicit RecursiveDiagonalPreconditioner(const MatType& mat) : m_invdiag(mat.cols())
    {
        compute(mat);
    }

    Index rows() const { return m_invdiag.size(); }
    Index cols() const { return m_invdiag.size(); }

    template <typename MatType>
    RecursiveDiagonalPreconditioner& analyzePattern(const MatType&)
    {
        return *this;
    }

    template <typename MatType>
    RecursiveDiagonalPreconditioner& factorize(const MatType& mat)
    {
        m_invdiag.resize(mat.cols());
        for (int j = 0; j < mat.outerSize(); ++j)
        {
            typename MatType::InnerIterator it(mat, j);
            while (it && it.index() != j) ++it;
            if (it && it.index() == j)
                //          m_invdiag(j) = Scalar(1)/it.value();
                m_invdiag(j) = inverseCholesky(it.value());
            else
                //                m_invdiag(j) = Scalar(1);
                m_invdiag(j) = MultiplicativeNeutral<Scalar>::get();
        }
        m_isInitialized = true;
        return *this;
    }

    template <typename MatType>
    RecursiveDiagonalPreconditioner& compute(const MatType& mat)
    {
        return factorize(mat);
    }

    /** \internal */
    template <typename Rhs, typename Dest>
    void _solve_impl(const Rhs& b, Dest& x) const
    {
        x = m_invdiag.array() * b.array();
    }

    template <typename Rhs>
    inline const Solve<RecursiveDiagonalPreconditioner, Rhs> solve(const MatrixBase<Rhs>& b) const
    {
        eigen_assert(m_isInitialized && "DiagonalPreconditioner is not initialized.");
        eigen_assert(m_invdiag.size() == b.rows() &&
                     "DiagonalPreconditioner::solve(): invalid number of rows of the right hand side matrix b");
        return Solve<RecursiveDiagonalPreconditioner, Rhs>(*this, b.derived());
    }

    ComputationInfo info() { return Success; }

   protected:
    Vector m_invdiag;
    bool m_isInitialized;
};

template <typename MatrixType, typename Rhs, typename Dest, typename Preconditioner, typename SuperScalar>
EIGEN_DONT_INLINE void conjugate_gradient2(const MatrixType& mat, const Rhs& rhs, Dest& x,
                                           const Preconditioner& precond, Index& iters, SuperScalar& tol_error)
{
    using std::abs;
    using std::sqrt;
    typedef SuperScalar RealScalar;
    typedef SuperScalar Scalar;
    typedef Rhs VectorType;

    RealScalar tol = tol_error;
    Index maxIters = iters;

    Index n = mat.cols();

    VectorType residual = rhs - mat * x;  // initial residual

    RealScalar rhsNorm2 = squaredNorm(rhs);  //.squaredNorm();
    if (rhsNorm2 == 0)
    {
        x.setZero();
        iters     = 0;
        tol_error = 0;
        return;
    }
    RealScalar threshold     = tol * tol * rhsNorm2;
    RealScalar residualNorm2 = squaredNorm(residual);
    if (residualNorm2 < threshold)
    {
        iters     = 0;
        tol_error = sqrt(residualNorm2 / rhsNorm2);
        return;
    }

    VectorType p(n);
    p = precond.solve(residual);  // initial search direction

    VectorType z(n), tmp(n);
    //    RealScalar absNew = (residual.dot(p));  // the square of the absolute value of r scaled by invM
    RealScalar absNew = dot(residual, p);
    Index i           = 0;
    while (i < maxIters)
    {
        //        tmp.noalias() = mat * p;  // the bottleneck of the algorithm

        tmp          = mat * p;
        Scalar alpha = absNew / dot(p, tmp);  // the amount we travel on dir
                                              //        x += (p * alpha).eval();              // update solution
        x += scalarMult(p, alpha);
        //        residual -= (alpha * tmp).eval();  // update residual
        residual -= scalarMult(tmp, alpha);  // update residual

        residualNorm2 = squaredNorm(residual);
        if (residualNorm2 < threshold) break;

        z = precond.solve(residual);  // approximately solve for "A z = residual"

        RealScalar absOld = absNew;
        absNew            = dot(residual, z);  // update the absolute value of r
        RealScalar beta = absNew / absOld;  // calculate the Gram-Schmidt value used to create the new search direction
        p               = z + scalarMult(p, beta);  // update search direction
        i++;
    }
    tol_error = sqrt(residualNorm2 / rhsNorm2);
    iters     = i;
}

}  // namespace Eigen

namespace Saiga
{
void testCG()
{
    cout << "perfTestSparseCholesky" << endl;

    Eigen::setNbThreads(1);
    Saiga::Random::setSeed(34534);


    int n = 10000;
    int m = 10000;

    int numNonZeroBlocks = 200;



    using CompleteMatrix = Eigen::SparseMatrix<Scalar, Eigen::RowMajor>;
    using CompleteVector = Eigen::Matrix<Scalar, -1, 1>;



    CompleteMatrix A(n * bn, m * bm);
    CompleteVector x(n * bn);
    CompleteVector b(n * bn);
    b = CompleteVector::Random(n * bn);

    Eigen::SparseMatrix<MatrixScalar<Block>, Eigen::RowMajor> bA(n, m);
    Eigen::Matrix<MatrixScalar<Vector>, -1, 1> bx(n), bb(n);


    std::vector<Eigen::Triplet<Scalar>> data;
    data.reserve(numNonZeroBlocks * n * bn * bm);
    std::vector<Eigen::Triplet<Block>> bdata;
    bdata.reserve(numNonZeroBlocks * n);


    Eigen::DiagonalMatrix<MatrixScalar<Block>, -1> diag(n), invdiag(n);
    //    Eigen::Matrix<double, -1, -1> invdiag2;


    // generate diagonal blocks
    for (int i = 0; i < n; ++i)
    {
        Vector t = Vector::Random() * 5;
        Block b  = t * t.transpose();
        b.diagonal() += Vector::Ones() * 100;
        auto v = to_triplets(b);
        addOffsetToTriplets(v, i * bn, i * bn);
        data.insert(data.end(), v.begin(), v.end());
        bdata.emplace_back(i, i, b);

        diag.diagonal()(i)    = b;
        invdiag.diagonal()(i) = b.inverse();
    }

    //    invdiag2 = expand(invdiag);


    // generate the rest
    for (int q = 0; q < n; ++q)
    {
        auto ind = Random::uniqueIndices(numNonZeroBlocks, m);

        for (auto j : ind)
        {
            auto i = q;
            if (i < j) continue;

            Block b = Block::Random();
            //            Vector t = Vector::Random();
            //            Vector t = Vector::Ones();
            //            Block b = t * t.transpose();

            auto v = to_triplets(b);
            addOffsetToTriplets(v, i * bn, j * bm);
            data.insert(data.end(), v.begin(), v.end());
            bdata.emplace_back(i, j, b);

            b.transposeInPlace();
            v = to_triplets(b);
            addOffsetToTriplets(v, j * bn, i * bm);
            data.insert(data.end(), v.begin(), v.end());
            bdata.emplace_back(j, i, b);
        }
    }

    for (auto i = 0; i < bA.rows(); ++i)
    {
        bb(i) = b.segment(i * bn, bn);
    }


    A.setFromTriplets(data.begin(), data.end());
    bA.setFromTriplets(bdata.begin(), bdata.end());

    cout << "matrix constructed" << endl;

    // sanity checks
    //    SAIGA_ASSERT((expand(bA) - A.toDense()).norm() == 0);
    //    SAIGA_ASSERT((expand(bb) - b).norm() == 0);

#if 0

    {
        x.setZero();
        //        Eigen::ConjugateGradient
        Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, 0> solver;
        {
            SAIGA_BLOCK_TIMER();
            solver.compute(A);
            x = solver.solve(b);
        }

        cout << "error " << solver.error() << " iterations " << solver.iterations() << endl;
        cout << "Eigen cg error: " << (A * x - b).squaredNorm() << endl << endl;
    }

    {
        x.setZero();
        //        Eigen::ConjugateGradient
        Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower,
                                 Eigen::RecursiveDiagonalPreconditioner<double>>
            solver;
        x = AdditiveNeutral<decltype(x)>::get(x.rows(), x.cols());
        {
            SAIGA_BLOCK_TIMER();
            solver.compute(A);
            x = solver.solve(b);
        }

        cout << "error " << solver.error() << " iterations " << solver.iterations() << endl;
        cout << "Eigen cg error: " << (A * x - b).squaredNorm() << endl << endl;
    }
#endif
    {
        x.setZero();
        //        Eigen::ConjugateGradient
        Eigen::RecursiveDiagonalPreconditioner<Scalar> P;
        Eigen::Index iters = 20;
        Scalar tol         = 1e-30;
        {
            SAIGA_BLOCK_TIMER();
            P.compute(A);
            //            Eigen::conjugate_gradient2(A, b, x, P, iters, tol);
            Eigen::internal::conjugate_gradient(A, b, x, P, iters, tol);
        }
        cout << "error " << tol << " iterations " << iters << endl;
        cout << "Eigen cg error: " << (A * x - b).squaredNorm() << endl << endl;
    }

    {
        bx.setZero();
        //        Eigen::ConjugateGradient
        Eigen::RecursiveDiagonalPreconditioner<MatrixScalar<Block>> P;
        Eigen::Index iters = 20;
        Scalar tol         = 1e-30;
        {
            SAIGA_BLOCK_TIMER();
            P.compute(bA);
            Eigen::conjugate_gradient2(bA, bb, bx, P, iters, tol);
        }
        x = expand(bx);
        cout << "error " << tol << " iterations " << iters << endl;
        cout << "Eigen cg error: " << (A * x - b).squaredNorm() << endl << endl;
    }
}

}  // namespace Saiga
