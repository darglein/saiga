/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#define EIGEN_DONT_PARALLELIZE



#include "saiga/core/time/timer.h"
#include "saiga/core/util/random.h"
#include "saiga/vision/recursiveMatrices/RecursiveMatrices.h"

#include "cholesky.h"

using Scalar = float;
const int bn = 2;
const int bm = 2;
using Block  = Eigen::Matrix<Scalar, bn, bm>;
using Vector = Eigen::Matrix<Scalar, bn, 1>;

using BlockVector = Eigen::Matrix<Saiga::MatrixScalar<Vector>, -1, 1>;

// SAIGA_RM_CREATE_RETURN(Saiga::MatrixScalar<Block>, Saiga::MatrixScalar<Vector>, Saiga::MatrixScalar<Vector>);
// SAIGA_RM_CREATE_SMV_ROW_MAJOR(BlockVector);



namespace Saiga
{
void testCG()
{
    cout << "perfTestCG" << endl;

    Eigen::setNbThreads(1);
    Saiga::Random::setSeed(34534);


    int n = 10;
    int m = 10;

    int numNonZeroBlocks = 1;



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


    Eigen::DiagonalMatrix<MatrixScalar<Block>, -1> diag(n);

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

        diag.diagonal()(i) = b;
    }


    // generate the rest
    for (int q = 0; q < n; ++q)
    {
        auto ind = Random::uniqueIndices(numNonZeroBlocks, m);

        for (auto j : ind)
        {
            auto i = q;
            if (i < j) continue;

            Block b = Block::Random();
            auto v  = to_triplets(b);
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
        RecursiveDiagonalPreconditioner<Scalar> P;
        Eigen::Index iters = 20;
        Scalar tol         = 1e-30;
        {
            SAIGA_BLOCK_TIMER();
            P.compute(A);
            Eigen::internal::conjugate_gradient(A, b, x, P, iters, tol);
        }
        cout << "error " << tol << " iterations " << iters << endl;
        cout << "Eigen cg error: " << (A * x - b).squaredNorm() << endl << endl;
    }

    {
        bx.setZero();
        RecursiveDiagonalPreconditioner<MatrixScalar<Block>> P;
        Eigen::Index iters = 20;
        Scalar tol         = 1e-30;
        {
            SAIGA_BLOCK_TIMER();
            P.compute(bA);
            recursive_conjugate_gradient([&](const Eigen::Matrix<MatrixScalar<Vector>, -1, 1>& v) { return bA * v; },
                                         bb, bx, P, iters, tol);
        }
        x = expand(bx);
        cout << "error " << tol << " iterations " << iters << endl;
        cout << "Eigen cg error: " << (A * x - b).squaredNorm() << endl << endl;
    }
}

}  // namespace Saiga
