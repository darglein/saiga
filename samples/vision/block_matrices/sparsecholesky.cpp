/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */



#include "saiga/core/math/random.h"
#include "saiga/core/time/timer.h"
#include "saiga/vision/VisionIncludes.h"

#include "EigenRecursive/All.h"
//#include "saiga/vision/ba/BlockRecursiveBATemplates.h"


using Block  = Eigen::Matrix<double, 2, 2>;
using Vector = Eigen::Matrix<double, 2, 1>;

using namespace Eigen::Recursive;


namespace Saiga
{
template <typename MatrixScalar, typename VectorType>
Eigen::Matrix<VectorType, -1, 1> solveSparseLDLT(const Eigen::SparseMatrix<MatrixScalar, Eigen::RowMajor>& A,
                                                 const Eigen::Matrix<VectorType, -1, 1>& b)
{
    auto _Rows = A.rows();



    //    Eigen::Matrix<_Scalar, _Rows, _Cols> L;
    Eigen::SparseMatrix<MatrixScalar, Eigen::RowMajor> L(A.rows(), A.cols());
    Eigen::DiagonalMatrix<MatrixScalar, -1> D(_Rows);
    Eigen::DiagonalMatrix<MatrixScalar, -1> Dinv(_Rows);
    Eigen::Matrix<VectorType, -1, 1> x(_Rows), y(_Rows), z(_Rows);


    for (int i = 0; i < A.rows(); ++i)
    {
        // compute Dj
        MatrixScalar sumd = AdditiveNeutral<MatrixScalar>::get();

        typename Eigen::SparseMatrix<MatrixScalar, Eigen::RowMajor>::InnerIterator it(A, i);


        for (int j = 0; j < i; ++j)

        {
            SAIGA_ASSERT(it.col() >= j);
            MatrixScalar sum = AdditiveNeutral<MatrixScalar>::get();

            // dot product in L of row i with row j
            // but only until column j
            typename Eigen::SparseMatrix<MatrixScalar, Eigen::RowMajor>::InnerIterator Li(L, i);
            typename Eigen::SparseMatrix<MatrixScalar, Eigen::RowMajor>::InnerIterator Lj(L, j);
            while (Li && Lj && Li.col() < j && Lj.col() < j)
            {
                if (Li.col() == Lj.col())
                {
                    sum += Li.value() * D.diagonal()(Li.col()) * Eigen::Recursive::transpose(Lj.value());
                    ++Li;
                    ++Lj;
                }
                else if (Li.col() < Lj.col())
                {
                    ++Li;
                }
                else
                {
                    ++Lj;
                }
            }



            if (it.col() == j)
            {
                sum = it.value() - sum;
                ++it;
            }
            else
            {
                sum = -sum;
            }

            sum            = sum * Dinv.diagonal()(j);
            L.insert(i, j) = sum;
            sumd += sum * D.diagonal()(j) * Eigen::Recursive::transpose(sum);
        }
        L.insert(i, i) = MultiplicativeNeutral<MatrixScalar>::get();
        SAIGA_ASSERT(it && it.col() == i);
        D.diagonal()(i)    = it.value() - sumd;
        Dinv.diagonal()(i) = inverseCholesky(D.diagonal()(i));
    }



    // forward / backward substituion
    //    z = L.template triangularView<Eigen::Lower>().solve(b);
    z = forwardSubstituteDiagOne2(L, b);
    y = multDiagVector(Dinv, z);
    x = backwardSubstituteDiagOneTranspose2(L, y);
    //    x = L.transpose().template triangularView<Eigen::Upper>().solve(y);

    return x;
}


void testSparseBlockCholesky()
{
    cout << "testSparseBlockCholesky" << endl;
    using CompleteMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;
    using CompleteVector = Eigen::Matrix<double, -1, 1>;


    CompleteMatrix A(6, 6);
    CompleteVector x(6);
    CompleteVector b(6);
    b = CompleteVector::Random(6);

    Eigen::SparseMatrix<MatrixScalar<Block>, Eigen::RowMajor> bA(3, 3);
    Eigen::Matrix<MatrixScalar<Vector>, -1, 1> bx(3), bb(3);

    CompleteVector t         = CompleteVector::Random(6);
    auto Adense              = (t * t.transpose()).eval();
    Adense.block(2, 0, 2, 2) = Block::Zero();
    Adense.block(0, 2, 2, 2) = Block::Zero();
    Adense.diagonal() += CompleteVector::Ones(6);

    for (auto i = 0; i < Adense.rows(); ++i)
    {
        for (auto j = 0; j < Adense.cols(); ++j)
        {
            if (Adense(i, j) != 0) A.insert(i, j) = Adense(i, j);
        }
    }

    for (auto i = 0; i < bA.rows(); ++i)
    {
        for (auto j = 0; j < bA.cols(); ++j)
        {
            if ((i == 1 && j == 0) || (i == 0 && j == 1)) continue;
            bA.insert(i, j) = Adense.block(i * 2, j * 2, 2, 2);
        }
        bb(i) = b.segment(i * 2, 2);
    }
#if 0
    //        cout << Adense << endl << endl;
    cout << A.toDense() << endl << endl;
    cout << blockMatrixToMatrix(bA.toDense()) << endl << endl;
    cout << b << endl << endl;
    cout << blockVectorToVector(bb) << endl << endl;
#endif


    cout << "non zeros: " << A.nonZeros() << endl;

    {
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
        solver.compute(A);
        x = solver.solve(b);
        //        x = A.llt().solve(b);
        cout << "x " << x.transpose() << endl;
        cout << "error: " << (A * x - b).squaredNorm() << endl;
    }

    {
        x = solveSparseLDLT(A, b);
        cout << "x " << x.transpose() << endl;
        cout << "error: " << (A * x - b).squaredNorm() << endl;
    }

    {
        bx = solveSparseLDLT(bA, bb);
        x  = expand(bx);
        cout << "x " << x.transpose() << endl;
        cout << "error: " << (A * x - b).squaredNorm() << endl;
    }

    {
        SparseRecursiveLDLT<decltype(bA), decltype(bb)> ldlt;
        ldlt.compute(bA);
        bx = ldlt.solve(bb);

        x = expand(bx);
        cout << "x " << x.transpose() << endl;
        cout << "error: " << (A * x - b).squaredNorm() << endl;

        //        solveLDLT2(bA, bb);

        //        x = fixedBlockMatrixToMatrix(bx);
        //        cout << "x " << x.transpose() << endl;
        //        cout << "error: " << (A * x - b).squaredNorm() << endl;
    }
}


void perfTestSparseCholesky()
{
    cout << "perfTestSparseCholesky" << endl;

    Saiga::Random::setSeed(34534);
    const int bn = 4;
    const int bm = 4;

    int n = 500;
    int m = 500;

    int numNonZeroBlocks = 3;

    using Block  = Eigen::Matrix<double, bn, bm, Eigen::RowMajor>;
    using Vector = Eigen::Matrix<double, bn, 1>;


    using CompleteMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;
    using CompleteVector = Eigen::Matrix<double, -1, 1>;



    CompleteMatrix A(n * bn, m * bm);
    CompleteVector x(n * bn);
    CompleteVector b(n * bn);
    b = CompleteVector::Random(n * bn);

    Eigen::SparseMatrix<MatrixScalar<Block>, Eigen::RowMajor> bA(n, m);
    Eigen::Matrix<MatrixScalar<Vector>, -1, 1> bx(n), bb(n);


    std::vector<Eigen::Triplet<double>> data;
    std::vector<Eigen::Triplet<Block>> bdata;


    Eigen::DiagonalMatrix<MatrixScalar<Block>, -1> diag(n), invdiag(n);
    Eigen::Matrix<double, -1, -1> invdiag2;


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

    invdiag2 = expand(invdiag);


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

    // sanity checks
    SAIGA_ASSERT((expand(bA) - A.toDense()).norm() == 0);
    SAIGA_ASSERT((expand(bb) - b).norm() == 0);



    //    cout << A.toDense() << endl << endl;
    //    cout << expand(bA.toDense()) << endl << endl;



    //    cout << b.transpose() << endl << endl;
    //    cout << expand(bb).transpose() << endl << endl;


    cout << "non zeros: " << A.nonZeros() << " fillrate " << ((double)A.nonZeros() / (A.cols() * A.rows())) * 100 << "%"
         << endl
         << endl;

    {
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Lower> solver;
        {
            SAIGA_BLOCK_TIMER();
            solver.compute(A);
        }
        x = solver.solve(b);
        cout << "Eigen error: " << (A * x - b).squaredNorm() << endl << endl;
    }


#if 0

    {
        SAIGA_BLOCK_TIMER();
        Eigen::Matrix<double, -1, -1> adense = A.toDense();

        DenseLDLT<decltype(adense), decltype(b)> ldlt;
        ldlt.compute(adense);
        cout << expand(ldlt.L) << endl << endl;
        x = ldlt.solve(b);
    }
    cout << "my dense error: " << (A * x - b).squaredNorm() << endl << endl;

#endif
    {
        Eigen::Matrix<double, -1, -1> adense = A.toDense();
        Eigen::Matrix<MatrixScalar<Block>, -1, -1> bA(n, m);
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < m; ++j)
            {
                bA(i, j) = adense.block(i * bn, j * bm, bn, bm);
            }
            //            bb(i) = b.segment(i * bn, bn);
        }

        // sanity checks
        SAIGA_ASSERT((expand(bA) - A.toDense()).norm() == 0);
        SAIGA_ASSERT((expand(bb) - b).norm() == 0);

        DenseLDLT<decltype(bA)> ldlt3;
        {
            SAIGA_BLOCK_TIMER();
            ldlt3.compute(bA);
        }
        bx = ldlt3.solve(bb);
        x  = expand(bx);
        cout << "my recursive dense error: " << (A * x - b).squaredNorm() << endl << endl;
    }



#if 0

    {
        SAIGA_BLOCK_TIMER();
        x = solveSparseLDLT(A, b);
        //        cout << "x " << x.transpose() << endl;
    }
    cout << "my sparse error: " << (A * x - b).squaredNorm() << endl << endl;



    {
        SAIGA_BLOCK_TIMER();
        bx = solveSparseLDLT(bA, bb);
        x  = expand(bx);
//        cout << "x " << x.transpose() << endl;
        cout << "error: " << (A * x - b).squaredNorm() << endl << endl;
    }
#endif

    {
        SparseRecursiveLDLT<decltype(A), decltype(b)> ldlt;
        {
            SAIGA_BLOCK_TIMER();
            ldlt.compute(A);
        }
        x = ldlt.solve(b);
        cout << "My sparse error: " << (A * x - b).squaredNorm() << endl << endl;
    }

    {
        SparseRecursiveLDLT<decltype(bA), decltype(bb)> ldlt;
        {
            SAIGA_BLOCK_TIMER();
            ldlt.compute(bA);
        }
        bx = ldlt.solve(bb);
        x  = expand(bx);
        cout << "My recursive sparse error: " << (A * x - b).squaredNorm() << endl << endl;
    }
}

}  // namespace Saiga
