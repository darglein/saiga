/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */



#include "saiga/time/timer.h"
#include "saiga/util/random.h"
#include "saiga/vision/BlockRecursiveBATemplates.h"
#include "saiga/vision/MatrixScalar.h"
#include "saiga/vision/VisionIncludes.h"
#include "saiga/vision/recursiveMatrices/ForwardBackwardSubs.h"
#include "saiga/vision/recursiveMatrices/Inverse.h"
#include "saiga/vision/recursiveMatrices/NeutralElements.h"
#include "saiga/vision/recursiveMatrices/Transpose.h"

#include "cholesky.h"

using Block  = Eigen::Matrix<double, 2, 2>;
using Vector = Eigen::Matrix<double, 2, 1>;

using namespace Saiga;



namespace Saiga
{
template <typename _Scalar, typename _Scalar2>
Eigen::Matrix<_Scalar2, -1, 1> solveSparseLDLT(const Eigen::SparseMatrix<_Scalar, Eigen::RowMajor>& A,
                                               const Eigen::Matrix<_Scalar2, -1, 1>& b)
{
    auto _Rows = A.rows();

    //    Eigen::Matrix<_Scalar, _Rows, _Cols> L;
    Eigen::SparseMatrix<_Scalar, Eigen::RowMajor> L(A.rows(), A.cols());
    Eigen::DiagonalMatrix<_Scalar, -1> D(_Rows);
    Eigen::DiagonalMatrix<_Scalar, -1> Dinv(_Rows);
    Eigen::Matrix<_Scalar2, -1, 1> x(_Rows), y(_Rows), z(_Rows);


    for (int i = 0; i < A.outerSize(); ++i)
    {
        // compute Dj
        _Scalar sumd = AdditiveNeutral<_Scalar>::get();

        typename Eigen::SparseMatrix<_Scalar, Eigen::RowMajor>::InnerIterator it(A, i);


        for (int j = 0; j < i; ++j)

        {
            _Scalar sum = AdditiveNeutral<_Scalar>::get();

            // dot product in L of row i with row j
            // but only until column j
            typename Eigen::SparseMatrix<_Scalar, Eigen::RowMajor>::InnerIterator Li(L, i);
            typename Eigen::SparseMatrix<_Scalar, Eigen::RowMajor>::InnerIterator Lj(L, j);
            while (Li && Lj && Li.col() < j && Lj.col() < j)
            {
                if (Li.col() == Lj.col())
                {
                    sum += Li.value() * D.diagonal()(Li.col()) * transpose(Lj.value());
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
            sum = sum * Dinv.diagonal()(j);

            L.insert(i, j) = sum;

            sumd += sum * D.diagonal()(j) * transpose(sum);
        }
        SAIGA_ASSERT(it.col() == i);
        L.insert(i, i)     = MultiplicativeNeutral<_Scalar>::get();
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
    return;
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
        //        x = solveSparseLDLT(A, b);
        cout << "x " << x.transpose() << endl;
        cout << "error: " << (A * x - b).squaredNorm() << endl;
    }

    {
        //        bx = solveSparseLDLT(bA, bb);
        x = blockVectorToVector(bx);
        cout << "x " << x.transpose() << endl;
        cout << "error: " << (A * x - b).squaredNorm() << endl;
    }

    {
        //        solveLDLT2(bA, bb);

        //        x = fixedBlockMatrixToMatrix(bx);
        //        cout << "x " << x.transpose() << endl;
        //        cout << "error: " << (A * x - b).squaredNorm() << endl;
    }
}

}  // namespace Saiga
