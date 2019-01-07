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
#include "saiga/vision/recursiveMatrices/Inverse.h"
#include "saiga/vision/recursiveMatrices/NeutralElements.h"

#include "cholesky.h"

using Block  = Eigen::Matrix<double, 2, 2>;
using Vector = Eigen::Matrix<double, 2, 1>;

using namespace Saiga;



namespace Saiga
{
template <typename _Scalar, int _Rows>
Eigen::Matrix<_Scalar, _Rows, 1> solveSparseLDLT(const Eigen::SparseMatrix<_Scalar, Eigen::RowMajor>& A,
                                                 const Eigen::Matrix<_Scalar, _Rows, 1>& b)
{
    //    Eigen::Matrix<_Scalar, _Rows, _Cols> L;
    Eigen::SparseMatrix<_Scalar, Eigen::RowMajor> L(A.rows(), A.cols());
    Eigen::DiagonalMatrix<_Scalar, _Rows> D;
    Eigen::DiagonalMatrix<_Scalar, _Rows> Dinv;
    Eigen::Matrix<_Scalar, _Rows, 1> x, y, z;

    //    auto cols = A.cols();

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
                    sum += Li.value() * D.diagonal()(Li.col()) * Lj.value();
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

            sumd += sum * D.diagonal()(j) * sum;
        }
        SAIGA_ASSERT(it.col() == i);
        L.insert(i, i)  = MultiplicativeNeutral<_Scalar>::get();
        D.diagonal()(i) = it.value() - sumd;
        //        Dinv.diagonal()(i) = D.diagonal()(i).get().inverse();
        Dinv.diagonal()(i) = InverseSymmetric<_Scalar>::get(D.diagonal()(i));
    }


#if 0
    // compute L
    for (int i = 0; i < _Rows; i++)
    {
        // compute Dj
        _Scalar sumd = AdditiveNeutral<_Scalar>::get();

        for (int j = 0; j < i; ++j)
        {
            // compute all l's for this row

            _Scalar sum = AdditiveNeutral<_Scalar>::get();
            for (int k = 0; k < j; ++k)
            {
                sum += L(i, k) * L(j, k) * D.diagonal()(k);
            }
            L(i, j) = Dinv.diagonal()(j) * (A(i, j) - sum);
            L(j, i) = AdditiveNeutral<_Scalar>::get();
            sumd += L(i, j) * L(i, j) * D.diagonal()(j);
        }
        L(i, i)            = MultiplicativeNeutral<_Scalar>::get();
        D.diagonal()(i)    = A(i, i) - sumd;
        Dinv.diagonal()(i) = InverseSymmetric<_Scalar>::get(D.diagonal()(i));
    }
#endif

    //    cout << L << endl << endl;

    // forward / backward substituion
    z = L.template triangularView<Eigen::Lower>().solve(b);
    y = Dinv * z;
    x = L.transpose().template triangularView<Eigen::Upper>().solve(y);

    return x;
}


void testSparseBlockCholesky()
{
    cout << "testSparseBlockCholesky" << endl;
    using CompleteMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;
    using CompleteVector = Eigen::Matrix<double, 4, 1>;



    CompleteMatrix A(4, 4);
    CompleteVector x;
    CompleteVector b;


    CompleteVector t = CompleteVector::Random();
    auto Adense      = (t * t.transpose()).eval();
    Adense(1, 0)     = 0;
    Adense(0, 1)     = 0;

    for (auto i = 0; i < Adense.rows(); ++i)
    {
        for (auto j = 0; j < Adense.cols(); ++j)
        {
            if (Adense(i, j) != 0) A.insert(i, j) = Adense(i, j);
        }
    }
    b = CompleteVector::Random();
    A.diagonal() += CompleteVector::Ones();


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
        //        solveLDLT2(bA, bb);

        //        x = fixedBlockMatrixToMatrix(bx);
        //        cout << "x " << x.transpose() << endl;
        //        cout << "error: " << (A * x - b).squaredNorm() << endl;
    }
}

}  // namespace Saiga
