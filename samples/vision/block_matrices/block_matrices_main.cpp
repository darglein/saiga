/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/time/timer.h"
#include "saiga/core/util/random.h"
#include "saiga/vision/VisionIncludes.h"
#include "saiga/vision/recursiveMatrices/All.h"

#include "Eigen/Sparse"
#include "cholesky.h"

using namespace Saiga;
using namespace Eigen::Recursive;

void testMatrixMatrixOperations()
{
    using CompleteMatrix = Eigen::Matrix<double, 4, 4>;
    using Block          = Eigen::Matrix<double, 2, 2>;


    CompleteMatrix m1 = CompleteMatrix::Random();
    CompleteMatrix m2 = CompleteMatrix::Random();
    CompleteMatrix result;

    result = m1 * m2;
    cout << "Result Dense Matrix" << endl << result << endl;



    Eigen::Matrix<MatrixScalar<Block>, 2, 2> b1, b2, res2;
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            b1(i, j) = m1.block(i * 2, j * 2, 2, 2);
            b2(i, j) = m2.block(i * 2, j * 2, 2, 2);
        }
    }

    res2 = b1 + b2;
    res2 = b1 - b2;
    res2 -= b1;
    res2 += b1;
    res2   = b1 * b2;
    result = expand(res2);
    cout << "Result Block Matrix" << endl << result << endl;
}

using Block  = Eigen::Matrix<double, 2, 2>;
using Vector = Eigen::Matrix<double, 2, 1>;


template <typename BinaryOp>
struct Eigen::ScalarBinaryOpTraits<MatrixScalar<Block>, MatrixScalar<Vector>, BinaryOp>
{
    typedef MatrixScalar<Vector> ReturnType;
};

void testMatrixVectorOperations()
{
    using CompleteMatrix = Eigen::Matrix<double, 4, 4>;
    using CompleteVector = Eigen::Matrix<double, 4, 1>;

    CompleteMatrix m = CompleteMatrix::Random();
    CompleteVector x = CompleteVector::Random();
    CompleteVector result;

    result = m * x;
    cout << "Result Dense Matrix" << endl << result << endl;



    Eigen::Matrix<MatrixScalar<Block>, 2, 2> bm;
    Eigen::Matrix<MatrixScalar<Vector>, 2, 1> bx, res2;
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            bm(i, j) = m.block(i * 2, j * 2, 2, 2);
        }
        bx(i) = x.segment(i * 2, 2);
    }

    res2   = bm * bx;
    result = expand(res2);
    cout << "Result Block Matrix" << endl << result << endl;
}



template <typename SparseLhsType, typename DenseRhsType, typename DenseResType>
struct Eigen::internal::sparse_time_dense_product_impl<SparseLhsType, DenseRhsType, DenseResType, MatrixScalar<Vector>,
                                                       Eigen::ColMajor, true>
{
    typedef typename internal::remove_all<SparseLhsType>::type Lhs;
    typedef typename internal::remove_all<DenseRhsType>::type Rhs;
    typedef typename internal::remove_all<DenseResType>::type Res;
    typedef typename evaluator<Lhs>::InnerIterator LhsInnerIterator;
    using AlphaType = MatrixScalar<Vector>;
    static void run(const SparseLhsType& lhs, const DenseRhsType& rhs, DenseResType& res, const AlphaType& alpha)
    {
        evaluator<Lhs> lhsEval(lhs);
        for (Index c = 0; c < rhs.cols(); ++c)
        {
            for (Index j = 0; j < lhs.outerSize(); ++j)
            {
                for (LhsInnerIterator it(lhsEval, j); it; ++it)
                {
                    res.coeffRef(it.index(), c) += (it.value() * rhs.coeff(j, c));
                }
            }
        }
    }
};


void sparseMatrixVector()
{
    using CompleteMatrix = Eigen::Matrix<double, 4, 4>;
    using CompleteVector = Eigen::Matrix<double, 4, 1>;

    CompleteMatrix m = CompleteMatrix::Random();
    CompleteVector x = CompleteVector::Random();
    CompleteVector result;

    Eigen::Matrix<MatrixScalar<Block>, 2, 2> bm;
    Eigen::Matrix<MatrixScalar<Vector>, 2, 1> bx, res2;

    // test with a sparse block matrix
    Eigen::SparseMatrix<MatrixScalar<Block>> sbm(2, 2);
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            sbm.insert(i, j) = m.block(i * 2, j * 2, 2, 2);
        }
        bx(i) = x.segment(i * 2, 2);
    }

    res2   = sbm * bx;
    result = expand(res2);
    cout << "Result Sparse Block Matrix" << endl << result << endl;
}

using RectBlock = Eigen::Matrix<double, 3, 2>;

// a 3x2 matrix times a 2x1 vector is a 3x1 vector
using RectResult = Eigen::Matrix<double, 3, 1>;


template <typename BinaryOp>
struct Eigen::ScalarBinaryOpTraits<MatrixScalar<RectBlock>, MatrixScalar<Vector>, BinaryOp>
{
    typedef MatrixScalar<RectResult> ReturnType;
};


template <typename SparseLhsType, typename DenseRhsType, typename DenseResType>
struct Eigen::internal::sparse_time_dense_product_impl<SparseLhsType, DenseRhsType, DenseResType,
                                                       MatrixScalar<RectResult>, Eigen::ColMajor, true>
{
    typedef typename internal::remove_all<SparseLhsType>::type Lhs;
    typedef typename internal::remove_all<DenseRhsType>::type Rhs;
    typedef typename internal::remove_all<DenseResType>::type Res;
    typedef typename evaluator<Lhs>::InnerIterator LhsInnerIterator;
    using AlphaType = MatrixScalar<RectResult>;
    static void run(const SparseLhsType& lhs, const DenseRhsType& rhs, DenseResType& res, const AlphaType& alpha)
    {
        evaluator<Lhs> lhsEval(lhs);
        for (Index c = 0; c < rhs.cols(); ++c)
        {
            for (Index j = 0; j < lhs.outerSize(); ++j)
            {
                for (LhsInnerIterator it(lhsEval, j); it; ++it)
                {
                    res.coeffRef(it.index(), c) += (it.value() * rhs.coeff(j, c));
                }
            }
        }
    }
};


void sparseRectangularMatrixVector()
{
    cout << "testing sparse matrix times vector with rectangular inner blocks" << endl;
    using CompleteMatrix       = Eigen::Matrix<double, 6, 4>;
    using CompleteVector       = Eigen::Matrix<double, 4, 1>;
    using ResultCompleteVector = Eigen::Matrix<double, 6, 1>;

    CompleteMatrix m = CompleteMatrix::Random();
    CompleteVector x = CompleteVector::Random();
    ResultCompleteVector result;

    cout << "reference: " << endl << m * x << endl;

    Eigen::Matrix<MatrixScalar<RectBlock>, 2, 2> bm;
    Eigen::Matrix<MatrixScalar<Vector>, 2, 1> bx;
    Eigen::Matrix<MatrixScalar<RectResult>, 2, 1> res2;

    // test with a sparse block matrix
    Eigen::SparseMatrix<MatrixScalar<RectBlock>> sbm(2, 2);
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            sbm.insert(i, j) = m.block(i * 3, j * 2, 3, 2);
        }
        bx(i) = x.segment(i * 2, 2);
    }

    res2   = sbm * bx;
    result = expand(res2);
    cout << "Result Sparse Block Matrix" << endl << result << endl;
}

int main(int argc, char* argv[])
{
    //    testMatrixMatrixOperations();
    //    testMatrixVectorOperations();
    //    sparseMatrixVector();
    //    sparseRectangularMatrixVector();

    //    perfTestDenseCholesky();
    //    perfTestSparseCholesky();
    //    testCG();
    testBlockCholesky();
    //    testSparseBlockCholesky();
    return 0;
}
