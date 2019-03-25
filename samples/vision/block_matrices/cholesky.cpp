/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */



#include "cholesky.h"

#include "saiga/core/time/timer.h"
#include "saiga/core/util/random.h"
#include "saiga/vision/VisionIncludes.h"
#include "saiga/vision/recursiveMatrices/BlockRecursiveBATemplates.h"
#include "saiga/vision/recursiveMatrices/Cholesky.h"
#include "saiga/vision/recursiveMatrices/Expand.h"
#include "saiga/vision/recursiveMatrices/ForwardBackwardSubs.h"
#include "saiga/vision/recursiveMatrices/Inverse.h"
#include "saiga/vision/recursiveMatrices/MatrixScalar.h"
#include "saiga/vision/recursiveMatrices/NeutralElements.h"
#include "saiga/vision/recursiveMatrices/Transpose.h"

using Block  = Eigen::Matrix<double, 2, 2>;
using Vector = Eigen::Matrix<double, 2, 1>;

using namespace Saiga;



namespace Saiga
{
template <typename _Scalar, int _Rows, int _Cols>
Eigen::Matrix<_Scalar, _Rows, 1> solveLLT(const Eigen::Matrix<_Scalar, _Rows, _Cols>& A,
                                          const Eigen::Matrix<_Scalar, _Rows, 1>& b)
{
    Eigen::Matrix<_Scalar, _Rows, _Cols> L;
    Eigen::Matrix<_Scalar, _Rows, 1> x, z;

    // compute L
    for (int i = 0; i < _Rows; i++)
    {
        // for all cols until diagonal
        for (int j = 0; j <= i; j++)
        {
            double s = 0;
            // dot product of row i with row j,
            // but only until col j
            // this requires the top left block to be computed
            for (int k = 0; k < j; k++)
            {
                s += L(i, k) * L(j, k);
            }
            s       = A(i, j) - s;
            L(i, j) = (i == j) ? sqrt(s) : (1.0 / L(j, j) * (s));
            L(j, i) = L(i, j);
        }
    }

    cout << "L" << endl << L << endl << endl;

    // forward / backward substituion
    z = L.template triangularView<Eigen::Lower>().solve(b);
    x = L.template triangularView<Eigen::Upper>().solve(z);

    return x;
}

inline double inverse(double d)
{
    return 1.0 / d;
}

template <typename _Scalar, int _Rows, int _Cols>
Eigen::Matrix<_Scalar, _Rows, 1> solveLDLT(const Eigen::Matrix<_Scalar, _Rows, _Cols>& A,
                                           const Eigen::Matrix<_Scalar, _Rows, 1>& b)
{
    Eigen::Matrix<_Scalar, _Rows, _Cols> L;
    Eigen::DiagonalMatrix<_Scalar, _Rows> D;
    Eigen::DiagonalMatrix<_Scalar, _Rows> Dinv;
    Eigen::Matrix<_Scalar, _Rows, 1> x, y, z;

    // compute L
    for (int i = 0; i < _Rows; i++)
    {
        // compute Dj
        _Scalar sumd = _Scalar(0);

        for (int j = 0; j < i; ++j)
        {
            // compute all l's for this row

            _Scalar sum = _Scalar(0);
            for (int k = 0; k < j; ++k)
            {
                sum += L(i, k) * L(j, k) * D.diagonal()(k);
            }
            L(i, j) = Dinv.diagonal()(j) * (A(i, j) - sum);
            L(j, i) = 0;
            sumd += L(i, j) * L(i, j) * D.diagonal()(j);
        }
        L(i, i)            = 1;
        D.diagonal()(i)    = A(i, i) - sumd;
        Dinv.diagonal()(i) = inverse(D.diagonal()(i));
    }

    //    cout << L << endl << endl;
    cout << "L" << endl << L << endl << endl;

    cout << "D" << endl << expand(D) << endl << endl;

    int asdf = 2;
    cout << "block" << endl << L.block(asdf, 0, asdf, asdf).eval() << endl;

    // forward / backward substituion
    z = L.template triangularView<Eigen::Lower>().solve(b);
    y = Dinv * z;
    x = L.transpose().template triangularView<Eigen::Upper>().solve(y);

    return x;
}



template <typename MatrixType, typename VectorType>
VectorType solveLDLT2(const MatrixType& A, const VectorType& b)
{
    SAIGA_ASSERT(A.rows() == A.cols() && A.rows() == b.rows());
    using MatrixScalar = typename MatrixType::Scalar;
    //    using VectorScalar = typename VectorType::Scalar;

    MatrixType L;
    L.resize(A.rows(), A.cols());

    Eigen::DiagonalMatrix<MatrixScalar, MatrixType::RowsAtCompileTime> D;
    Eigen::DiagonalMatrix<MatrixScalar, MatrixType::RowsAtCompileTime> Dinv;
    D.resize(A.rows());
    Dinv.resize(A.rows());

    VectorType x, y, z;
    x.resize(A.rows());
    y.resize(A.rows());
    z.resize(A.rows());


    // compute L
    for (int i = 0; i < A.rows(); i++)
    {
        // compute Dj
        MatrixScalar sumd = A(i, i);

        for (int j = 0; j < i; ++j)
        {
            // compute all l's for this row
            //            MatrixScalar sum = AdditiveNeutral<MatrixScalar>::get();
            MatrixScalar sum = A(i, j);

            // dot product of row i with row j
            // but only until column j
            for (int k = 0; k < j; ++k)
            {
                sum -= L(i, k) * D.diagonal()(k) * transpose(L(j, k));
            }

            L(i, j) = sum * Dinv.diagonal()(j);
            L(j, i) = AdditiveNeutral<MatrixScalar>::get();
            sumd -= L(i, j) * D.diagonal()(j) * transpose(L(i, j));
        }
        L(i, i)            = MultiplicativeNeutral<MatrixScalar>::get();
        D.diagonal()(i)    = sumd;
        Dinv.diagonal()(i) = inverse(D.diagonal()(i));
    }

    cout << "L" << endl << expand(L) << endl << endl;
    cout << "D" << endl << expand(D) << endl << endl;

    z = forwardSubstituteDiagOne(L, b);
    //    z = forwardSubstituteDiagOne2(L, b);
    y = multDiagVector(Dinv, z);
    x = backwardSubstituteDiagOneTranspose(L, y);


#if 0
    // Test if (Ax-b)==0
    double test =
        (fixedBlockMatrixToMatrix(A) * fixedBlockMatrixToMatrix(x) - fixedBlockMatrixToMatrix(b)).squaredNorm();
    cout << "error solveLDLT2: " << test << endl;
#endif


    return x;
}



template <typename MatrixType, typename VectorType>
VectorType solveLDLT_ybuffer(const MatrixType& A, const VectorType& b)
{
    SAIGA_ASSERT(A.rows() == A.cols() && A.rows() == b.rows());
    using MatrixScalar = typename MatrixType::Scalar;
    using BlockType    = double;
    using BlockVector  = double;
    //    using VectorScalar = typename VectorType::Scalar;

    MatrixType L;
    L.resize(A.rows(), A.cols());

    Eigen::DiagonalMatrix<MatrixScalar, MatrixType::RowsAtCompileTime> D;
    Eigen::DiagonalMatrix<MatrixScalar, MatrixType::RowsAtCompileTime> Dinv;
    D.resize(A.rows());
    Dinv.resize(A.rows());

    VectorType x, y, z;
    x.resize(A.rows());
    y.resize(A.rows());
    z.resize(A.rows());



    // A cache for the current row
    // This makes sense mostly for sparse matrices when stored in column major order
    std::vector<MatrixScalar> rowCache(A.rows());
    for (int i = 0; i < A.rows(); ++i) rowCache[i] = 0;

    MatrixScalar fakeYbuffer = 0;
    // compute L
    for (int k = 0; k < A.rows(); k++)
    {
        // Clear and load current row from the A matrix.
        std::fill(rowCache.begin(), rowCache.end(), AdditiveNeutral<MatrixScalar>::get());
        for (int i = 0; i <= k; ++i)
        {
            rowCache[i] = transpose(A(i, k));
        }

        // This is the diagonal element of the current row
        MatrixScalar sumd = rowCache[k];
        rowCache[k]       = Saiga::AdditiveNeutral<MatrixScalar>::get();



        for (int i = 0; i < k; ++i)
        {
            MatrixScalar yi = rowCache[i];
            rowCache[i]     = Saiga::AdditiveNeutral<MatrixScalar>::get();

            if (k == 0 && i == 0)
            {
                fakeYbuffer = yi;
            }


            // Compute y buffer for next column
            for (int j = i + 1; j < k; ++j)
            {
                rowCache[j] -= yi * transpose(L(j, i));
                //                cout << "subtract y: " << j << "," << i << endl;
            }

            auto& inv         = Dinv.diagonal()[i];
            MatrixScalar l_ki = yi * inv;
            L(k, i)           = l_ki;
            sumd -= yi * transpose(l_ki);


            //            if (k == 1 && i == 0)
            //            {
            //                //                fakeYbuffer -= yi * transpose(L(0, 0));
            //                cout << "yi" << endl << expand(yi) << endl << endl;
            //                cout << "test" << endl << expand(yi)(0,1) - (expand(yi)(0,0)) << endl << endl;
            //            }

            //            cout << expand(fakeYbuffer) << endl;
            //            cout << expand(yi * transpose(L(i, i))) << endl;
        }



        L(k, k)            = MultiplicativeNeutral<MatrixScalar>::get();
        D.diagonal()(k)    = sumd;
        Dinv.diagonal()(k) = inverse(D.diagonal()(k));
    }

    cout << "L" << endl << expand(L) << endl << endl;
    cout << "D" << endl << expand(D) << endl << endl;

    z = forwardSubstituteDiagOne(L, b);
    y = multDiagVector(Dinv, z);
    x = backwardSubstituteDiagOneTranspose(L, y);


#if 0
    // Test if (Ax-b)==0
    double test =
        (fixedBlockMatrixToMatrix(A) * fixedBlockMatrixToMatrix(x) - fixedBlockMatrixToMatrix(b)).squaredNorm();
    cout << "error solveLDLT2: " << test << endl;
#endif


    return x;
}

template <typename MatrixType, typename VectorType>
Eigen::Matrix<double, -1, 1> solveLDLT_ybuffer_block(const MatrixType& A, const VectorType& b)
{
    SAIGA_ASSERT(A.rows() == A.cols() && A.rows() == b.rows());
    using MatrixScalar = typename MatrixType::Scalar;
    using BlockType    = typename MatrixScalar::M;
    using BlockVector  = typename VectorType::Scalar::M;
    //    using VectorScalar = typename VectorType::Scalar;

    MatrixType L;
    L.resize(A.rows(), A.cols());

    Eigen::DiagonalMatrix<MatrixScalar, MatrixType::RowsAtCompileTime> D;
    Eigen::DiagonalMatrix<MatrixScalar, MatrixType::RowsAtCompileTime> Dinv;
    D.resize(A.rows());
    Dinv.resize(A.rows());

    VectorType x, y, z;
    x.resize(A.rows());
    y.resize(A.rows());
    z.resize(A.rows());



    std::vector<MatrixScalar> rowCache(A.rows());

    // compute L
    for (int k = 0; k < A.rows(); k++)
    {
        rowCache[k] = AdditiveNeutral<MatrixScalar>::get();

        for (int i = 0; i <= k; ++i)
        {
            rowCache[i] = transpose(A(i, k));
        }

        MatrixScalar diagElement = rowCache[k];
        rowCache[k]              = Saiga::AdditiveNeutral<MatrixScalar>::get();



        for (int i = 0; i < k; ++i)
        {
            // propate into rowcache
            auto& block   = rowCache[i].get();
            auto& LDiagUp = L(i, i).get();
            auto& inv     = Dinv.diagonal()[i].get();

            const int inner_block_size = MatrixScalar::M::RowsAtCompileTime;


            MatrixScalar prop;

            //            cout << "block before" << endl << expand(block) << endl;

            // Compute dense propagation on current block inplace
            for (int k = 0; k < inner_block_size; ++k)
            {
                for (int i = 0; i < inner_block_size; ++i)
                {
                    auto yi          = block(k, i);
                    prop.get()(k, i) = yi;

                    // propagate to the right in this row
                    for (int j = i + 1; j < inner_block_size; ++j)
                    {
                        block(k, j) -= yi * LDiagUp(j, i);
                    }

                    auto l_ki   = yi * inv(i, i);
                    block(k, i) = l_ki;
                }
            }
            L(k, i) = block;


            diagElement -= prop * transpose(block);

            for (int j = i + 1; j < k; ++j)
            {
                rowCache[j] -= prop * transpose(L(j, i));
            }

            //            cout << "block after" << endl << expand(block) << endl;

            continue;
#if 0
            for (int c = 1; c < 3; ++c)
            {
                rowCache[i].get()(0, c) -= rci(0, 0) * diagUp(c, 0);
            }


            MatrixScalar yi = rowCache[i];
            rowCache[i]     = Saiga::AdditiveNeutral<MatrixScalar>::get();



            // Compute y buffer for next column
            for (int j = i + 1; j < k; ++j)
            {
                rowCache[j] -= yi * transpose(L(j, i));
            }

            if (k == 1 && i == 0)
            {
                cout << "asdg" << endl << expand(yi) << endl;
                cout << "diag up" << endl << expand(L(i, i)) << endl;
            }


            MatrixScalar l_ki = yi * inv;
            L(k, i)           = l_ki;
            sumd -= yi * transpose(l_ki);
#endif
        }

#if 1
        Saiga::DenseLDLT<BlockType, BlockVector> ldlt;
        ldlt.compute(diagElement.get());

        L(k, k)            = ldlt.L.template triangularView<Eigen::Lower>();
        D.diagonal()(k)    = ldlt.D;
        Dinv.diagonal()(k) = ldlt.Dinv;
#else

        L(k, k)            = MultiplicativeNeutral<MatrixScalar>::get();
        D.diagonal()(k)    = sumd;
        Dinv.diagonal()(k) = inverse(D.diagonal()(k));
#endif
    }

    cout << "L" << endl << expand(L) << endl << endl;
    cout << "D" << endl << expand(D) << endl << endl;

#if 0
    z = forwardSubstituteDiagOne(L, b);
    //    z = forwardSubstituteDiagOne2(L, b);
    y = multDiagVector(Dinv, z);
    x = backwardSubstituteDiagOneTranspose(L, y);
#else

    Eigen::Matrix<double, -1, 1> x2, z2, y2;
    z2 = forwardSubstituteDiagOne(expand(L), expand(b));
    y2 = multDiagVector(expand(Dinv), z2);
    x2 = backwardSubstituteDiagOneTranspose(expand(L), expand(y2));
    return x2;
#endif


#if 0
    // Test if (Ax-b)==0
    double test =
        (fixedBlockMatrixToMatrix(A) * fixedBlockMatrixToMatrix(x) - fixedBlockMatrixToMatrix(b)).squaredNorm();
    cout << "error solveLDLT2: " << test << endl;
#endif


    //    return x;
}

template <typename T, typename D>
T ldltTriDot(const T& Asrc, const T& sum, const T& srcBlock, const D& srcDiag, const D& srcDiaginv);

template <>
double ldltTriDot<double, double>(const double& Asrc, const double& sum, const double& srcBlock, const double& srcDiag,
                                  const double& srcDiaginv)
{
    return (Asrc - sum) * srcDiaginv;
}


template <typename T, typename D>
T ldltTriDot(const T& Asrc, const T& sum, const T& _srcBlock, const D& _srcDiag, const D& _srcDiaginv)
{
    using Scalar = typename T::Scalar;

    auto& A = removeMatrixScalar(Asrc);

    auto& initialSum = removeMatrixScalar(sum);
    auto& srcBlock   = removeMatrixScalar(_srcBlock);
    auto& srcDiag    = removeMatrixScalar(_srcDiag);
    auto& srcDiaginv = removeMatrixScalar(_srcDiaginv);

    //    auto srcBlock     = L(j, j).get();
    //    auto& targetBlock = L(i, j).get();
    T _targetBlock;
    auto& targetBlock = removeMatrixScalar(_targetBlock);
    //    auto srcDiag      = D.diagonal()(j).get();
    //    auto srcDiaginv   = Dinv.diagonal()(j).get();
    //    auto Asrc         = A(i, j).get();
    for (int l = 0; l < targetBlock.rows(); ++l)
    {
        for (int m = 0; m < targetBlock.cols(); ++m)
        {
            Scalar sum = initialSum(l, m);
            for (int n = 0; n < m; ++n)
            {
                sum += targetBlock(l, n) * srcDiag.diagonal()(n) * transpose(srcBlock(m, n));
            }

            //            targetBlock(l, m) = (Asrc(l, m) - sum) * srcDiaginv.diagonal()(m);
            targetBlock(l, m) =
                ldltTriDot(A(l, m), sum, srcBlock(m, m), srcDiag.diagonal()(m), srcDiaginv.diagonal()(m));
        }
    }
    return targetBlock;
}



template <typename MatrixType, typename VectorType>
VectorType solveLDLT3(const MatrixType& A, const VectorType& b)
{
    SAIGA_ASSERT(A.rows() == A.cols() && A.rows() == b.rows());
    using MatrixScalar = typename MatrixType::Scalar;
    using BlockType    = typename MatrixScalar::M;
    using BlockVector  = typename VectorType::Scalar::M;
    //    using VectorScalar = typename VectorType::Scalar;

    MatrixType L;
    L.resize(A.rows(), A.cols());
    L.setZero();


    Eigen::DiagonalMatrix<MatrixScalar, MatrixType::RowsAtCompileTime> D;
    Eigen::DiagonalMatrix<MatrixScalar, MatrixType::RowsAtCompileTime> Dinv;
    D.resize(A.rows());
    Dinv.resize(A.rows());
    D.setZero();
    Dinv.setZero();

    VectorType x, y, z;
    x.resize(A.rows());
    y.resize(A.rows());
    z.resize(A.rows());


    // compute L
    for (int i = 0; i < A.rows(); i++)
    {
        // compute Dj
        MatrixScalar sumd = AdditiveNeutral<MatrixScalar>::get();

        for (int j = 0; j < i; ++j)
        {
            cout << "Compute L " << i << "," << j << endl;
            // compute all l's for this row
            MatrixScalar sum = AdditiveNeutral<MatrixScalar>::get();

            // dot product of row i with row j
            // but only until column j
            for (int k = 0; k < j; ++k)
            {
                sum += L(i, k) * D.diagonal()(k) * transpose(L(j, k));
            }

            L(i, j) = ldltTriDot(A(i, j), sum, L(j, j), D.diagonal()(j), Dinv.diagonal()(j));

            L(j, i) = AdditiveNeutral<MatrixScalar>::get();
            sumd += L(i, j) * D.diagonal()(j) * transpose(L(i, j));
        }


        cout << "Compute D " << i << "," << i << endl;
        BlockType diagBlock = (A(i, i) - sumd).get();

        //        Eigen::LDLT<BlockType> ldlt(diagBlock);

        Saiga::DenseLDLT<BlockType, BlockVector> ldlt;
        ldlt.compute(diagBlock);



        //        BlockVector d =
        //        BlockType Dblock;
        //        Dblock = ldlt.D;
        //        Dblock.setZero();
        //        Dblock.diagonal() = ldlt.vectorD();

        D.diagonal()(i)    = ldlt.D;
        Dinv.diagonal()(i) = ldlt.Dinv;
        //        Dinv.diagonal()(i) = inverse(D.diagonal()(i));

        //        L(i, i)         = ldlt.matrixL();
        L(i, i) = ldlt.L.template triangularView<Eigen::Lower>();
        //        L(i, i)            = MultiplicativeNeutral<MatrixScalar>::get();
    }

    cout << "L" << endl << expand(L) << endl << endl;
    cout << "D" << endl << expand(D) << endl << endl;


    z = forwardSubstituteDiagOne(L, b);
    //    z = forwardSubstituteDiagOne2(L, b);
    y = multDiagVector(Dinv, z);
    x = backwardSubstituteDiagOneTranspose(L, y);


#if 0
    // Test if (Ax-b)==0
    double test =
        (fixedBlockMatrixToMatrix(A) * fixedBlockMatrixToMatrix(x) - fixedBlockMatrixToMatrix(b)).squaredNorm();
    cout << "error solveLDLT2: " << test << endl;
#endif


    //    return x;
}


void testBlockCholesky()
{
    cout << "testBlockCholesky" << endl;

    const int n          = 4;
    const int block_size = 4;


    using CompleteMatrix = Eigen::Matrix<double, n * block_size, n * block_size>;
    using CompleteVector = Eigen::Matrix<double, n * block_size, 1>;

    using Block  = Eigen::Matrix<double, block_size, block_size>;
    using Vector = Eigen::Matrix<double, block_size, 1>;

    //    return;

    CompleteMatrix A;
    CompleteVector d;
#if 0
    // clang-format off
    A <<
            1, 0, 0, 0, 0, 0,
            4, 1, 0, 0, 0, 0,
            3, 4, 1, 0, 0, 0,
            2, 3, 4, 1, 0, 0,
            7, 4, 1, 2, 1, 0,
            5, 2, 3, 5, 9, 1;
    // clang-format on



    d << 4, 6, 3, 2, 1, 4;
#else
    A.setRandom();
    d.setRandom();
#endif

    Eigen::DiagonalMatrix<double, n * block_size> D;
    D.diagonal() = d;
    A            = A.triangularView<Eigen::Lower>() * D.toDenseMatrix() * A.triangularView<Eigen::Lower>().transpose();



    CompleteVector x;
    CompleteVector b;


    //    CompleteVector t = CompleteVector::Random();
    //    A                = t * t.transpose();
    b = CompleteVector::Random();
    A.diagonal() += CompleteVector::Ones();

    cout << A << endl << endl;

    Eigen::Matrix<MatrixScalar<Block>, n, n> bA;
    Eigen::Matrix<MatrixScalar<Vector>, n, 1> bx, bb;
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            bA(i, j) = A.block<block_size, block_size>(i * block_size, j * block_size);
        }
        bb(i) = b.segment<block_size>(i * block_size);
    }


#if 0
    {
        x = A.llt().solve(b);
        cout << "x " << x.transpose() << endl;
        cout << "error: " << (A * x - b).squaredNorm() << endl;
    }
    //    {
    //        x = solveLLT(A, b);
    //        cout << "x " << x.transpose() << endl;
    //        cout << "error: " << (A * x - b).squaredNorm() << endl;
    //    }


    {
        //        x = solveLDLT2(A, b);
        DenseLDLT<decltype(A), decltype(b)> ldlt;
        ldlt.compute(A);
        x = ldlt.solve(b);

        cout << "x " << x.transpose() << endl;
        cout << "error: " << (A * x - b).squaredNorm() << endl;
    }
#endif
    if (0)
    {
        x = solveLDLT(A, b);
        //        cout << "x " << x.transpose() << endl;
        cout << "error: " << (A * x - b).squaredNorm() << endl;
    }


    if (0)
    {
        bx = solveLDLT2(bA, bb);

        x = expand(bx);
        //        cout << "x " << x.transpose() << endl;
        cout << "error: " << (A * x - b).squaredNorm() << endl;
    }


    if (0)
    {
        using SMat = Eigen::SparseMatrix<MatrixScalar<Block>, Eigen::ColMajor>;
        using LDLT = Eigen::RecursiveSimplicialLDLT<SMat, Eigen::Lower>;

        SMat sm(n, n);
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                sm.insert(i, j) = bA(i, j);
            }
        }
        sm.makeCompressed();


        LDLT ldlt;
        ldlt.compute(sm);
        bx = ldlt.solve(bb);
        //        bx = solveLDLT2(bA, bb);

        SMat L = ldlt.matrixL();
        cout << expand(L) << endl << endl;
        x = expand(bx);
        //        cout << "x " << x.transpose() << endl;
        cout << "error: " << (A * x - b).squaredNorm() << endl;
    }
    if (1)
    {
        x = solveLDLT_ybuffer(A, b);

        //        cout << "x " << x.transpose() << endl;
        cout << "error: " << (A * x - b).squaredNorm() << endl;
    }

    if (1)
    {
        x = solveLDLT_ybuffer_block(bA, bb);

        //        x = expand(bx);
        //        cout << "x " << x.transpose() << endl;
        cout << "error: " << (A * x - b).squaredNorm() << endl;
    }

    if (0)
    {
        bx = solveLDLT3(bA, bb);

        x = expand(bx);
        //        cout << "x " << x.transpose() << endl;
        cout << "error: " << (A * x - b).squaredNorm() << endl;
    }
#if 0

    {
        x = solveLDLT3(A, b);

        //        cout << "x " << x.transpose() << endl;
        cout << "error: " << (A * x - b).squaredNorm() << endl;
    }
    {
        DenseLDLT<decltype(bA), decltype(bb)> ldlt;
        ldlt.compute(bA);
        bx = ldlt.solve(bb);


        x = expand(bx);
        cout << "x " << x.transpose() << endl;
        cout << "error: " << (A * x - b).squaredNorm() << endl;
    }
#endif
}

void perfTestDenseCholesky()
{
    cout << "perfTestDenseCholesky" << endl;

    const int bn = 4;
    const int bm = 4;

    int n = 300;
    int m = 300;

    using Block  = Eigen::Matrix<double, bn, bm, Eigen::RowMajor>;
    using Vector = Eigen::Matrix<double, bn, 1>;


    using CompleteMatrix = Eigen::Matrix<double, -1, -1, Eigen::RowMajor>;
    using CompleteVector = Eigen::Matrix<double, -1, 1>;



    CompleteMatrix A(n * bn, m * bm);
    CompleteVector x(n * bn);
    CompleteVector b(n * bn);


    CompleteVector t = CompleteVector::Random(n * bn);
    A                = t * t.transpose();
    b                = CompleteVector::Random(n * bn);
    A.diagonal() += CompleteVector::Ones(n * bn);

    Eigen::Matrix<MatrixScalar<Block>, -1, -1> bA(n, m);
    Eigen::Matrix<MatrixScalar<Vector>, -1, 1> bx(n), bb(n);
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            bA(i, j) = A.block(i * bn, j * bm, bn, bm);
        }
        bb(i) = b.segment(i * bn, bn);
    }

    // Solve Ax = b
    // 1. with eigen
    // 2. with my impl
    // 3. with my recursive impl

    Eigen::LDLT<CompleteMatrix, Eigen::Upper> ldlt;
    {
        SAIGA_BLOCK_TIMER();
        ldlt.compute(A);
        //        x = A.ldlt().solve(b);
    }
    x = ldlt.solve(b);
    cout << "Eigen error: " << (A * x - b).squaredNorm() << endl;


    DenseLDLT<decltype(A), decltype(b)> ldlt2;
    {
        SAIGA_BLOCK_TIMER();
        ldlt2.compute(A);
    }
    x = ldlt2.solve(b);
    cout << "My error: " << (A * x - b).squaredNorm() << endl;


    DenseLDLT<decltype(bA), decltype(bb)> ldlt3;
    {
        SAIGA_BLOCK_TIMER();
        ldlt3.compute(bA);
    }
    bx = ldlt3.solve(bb);
    x  = expand(bx);
    cout << "My recursive error: " << (A * x - b).squaredNorm() << endl;
}

}  // namespace Saiga
