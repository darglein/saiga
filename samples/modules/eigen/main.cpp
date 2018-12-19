/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/eigen/eigen.h"
#include "saiga/eigen/lse.h"
#include "saiga/time/performanceMeasure.h"
#include "saiga/util/crash.h"
#include "saiga/util/random.h"

#include "Eigen/Sparse"

#include <random>

using namespace Saiga;

static void printVectorInstructions()
{
    cout << "Eigen Version: " << EIGEN_WORLD_VERSION << "." << EIGEN_MAJOR_VERSION << "." << EIGEN_MINOR_VERSION
         << endl;

    std::cout << "defined EIGEN Macros:" << std::endl;

#ifdef EIGEN_NO_DEBUG
    std::cout << "EIGEN_NO_DEBUG" << std::endl;
#else
    std::cout << "EIGEN_DEBUG" << std::endl;
#endif

#ifdef EIGEN_VECTORIZE_FMA
    std::cout << "EIGEN_VECTORIZE_FMA" << std::endl;
#endif
#ifdef EIGEN_VECTORIZE_SSE3
    std::cout << "EIGEN_VECTORIZE_SSE3" << std::endl;
#endif
#ifdef EIGEN_VECTORIZE_SSSE3
    std::cout << "EIGEN_VECTORIZE_SSSE3" << std::endl;
#endif
#ifdef EIGEN_VECTORIZE_SSE4_1
    std::cout << "EIGEN_VECTORIZE_SSE4_1" << std::endl;
#endif
#ifdef EIGEN_VECTORIZE_SSE4_2
    std::cout << "EIGEN_VECTORIZE_SSE4_2" << std::endl;
#endif
#ifdef EIGEN_VECTORIZE_AVX
    std::cout << "EIGEN_VECTORIZE_AVX" << std::endl;
#endif
#ifdef EIGEN_VECTORIZE_AVX2
    std::cout << "EIGEN_VECTORIZE_AVX2" << std::endl;
#endif

    std::cout << std::endl;
}

inline void empty() {}

struct EmptyOp
{
    void operator()() {}
};

inline void multMatrixVector(const Eigen::MatrixXf& M, const Eigen::VectorXf& x, Eigen::VectorXf& y)
{
    y += M * x;
}

template <typename MatrixType>
void randomMatrix(MatrixType& M)
{
    std::mt19937 engine(345345);
    std::uniform_real_distribution<float> dist(-1, 1);
    for (int i = 0; i < M.rows(); ++i)
    {
        for (int j = 0; j < M.cols(); ++j)
        {
            M(i, j) = dist(engine);
        }
    }
}

void eigenHeatTest()
{
    cout << "Starting Thermal Test: Matrix Multiplication" << endl;

    using MatrixType2 = Eigen::Matrix<float, 100, 100, Eigen::ColMajor>;

    MatrixType2 m1 = MatrixType2::Random();
    MatrixType2 m2 = MatrixType2::Identity();

    size_t limit = 100000000;

#pragma omp parallel for
    for (int i = 0; i < limit; ++i)
    {
        m2 += m1 * m2;
    }

    cout << "Done." << endl << m2 << endl;
}

template <typename _Scalar, int _Rows, int _Cols, int _Options = Eigen::ColMajor>
struct MatrixScalar
{
    using MatrixType = Eigen::Matrix<_Scalar, _Rows, _Cols, _Options>;

    MatrixType data;

    MatrixScalar() = default;
    MatrixScalar(_Scalar v)
    {
        SAIGA_ASSERT(v == 0);
        data.setZero();
    }

    MatrixScalar(const MatrixType& v) : data(v) {}
    MatrixScalar& operator=(const MatrixType& v)
    {
        data = v;
        return *this;
    }

    explicit operator MatrixType() const { return data; }

    MatrixScalar operator+(const MatrixScalar& other) const { return {data + other.data}; }
    MatrixScalar operator*(const MatrixScalar& other) const { return {data * other.data}; }

    MatrixType& get() { return data; }
    const MatrixType& get() const { return data; }
};


void matrixOfMatrix()
{
    Eigen::Matrix<double, 4, 4> m1 = Eigen::Matrix4d::Random();
    Eigen::Matrix<double, 4, 4> m2 = Eigen::Matrix4d::Random();

    Eigen::Matrix<double, 4, 4> result = m1 * m2;

    cout << "Result " << endl << result << endl;

    using test = MatrixScalar<double, 2, 2>;

    Eigen::Matrix<test, 2, 2> b1, b2, res2;
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            b1(i, j) = m1.block(i * 2, j * 2, 2, 2);
            b2(i, j) = m2.block(i * 2, j * 2, 2, 2);
        }
    }

    res2 = b1 + b2;
    res2 = b1 * b2;

    result.setZero();

    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            result.block(i * 2, j * 2, 2, 2) = res2(i, j).get();
        }
    }

    cout << "Result " << endl << result << endl;
}

void sparseTest()
{
    // Solution of the following block-structured linear system
    //
    // | U   W |   | da |   | ea |
    // | Wt  V | * | db | = | eb |
    //
    // , where
    // U and V are diagonal matrices, and W is sparse.
    // V is assumed to be much larger then U.
    // If U is larger the schur complement should be computed in the other direction.

    // ======================== Parameters ========================

    // size of U
    int n = 5;
    // size of V
    int m = 10;
    // maximum number of non-zero elements per row in W
    int maxElementsPerRow = 2;

    // ======================== Initialize ========================

    // Diagonal matrices U and V
    // All elements are positive!
    Eigen::DiagonalMatrix<double, -1> U(n);
    Eigen::DiagonalMatrix<double, -1> V(m);
    U.diagonal().setRandom();
    V.diagonal().setRandom();


    // Right hand side of the linear system
    Eigen::Matrix<double, -1, 1> ea(n);
    Eigen::Matrix<double, -1, 1> eb(m);
    ea.setRandom();
    eb.setRandom();

    Eigen::SparseMatrix<double, Eigen::RowMajor> W(n, m);
    W.reserve(n * maxElementsPerRow);
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < maxElementsPerRow; ++j)
        {
            W.insert(i, Random::uniformInt(0, m - 1)) = Random::sampleDouble(-5, 5);
        }
    }
    cout << "W" << endl << W << endl;

    // ========================================================================================================

    {
        // Schur complement solution

        Eigen::DiagonalMatrix<double, -1> Vinv(m);
        for (int i = 0; i < m; ++i)
        {
            Vinv.diagonal()(i) = 1.0 / V.diagonal()(i);
        }
        cout << "U " << endl << U.toDenseMatrix() << endl;
        cout << "V " << endl << V.toDenseMatrix() << endl;
        cout << "Vinv " << endl << Vinv.toDenseMatrix() << endl;



        Eigen::SparseMatrix<double, Eigen::RowMajor> Y(n, m);
        Y = W * Vinv;
        cout << "Y" << endl << Y << endl;

        Eigen::Matrix<double, -1, -1> S(n, n);
        S            = -Y * W.transpose();
        S.diagonal() = U.diagonal() + S.diagonal();

        cout << "S" << endl << S << endl;

        Eigen::Matrix<double, -1, 1> ej(n);
        ej = ea - Y * eb;
        cout << "ej" << endl << ej.transpose() << endl;

        Eigen::Matrix<double, -1, 1> deltaA(n);
        deltaA = S.lu().solve(ej);

        Eigen::Matrix<double, -1, 1> q = eb - W.transpose() * deltaA;
        Eigen::Matrix<double, -1, 1> deltaB(m);
        deltaB = Vinv * q;
        cout << "delta schur" << endl << deltaA.transpose() << " " << deltaB.transpose() << endl;
        //        cout << "deltaB" << endl << deltaB.transpose() << endl;
    }

    {
        // "dense solution"
        Eigen::Matrix<double, -1, -1> A(n + m, n + m);
        A.block(0, 0, n, n) = U.toDenseMatrix();
        A.block(n, n, m, m) = V.toDenseMatrix();
        A.block(0, n, n, m) = W.toDense();
        A.block(n, 0, m, n) = W.toDense().transpose();

        Eigen::Matrix<double, -1, 1> e(n + m);
        e.segment(0, n) = ea;
        e.segment(n, m) = eb;

        Eigen::Matrix<double, -1, 1> delta;
        delta = A.lu().solve(e);

        cout << "delta dense" << endl << delta.transpose() << endl;
    }
}

int main(int argc, char* argv[])
{
    printVectorInstructions();

    sparseTest();
    return 0;
    matrixOfMatrix();
    eigenHeatTest();
    return 0;
    //    Eigen::setNbThreads(1);


    catchSegFaults();

    const int N = 10000;

    //    using MatrixType = Eigen::MatrixXf;
    //    using VectorType = Eigen::VectorXf;

    using MatrixType = Eigen::Matrix<float, -1, -1, Eigen::ColMajor>;
    using VectorType = Eigen::Matrix<float, -1, 1>;

    MatrixType M(N, N);
    VectorType x(N);
    VectorType y(N);

    randomMatrix(M);
    randomMatrix(x);

    cout << "random check: " << M(0, 0) << " == " << -0.571635 << endl;
    measureFunction("multMatrixVector", 100, multMatrixVector, M, x, y);


    MatrixType Ms(200, 200);
    VectorType xs(200);
    randomMatrix(Ms);
    measureFunction("solveNullspaceSVD", 100, solveNullspaceSVD<MatrixType, VectorType>, Ms, xs);


    using MatrixType2 = Eigen::Matrix<float, 100, 100, Eigen::ColMajor>;
    using VectorType2 = Eigen::Matrix<float, 100, 1>;
    MatrixType2 Ms2;
    VectorType2 xs2;
    randomMatrix(Ms2);
    measureFunction("solveNullspaceSVD2", 100, solveNullspaceSVD<MatrixType2, VectorType2>, Ms2, xs2);
}
