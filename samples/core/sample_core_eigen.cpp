/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/math/random.h"
#include "saiga/core/time/performanceMeasure.h"
#include "saiga/core/util/crash.h"

#include "Eigen/Sparse"

#include <random>

using namespace Saiga;

static void printVectorInstructions()
{
    std::cout << "Eigen Version: " << EIGEN_WORLD_VERSION << "." << EIGEN_MAJOR_VERSION << "." << EIGEN_MINOR_VERSION
              << std::endl;

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
    std::cout << "Starting Thermal Test: Matrix Multiplication" << std::endl;

    using MatrixType2 = Eigen::Matrix<float, 100, 100, Eigen::ColMajor>;

    MatrixType2 m1 = MatrixType2::Random();
    MatrixType2 m2 = MatrixType2::Identity();

    long limit = 100000000;

#pragma omp parallel for
    for (long i = 0; i < limit; ++i)
    {
        m2 += m1 * m2;
    }

    std::cout << "Done." << std::endl << m2 << std::endl;
}

// View Assembly code
// 1. Go to the directory of the .o file.
//        cd saiga/build/samples/modules/eigen/CMakeFiles/modules_eigen.dir
// 2. Dumb assembly to file
//        objdump -d -M intel -S mv_vector.cpp.o > vector_asm.txt


void testMatrixVector2()
{
    const int size = 8;
    using T        = double;
    using Mat      = Eigen::Matrix<T, size, size>;
    using Vec      = Eigen::Matrix<T, size, 1>;



    Mat A;
    Vec y, x;



    //    Eigen::internal::product_type
    //    Eigen::internal::general_matrix_vector_product
    // asm("# Start y=A*x");
    y = (A * x).eval();
    // asm("# End y=A*x");

    std::cout << y << std::endl;
}

void testMatrixVector()
{
    std::cout << "testMatrixVector" << std::endl;

    testMatrixVector2();

    std::cout << "testMatrixVector done." << std::endl;
}



int main(int argc, char* argv[])
{
    printVectorInstructions();


    testMatrixVector();

    return 0;

    eigenHeatTest();
    return 0;
}
