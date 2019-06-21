/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

//#include "saiga/core/time/performanceMeasure.h"
//#include "saiga/core/util/crash.h"
//#include "saiga/core/math/random.h"
#include "saiga/core/Core.h"
#include "saiga/extra/eigen/eigen.h"
#include "saiga/extra/eigen/lse.h"

#include "Eigen/Core"

#include <omp.h>
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

template <int size, typename T>
void eigenHeatTest(int numThreads)
{
    std::cout << "Starting Thermal Test: Matrix Multiplication." << std::endl;
    std::cout << "Threads: " << numThreads << std::endl;


    using MatrixType2 = Eigen::Matrix<T, size, size>;

    //    omp_set_dynamic(0);
    omp_set_num_threads(numThreads);

    //    numThreads = 2;

#pragma omp parallel
    {
        MatrixType2 m1 = MatrixType2::Random();
        MatrixType2 m2 = MatrixType2::Identity();

        long limit = 10000000;

#pragma omp parallel for
        for (long i = 0; i < limit; ++i)
        {
            m2 += m1 * m2;
        }

        std::cout << "Done." << std::endl;
    }
}

void testMatrixVector();

int main(int argc, char* argv[])
{
    Saiga::CommandLineArguments cla;
    cla.arguments.push_back({"threads", 't', "Number of threads", "2", false});
    cla.arguments.push_back({"avx", 0, "Use Avx", "0", true});
    cla.arguments.push_back({"double", 0, "Use double precision", "0", true});
    cla.parse(argc, argv);
    auto numThreads = cla.getLong("threads");

    printVectorInstructions();

    if (cla.getFlag("avx"))
    {
#ifndef EIGEN_VECTORIZE_AVX
        SAIGA_EXIT_ERROR("AVX not supported on your platform!");
#endif
        if (cla.getFlag("double"))
        {
            std::cout << "AVX, double" << std::endl;
            eigenHeatTest<7 * 8, double>(numThreads);
        }
        else
        {
            std::cout << "AVX, float" << std::endl;
            eigenHeatTest<13 * 8, float>(numThreads);
        }
    }
    else
    {
        if (cla.getFlag("double"))
        {
            std::cout << "SSE, double" << std::endl;
            eigenHeatTest<13 * 4, double>(numThreads);
        }
        else
        {
            std::cout << "SSE, float" << std::endl;
            eigenHeatTest<23 * 4, float>(numThreads);
        }
    }
    return 0;
}
