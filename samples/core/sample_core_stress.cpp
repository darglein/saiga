/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

//#include "saiga/core/time/performanceMeasure.h"
//#include "saiga/core/util/crash.h"
//#include "saiga/core/math/random.h"
#include "saiga/core/Core.h"
#include "saiga/core/math/Eigen_Compile_Checker.h"
#include "saiga/core/util/Thread/omp.h"
#include "saiga/core/util/commandLineArguments.h"

#include "Eigen/Core"

#include <random>

using namespace Saiga;

template <int size, typename T>
void eigenHeatTest(int numThreads)
{
    std::cout << "Starting Stress Test: Matrix Multiplication." << std::endl;
    std::cout << "Threads: " << numThreads << std::endl;


    using MatrixType2 = Eigen::Matrix<T, size, size>;

    //    omp_set_dynamic(0);
    OMP::setNumThreads(numThreads);


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


int main(int argc, char* argv[])
{
    CLI::App app{"CPU Stress Test.", "core_stress_test"};

    // Define Options
    //  1. Create variables + their default values
    //  2. Use add_option to parse them from command line

    int threads     = 0;
    bool use_avx    = false;
    bool use_double = false;
    app.add_option("-t,--threads", threads)->required();
    app.add_flag("--avx", use_avx);
    app.add_flag("--double", use_double);

    CLI11_PARSE(app, argc, argv);

    if (threads == 0)
    {
        threads = OMP::getMaxThreads();
    }


    EigenHelper::EigenCompileFlags f;
    f.create<2935672>();
    std::cout << f << std::endl;

    if (use_avx)
    {
#ifndef EIGEN_VECTORIZE_AVX
        SAIGA_EXIT_ERROR("AVX not supported on your platform!");
#endif
        if (use_double)
        {
            std::cout << "AVX, double" << std::endl;
            eigenHeatTest<7 * 8, double>(threads);
        }
        else
        {
            std::cout << "AVX, float" << std::endl;
            eigenHeatTest<13 * 8, float>(threads);
        }
    }
    else
    {
        if (use_double)
        {
            std::cout << "SSE, double" << std::endl;
            eigenHeatTest<13 * 4, double>(threads);
        }
        else
        {
            std::cout << "SSE, float" << std::endl;
            eigenHeatTest<23 * 4, float>(threads);
        }
    }
    return 0;
}
