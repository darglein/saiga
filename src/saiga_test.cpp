/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/config.h"
#include "saiga/core/Core.h"
#include "saiga/core/math/Eigen_Compile_Checker.h"
#include "saiga/core/util/ConsoleColor.h"
#include "saiga/core/util/table.h"

#ifdef EIGEN_VECTORIZE_AVX2
#    include <immintrin.h>
#endif

using namespace Saiga;

bool allTestsOk = true;

void printTestInfo(const std::string& name, bool success)
{
    Table table({2, 25, 15});
    table << "> " << name;

    if (success)
    {
        std::cout << ConsoleColor::GREEN;
        table << "SUCCESS";
    }
    else
    {
        std::cout << ConsoleColor::RED;
        table << "FAILED!";
        allTestsOk = false;
    }
    std::cout << ConsoleColor::RESET;
}


#ifdef EIGEN_VECTORIZE_AVX2
bool avx2Test()
{
    volatile __m256i first  = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
    volatile __m256i second = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
    volatile __m256i result = _mm256_add_epi32(first, second);
    volatile int* values    = (int*)&result;

    int sum = 0;
    for (int i = 0; i < 8; ++i)
    {
        sum += values[i];
    }
    return sum == 72;
}
#endif


struct Test
{
    double a[4];
};

int main()
{
    printSaigaInfo();
    auto saigaFlags = EigenHelper::getSaigaEigenCompileFlags();
    std::cout << saigaFlags << std::endl;


    std::cout << "Running a few basic tests..." << std::endl;
    printTestInfo("Find Saiga Shaders", findShaders(SaigaParameters()));

#ifdef EIGEN_VECTORIZE_AVX2
    if (saigaFlags.avx2)
    {
        printTestInfo("AVX2", avx2Test());
    }
#endif

    printTestInfo("Align vec3", alignof(vec3) == 4);
    printTestInfo("Align Vec3", alignof(Vec3) == 8);
    printTestInfo("Align Vec4", alignof(Vec4) == 32);
    printTestInfo("Align Quat", alignof(Quat) == 32);

    std::cout << std::endl;
    if (allTestsOk)
    {
        std::cout << "Everything OK. :)" << std::endl;
    }
    else
    {
        std::cout << "One or more tests have failed. :(" << std::endl;
    }



    return 0;
}
