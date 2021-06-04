/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/math/floatingPoint.h"
#include "saiga/core/math/math.h"
#include "saiga/core/util/assert.h"

#include "internal/noGraphicsAPI.h"

#include "test.h"

#include <iostream>
#include <random>
namespace Saiga
{
namespace Tests
{
using namespace std;

// https://stackoverflow.com/questions/4163126/dereferencing-type-punned-pointer-will-break-strict-aliasing-rules-warning
template <typename T, typename F>
struct alias_cast_t
{
    static_assert(sizeof(T) == sizeof(F), "Cannot cast types of different sizes");
    union {
        F raw;
        T data;
    };
};

template <typename T, typename F>
T alias_cast(F raw_data)
{
    alias_cast_t<T, F> ac;
    ac.raw = raw_data;
    return ac.data;
}

static unsigned int fToUint(float f)
{
    unsigned int i;
    //    i = *reinterpret_cast<unsigned int*>(&f);
    i = alias_cast<unsigned int>(f);
    return i;
}

#if 0
static float uintToF(unsigned int i){
    float f;
    //    f = *reinterpret_cast<float*>(&i);
    f = alias_cast<float>(i);
    return f;
}
#endif

// https://stackoverflow.com/questions/7295861/enabling-strict-floating-point-mode-in-gcc
// Compiling with -msse2 on an Intel/AMD processor that supports it will get you almost there.
// Do not let any library put the FPU in FTZ/DNZ mode, and you will be mostly set (processor bugs notwithstanding).

void fpTest(float x)
{
    SAIGA_ASSERT(FP::checkSSECSR());

    // make sure x = 1;
    SAIGA_ASSERT(fToUint(x) == 0x3f800000);


    FP::printCPUInfo();

    {
        // add test
        float a          = 0.1f;
        float b          = 0.2f;
        float erg        = a + b + x;
        unsigned int ref = 0x3fa66666;
        std::cout << "Addition test: " << (fToUint(erg) == ref ? "Success" : "Fail") << std::endl;
        //        std::cout << erg << " " << hex << fToUint(erg) << std::endl;
    }

    {
        // multiply test
        float f          = 0.1f;
        float erg        = f * f * x;
        unsigned int ref = 0x3c23d70b;
        std::cout << "Multiply test: " << (fToUint(erg) == ref ? "Success" : "Fail") << std::endl;
        //        std::cout << erg << " " << hex << fToUint(erg) << std::endl;
    }

    {
        // multiply test
        float a          = 3.0f;
        float b          = 7.0f;
        float erg        = a / b * x;
        unsigned int ref = 0x3edb6db7;
        std::cout << "Division test: " << (fToUint(erg) == ref ? "Success" : "Fail") << std::endl;
        //        std::cout << erg << " " << hex << fToUint(erg) << std::endl;
    }

    {
        // sin test
        const int N    = 9;
        float input[N] = {0.1f, 0.2f, 0.3f, 1.0f, 2.0f, 3.0f, pi<float>(), pi<float>() / 2.0f, pi<float>() / 4.0f};
        float output[N];
        for (int i = 0; i < N; ++i)
        {
            output[i] = sinf(input[i]);
        }
        unsigned int ref[N] = {0x3dcc7577, 0x3e4b6ff9, 0x3e974e6d, 0x3f576aa4, 0x3f68c7b7,
                               0x3e1081c3, 0xb3bbbd2e, 0x3f800000, 0x3f3504f3};
        bool success        = true;
        for (int i = 0; i < N; ++i)
        {
            if (fToUint(output[i]) != ref[i]) success = false;
            //            std::cout << hex << "0x" << fToUint(output[i]) << ", " ;
        }
        std::cout << "sinf(x) test: " << (success ? "Success" : "Fail") << std::endl;
    }

    {
        // sqrt test
        const int N    = 9;
        float input[N] = {0.1f, 0.2f, 0.3f, 1.0f, 2.0f, 3.0f, 10.0f, 100.0f, 1337.0f};
        float output[N];
        for (int i = 0; i < N; ++i)
        {
            output[i] = sqrt(input[i]);
        }
        unsigned int ref[N] = {0x3ea1e89b, 0x3ee4f92e, 0x3f0c378c, 0x3f800000, 0x3fb504f3,
                               0x3fddb3d7, 0x404a62c2, 0x41200000, 0x42124292};
        bool success        = true;
        for (int i = 0; i < N; ++i)
        {
            if (fToUint(output[i]) != ref[i]) success = false;
            //            std::cout << hex << "0x" << fToUint(output[i]) << ", " ;
        }
        std::cout << "sqrt(x) test: " << (success ? "Success" : "Fail") << std::endl;
    }

    {
        // integration test
        vec3 pos(0.0f, 0.0f, 0.0f);
        vec3 vel(0.0f, 0.0f, 0.0f);
        vec3 acc(0.1f, 0.2f, 0.5f);
        float dt = 1.0f / 60.0f;
        for (int i = 0; i < 10000; ++i)
        {
            pos += vel * dt * x;
            vel += acc * dt * x;
        }
        unsigned int refx = 0x44ad9a30;
        unsigned int refy = 0x452d9a30;
        unsigned int refz = 0x45d900b7;
        std::cout << "Integration test: "
                  << ((fToUint(pos[0]) == refx) && (fToUint(pos[1]) == refy) && (fToUint(pos[2]) == refz) ? "Success"
                                                                                                          : "Fail")
                  << std::endl;
        //        std::cout << hex << fToUint(pos[0]) << " " << fToUint(pos[1])<< " " << fToUint(pos[2])  << std::endl;
    }


    {
        const int seed = 49562631;
        std::mt19937 mt;
        mt.seed(seed);

        std::uniform_real_distribution<float> dis(0.0f, 1.0f);

        float f          = dis(mt);
        unsigned int ref = 0x3f36b6eb;

        std::cout << "Random distribution test: " << ((fToUint(f) == ref) ? "Success" : "Fail") << std::endl;


        for (int i = 0; i < 10 * 1000 * 1000; ++i)
        {
            // do some random floating point operations
            f += dis(mt);
            f -= dis(mt);
            f *= dis(mt) + 0.00001;
            f /= dis(mt) + 0.00001;
            f *= sqrtf(dis(mt));
            f *= sin(dis(mt));
            f *= cos(dis(mt));
        }

        ref = 0xbc93ad8d;
        std::cout << "Random Operations test: " << ((fToUint(f) == ref) ? "Success" : "Fail") << std::endl;
        //        std::cout << hex << fToUint(f) << " " << f << std::endl;
    }
}

}  // namespace Tests
}  // namespace Saiga
