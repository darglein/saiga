/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/config.h"
#include "saiga/core/math/Eigen_Compile_Checker.h"
#include "saiga/core/math/all.h"
#include "saiga/core/util/Align.h"
#include "saiga/core/util/ConsoleColor.h"
#include "saiga/core/util/table.h"

#include "gtest/gtest.h"

namespace Saiga
{
void CheckVectorInstructions()
{
    EigenHelper::EigenCompileFlags flags;
    flags.create<938476>();
    std::cout << flags << std::endl;
}
void CheckEigenVectorAlignment()
{
    Table table({20, 10, 10});

    table << "Type"
          << "size"
          << "alignment";

    using vec8  = Eigen::Matrix<float, 8, 1>;
    using vec16 = Eigen::Matrix<float, 16, 1>;
    using vec32 = Eigen::Matrix<float, 32, 1>;

    table << "vec2<float>" << sizeof(vec2) << alignof(vec2);
    table << "vec3<float>" << sizeof(vec3) << alignof(vec3);
    table << "vec4<float>" << sizeof(vec4) << alignof(vec4);
    table << "vec8<float>" << sizeof(vec8) << alignof(vec8);
    table << "vec16<float>" << sizeof(vec16) << alignof(vec16);
    table << "vec32<float>" << sizeof(vec32) << alignof(vec32);

    using Vec8  = Eigen::Matrix<double, 8, 1>;
    using Vec16 = Eigen::Matrix<double, 16, 1>;
    using Vec32 = Eigen::Matrix<double, 32, 1>;

    table << ""
          << ""
          << "";
    table << "Vec2<double>" << sizeof(Vec2) << alignof(Vec2);
    table << "Vec3<double>" << sizeof(Vec3) << alignof(Vec3);
    table << "Vec4<double>" << sizeof(Vec4) << alignof(Vec4);
    table << "Vec8<double>" << sizeof(Vec8) << alignof(Vec8);
    table << "Vec16<double>" << sizeof(Vec16) << alignof(Vec16);
    table << "Vec32<double>" << sizeof(vec32) << alignof(Vec32);


    std::cout << std::endl;
}

#ifdef EIGEN_VECTORIZE_NEON
#    include <arm_neon.h>
TEST(Vectorization, Neon)
{
    unsigned int data[4] = {1, 2, 3, 4};

    volatile uint32x4_t first  = vld1q_u32(data);
    volatile uint32x4_t second = vld1q_u32(data);
    volatile uint32x4_t result;

    result = vaddq_u32(first, second);

    vst1q_u32(data, result);

    int sum = 0;
    int ref = 0;
    for (int i = 0; i < 4; ++i)
    {
        sum += data[i];
        ref += 2 * (i + 1);
    }
    EXPECT_EQ(sum, ref);
}
#endif

#ifdef EIGEN_VECTORIZE_SSE
TEST(Vectorization, SSE)
{
    __m128i first  = _mm_set_epi32(1, 2, 3, 4);
    __m128i second = _mm_set_epi32(1, 2, 3, 4);
    __m128i result = _mm_add_epi32(first, second);
    int* values    = (int*)&result;

    int sum = 0;
    int ref = 0;
    for (int i = 0; i < 4; ++i)
    {
        sum += values[i];
        ref += 2 * (i + 1);
    }
    EXPECT_EQ(sum, ref);
}
#endif

#ifdef EIGEN_VECTORIZE_AVX2
TEST(Vectorization, AVX2)
{
    __m256i first  = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
    __m256i second = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
    __m256i result = _mm256_add_epi32(first, second);
    int* values    = (int*)&result;

    int sum = 0;
    int ref = 0;
    for (int i = 0; i < 8; ++i)
    {
        sum += values[i];
        ref += 2 * (i + 1);
    }
    EXPECT_EQ(sum, ref);
}
#endif


#ifdef EIGEN_VECTORIZE_AVX512
TEST(Vectorization, AVX512)
{
    __m512i first  = _mm512_set_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
    __m512i second = _mm512_set_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
    __m512i result = _mm512_add_epi32(first, second);
    int* values    = (int*)&result;

    int sum = 0;
    int ref = 0;
    for (int i = 0; i < 16; ++i)
    {
        sum += values[i];
        ref += 2 * (i + 1);
    }
    EXPECT_EQ(sum, ref);
}
#endif


}  // namespace Saiga

int main()
{
    Saiga::CheckVectorInstructions();
    Saiga::CheckEigenVectorAlignment();
    testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}
