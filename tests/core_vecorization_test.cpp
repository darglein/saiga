/**
 * Copyright (c) 2017 Darius Rückert
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
Table* table;
void PrintVectorEnabled(const std::string& intruction_set, bool enabled)
{
    (*table) << intruction_set << (enabled ? ConsoleColor::GREEN : ConsoleColor::RED) << (enabled ? "YES" : "NO")
             << ConsoleColor::RESET;
}

void CheckVectorInstructions()
{
    table = new Table({10, 0, 1, 0});
    EigenHelper::EigenCompileFlags flags;
    flags.create<938476>();

    std::cout << "Enabled Vector Instructions:" << std::endl;
    PrintVectorEnabled("fma", flags.fma);
    PrintVectorEnabled("sse3", flags.sse3);
    PrintVectorEnabled("ssse3", flags.ssse3);
    PrintVectorEnabled("sse41", flags.sse41);
    PrintVectorEnabled("sse42", flags.sse42);
    PrintVectorEnabled("avx", flags.avx);
    PrintVectorEnabled("avx2", flags.avx2);
    PrintVectorEnabled("avx512", flags.avx512);
    PrintVectorEnabled("neon", flags.neon);
    PrintVectorEnabled("vsx", flags.vsx);
    PrintVectorEnabled("altivec", flags.altivec);
    PrintVectorEnabled("zvector", flags.zvector);
}

#include <arm_neon.h>
TEST(Vectorization, Neon)
{
    const uint8_t uint8_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

    /* Create the vector with our data. */
    uint8x16_t data;

    /* Load our custom data into the vector register. */
    data = vld1q_u8(uint8_data);
}

}  // namespace Saiga

int main()
{
    Saiga::CheckVectorInstructions();
    testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}
