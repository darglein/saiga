/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "Eigen_Compile_Checker.h"

#include "saiga/core/util/ConsoleColor.h"
#include "saiga/core/util/table.h"
#include "saiga/core/util/tostring.h"
namespace Saiga
{
namespace EigenHelper
{
std::ostream& operator<<(std::ostream& strm, const EigenCompileFlags& flags)
{
    strm << ConsoleColor::CYAN;
    strm << "========= Eigen Info ==========" << std::endl;

    Table table({2, 18, 10, 1}, strm);
    table << "|"
          << "Eigen Version"
          << (to_string(flags.versionWorld) + "." + to_string(flags.versionMajor) + "." + to_string(flags.versionMinor))
          << "|";

    table << "|"
          << "Debug " << flags.debug << "|";
    table << "|"
          << "FAST_MATH " << flags.fast_math << "|";

    table << "|"
          << "FMA " << flags.fma << "|";

    table << "|"
          << "SSE3/SSSE3 " << to_string(flags.sse3) + "/" + to_string(flags.ssse3) << "|";

    table << "|"
          << "SSE4.1/SSE4.2 " << to_string(flags.sse41) + "/" + to_string(flags.sse42) << "|";
    table << "|"
          << "AVX/AVX2 " << to_string(flags.avx) + "/" + to_string(flags.avx2) << "|";
    table << "|"
          << "AVX512 " << flags.avx512 << "|";
    table << "|"
          << "NEON " << flags.neon << "|";
    table << "|"
          << "VSX " << flags.vsx << "|";
    table << "|"
          << "ALTIVEC" << flags.altivec << "|";
    table << "|"
          << "ZVECTOR " << flags.zvector << "|";

    strm << "===============================";

    strm.unsetf(std::ios_base::floatfield);
    strm << ConsoleColor::RESET;

    return strm;
}
EigenCompileFlags getSaigaEigenCompileFlags()
{
    EigenCompileFlags f;
    f.create<9267345>();
    return f;
}

size_t saiga_alignof_vec2()
{
    return alignof(vec2);
}

size_t saiga_alignof_vec4()
{
    return alignof(vec4);
}

size_t saiga_alignof_mat4()
{
    return alignof(mat4);
}

size_t saiga_alignof_Mat4()
{
    return alignof(Mat4);
}

}  // namespace EigenHelper
}  // namespace Saiga
