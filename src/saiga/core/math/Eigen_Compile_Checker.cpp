/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "Eigen_Compile_Checker.h"

namespace Saiga
{
namespace EigenHelper
{
std::ostream& operator<<(std::ostream& strm, const EigenCompileFlags& flags)
{
    strm << "[EigenCompileFlags]" << std::endl;
    strm << "Eigen Version: " << flags.versionWorld << "." << flags.versionMajor << "." << flags.versionMinor << std::endl;
    strm << "Eigen Debug: " << flags.debug << std::endl;
    strm << "FMA: " << flags.fma << std::endl;
    strm << "SSE3/SSSE3: " << flags.sse3 << "/" << flags.ssse3 << std::endl;
    strm << "SSE4.1/SSE4.2: " << flags.sse41 << "/" << flags.sse42 << std::endl;
    strm << "AVX/AVX2: " << flags.avx << "/" << flags.avx2 << std::endl;
    strm << "AVX512: " << flags.avx512;
    return strm;
}
EigenCompileFlags getSaigaEigenCompileFlags()
{
    EigenCompileFlags f;
    f.create<9267345>();
    return f;
}

}  // namespace EigenHelper
}  // namespace Saiga
