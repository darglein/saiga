/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/util/assert.h"
#include "saiga/util/glm.h"
#include "saiga/vision/VisionIncludes.h"
namespace Saiga
{
namespace EigenHelper
{
struct EigenCompileFlags
{
    int versionWorld, versionMajor, versionMinor;
    bool debug = true;
    bool fma = false, sse3 = false, ssse3 = false, sse41 = false, sse42 = false, avx = false, avx2 = false;


    /**
     * Use a useless template here, so the function is generated again for every project.
     */
    template <int a>
    void create()
    {
        versionWorld = EIGEN_WORLD_VERSION;
        versionMajor = EIGEN_MAJOR_VERSION;
        versionMinor = EIGEN_MINOR_VERSION;

#ifdef EIGEN_NO_DEBUG
        debug = false;
#endif
#ifdef EIGEN_VECTORIZE_FMA
        fma = true;
#endif
#ifdef EIGEN_VECTORIZE_SSE3
        sse3 = true;
#endif
#ifdef EIGEN_VECTORIZE_SSSE3
        ssse3 = true;
#endif
#ifdef EIGEN_VECTORIZE_SSE4_1
        sse41 = true;
#endif
#ifdef EIGEN_VECTORIZE_SSE4_2
        sse42 = true;
#endif
#ifdef EIGEN_VECTORIZE_AVX
        avx = true;
#endif
#ifdef EIGEN_VECTORIZE_AVX2
        avx2 = true;
#endif
    }

    bool operator==(const EigenCompileFlags& o)
    {
        return std::make_tuple(versionWorld, versionMajor, versionMinor, debug, fma, sse3, ssse3, sse41, sse42, avx,
                               avx2) == std::make_tuple(o.versionWorld, o.versionMajor, o.versionMinor, o.debug, o.fma,
                                                        o.sse3, o.ssse3, o.sse41, o.sse42, o.avx, o.avx2);
    }
};

/**
 * The eigen compile flags saiga was compiled with.
 * This should mach the applications flags to ensure compabitility
 */
SAIGA_GLOBAL EigenCompileFlags getSaigaEigenCompileFlags();


template <int n>
void checkEigenCompabitilty()
{
    EigenCompileFlags e;
    e.create<n>();

    if (!(e == getSaigaEigenCompileFlags()))
    {
        SAIGA_EXIT_ERROR("Your eigen compile flags do not match to Saiga's compile flags!");
    }
}

}  // namespace EigenHelper
}  // namespace Saiga
