/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/math/math.h"

#include "internal/noGraphicsAPI.h"

namespace Saiga
{
/**
 * SSE instructions:
 *
 * MOVSS  Destination[0..31] = Source[0..31];
 * ADDSS  dst[0-31]   := dst[0-31] + src[0-31],
 * ADDPS actual sse addition
 *
 *
 */



// See assembly:
// Linux objdump -d -M intel -S simd_test.cpp.o


// typedef tvec4<float, precision::aligned_highp> vec4_t;
// typedef aligned_vec4 vec4_t;
// typedef tmat4x4<float, precision::aligned_highp> mat4_t;

typedef vec4 vec4_t;
typedef mat4 mat4_t;

// typedef simdVec4 vec4_t;
// typedef simdMat4 mat4_t;

SAIGA_CORE_API vec4_t globalA;
SAIGA_CORE_API vec4_t globalB;


SAIGA_CORE_API vec4_t add2(vec4_t a, vec4_t b)
{
    //    aligned_vec4 A(a);
    //    aligned_vec4 B(b);
    // asm("#vec4 add start ");
    auto ret = a + b;
    // asm("#vec4 add end");
    return ret;
}


SAIGA_CORE_API vec4_t addGlobal()
{
    // asm("#vec4 add start");
    auto ret = globalA + globalB;
    // asm("#vec4 add end");
    return ret;
}

SAIGA_CORE_API mat4_t mult(mat4_t a, mat4_t b)
{
    mat4_t ret;
    // asm("#mat4 mult start");
    ret = a * b;
    // asm("#mat4 mult end");
    return ret;
}

}  // namespace Saiga
