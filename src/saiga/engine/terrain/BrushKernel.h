/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// BrushKernel (docs/engine_design.md §12.2; S22): the integer brush kernel of terraforming, pure functions on arrays
// (no World). A validated TerrainBrush is quantized with exact operations (power-of-two scaling and floor), then every
// sample of the affected rectangle is evaluated from the pre-edit heights only (reads from a separate input, writes to
// a separate output: the result does not depend on the iteration order) with int64 arithmetic:
//  - quartic falloff (1 - t^2)^2 without square roots: t2 = floorDiv(d2 << 16, r2), u = 65536 - t2,
//    w = (u * u) >> 16 in [0, 65536]; a sample is affected iff d2 < r2 (strictly inside);
//  - Add / Subtract: h +- ((s * w) >> 16);
//  - Flatten: g = (f * w) >> 16, h + roundShift16((cy - h) * g) (never overshoots the target);
//  - Smooth: g = (f * w) >> 16, h + floorDiv(2 * (sum - 9 h) * g + 9 * 65536, 18 * 65536) (the 3 x 3 mean blend rounded
//    once to the nearest, ties toward +infinity; sum over the indices clamped to the space);
//  - h' = (v == h) ? h : clamp(v, +-kTerrainHeightLimit); a sample is changed iff h' != h (after the clamp).

#pragma once

#include "saiga/engine/content/gen/FixedMath.h"
#include "saiga/engine/terrain/SampleRect.h"
#include "saiga/engine/terrain/TerrainBrush.h"
#include "saiga/export.h"

#include <cstdint>

namespace Saiga
{
namespace Engine
{
// floorDiv / ceilDiv (b > 0): the content toolkit's (S23a, gen/FixedMath.h)
// v / 65536 rounded to the nearest integer, ties toward positive infinity
inline int64_t roundShift16(int64_t v)
{
    return floorDiv(v + 32768, 65536);
}

// A quantized brush (BrushKernel::quantize, after validateBrush)
struct QuantizedBrush
{
    TerrainBrush::Op op = TerrainBrush::Add;
    // Center in 1/256 m; the Flatten target in height units; radius in 1/256 m (>= 1)
    int64_t cx = 0, cz = 0, cy = 0, r = 1;
    // Add/Subtract: height units at the center; Flatten/Smooth: the blend factor in 1/65536
    int64_t s = 0, f = 0;
};

namespace BrushKernel
{
// Exact quantization of a validated brush (floor of power-of-two multiples; the bounds of validateBrush keep every
// value small: |cx| < 2^26, |cy| <= 2^23, r <= 2^15, s and f <= 2^16)
SAIGA_ENGINE_API QuantizedBrush quantize(const TerrainBrush& brush);
// The affected rectangle: i in [ceilDiv(cx - r, sp * 256), floorDiv(cx + r, sp * 256)] (same for k), clipped to
// [0, maxSample] (maxSample = sizeChunks * 256 / sp)
SAIGA_ENGINE_API SampleRect affectedRect(const QuantizedBrush& q, int spacing, int64_t maxSample);
// The rectangle the evaluation reads: the affected one, for Smooth grown by one sample, clipped to [0, maxSample]
SAIGA_ENGINE_API SampleRect readRect(const QuantizedBrush& q, const SampleRect& affected, int64_t maxSample);
// The weight of sample (i, k) in [0, 65536]; -1 if the sample is not affected (d2 >= r2). Only for samples of the
// affected rectangle (|dx|, |dz| <= r: no overflow)
SAIGA_ENGINE_API int64_t weight(const QuantizedBrush& q, int spacing, int64_t i, int64_t k);
// The new height of sample (i, k) of the affected rectangle: pre holds the pre-edit heights of read (row-major, row =
// k); unaffected samples keep their height
SAIGA_ENGINE_API int32_t evaluateSample(const QuantizedBrush& q, int spacing, const SampleRect& read,
                                        const int32_t* pre, int64_t i, int64_t k);
// Every sample of affected into out (row-major over affected); returns the number of changed samples (out != pre)
SAIGA_ENGINE_API uint64_t evaluate(const QuantizedBrush& q, int spacing, const SampleRect& affected,
                                   const SampleRect& read, const int32_t* pre, int32_t* out);
}  // namespace BrushKernel

}  // namespace Engine
}  // namespace Saiga
