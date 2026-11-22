/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/terrain/BrushKernel.h"

#include "saiga/core/math/ChunkId.h"

#include <algorithm>
#include <cmath>

namespace Saiga
{
namespace Engine
{
namespace
{
std::string number(double v)
{
    if (std::isnan(v)) return "nan";
    if (std::isinf(v)) return v > 0 ? "inf" : "-inf";
    return std::to_string(v);
}
}  // namespace

bool validateBrush(const TerrainBrush& b, const SpaceTemplate& space, std::string* error)
{
    auto fail = [&](const std::string& message)
    {
        if (error) *error = "TerrainBrush: " + message;
        return false;
    };
    const double x = b.center.x(), y = b.center.y(), z = b.center.z();
    if (!std::isfinite(x)) return fail("center.x must be finite (is " + number(x) + ")");
    if (!std::isfinite(y)) return fail("center.y must be finite (is " + number(y) + ")");
    if (!std::isfinite(z)) return fail("center.z must be finite (is " + number(z) + ")");
    if (!std::isfinite(b.radius)) return fail("radius must be finite (is " + number(b.radius) + ")");
    if (!std::isfinite(b.strength)) return fail("strength must be finite (is " + number(b.strength) + ")");
    if (uint8_t(b.op) >= uint8_t(TerrainBrush::Count))
        return fail("op must be add, subtract, flatten or smooth (is " + std::to_string(int(b.op)) + ")");
    if (!(b.radius > 0) || b.radius > kMaxBrushRadius)
        return fail("radius must be in (0, " + number(kMaxBrushRadius) + "] m (is " + number(b.radius) + ")");
    const bool step     = b.op == TerrainBrush::Add || b.op == TerrainBrush::Subtract;
    const double maxStr = step ? kMaxBrushStep : 1.0;
    if (!(b.strength >= 0) || b.strength > maxStr)
        return fail("strength must be in [0, " + number(maxStr) + "]" + (step ? " m" : "") + " for " +
                    TerrainBrush::opNames[b.op] + " (is " + number(b.strength) + ")");
    const double lo = -kMaxBrushRadius, hi = space.sizeMeters() + kMaxBrushRadius;
    if (x < lo || x > hi)
        return fail("center.x must be in [" + number(lo) + ", " + number(hi) + "] m of the space '" + space.name +
                    "' (is " + number(x) + ")");
    if (z < lo || z > hi)
        return fail("center.z must be in [" + number(lo) + ", " + number(hi) + "] m of the space '" + space.name +
                    "' (is " + number(z) + ")");
    const double limit = double(kTerrainHeightLimit) * kHeightUnit;
    if (b.op == TerrainBrush::Flatten && std::abs(y) > limit)
        return fail("center.y (the flatten target) must be within +-" + number(limit) + " m (is " + number(y) + ")");
    return true;
}

namespace BrushKernel
{
QuantizedBrush quantize(const TerrainBrush& b)
{
    QuantizedBrush q;
    q.op = b.op;
    // Exact: power-of-two scaling, then floor (the bounds are checked by validateBrush)
    q.cx = int64_t(std::floor(b.center.x() * 256));
    q.cz = int64_t(std::floor(b.center.z() * 256));
    q.r  = std::max<int64_t>(1, int64_t(std::floor(b.radius * 256)));
    if (b.op == TerrainBrush::Flatten) q.cy = int64_t(std::floor(b.center.y() * 1024));
    if (b.op == TerrainBrush::Add || b.op == TerrainBrush::Subtract)
        q.s = int64_t(std::floor(b.strength * 1024));
    else
        q.f = int64_t(std::floor(b.strength * 65536));
    return q;
}

SampleRect affectedRect(const QuantizedBrush& q, int spacing, int64_t maxSample)
{
    const int64_t unit = int64_t(spacing) * 256;
    SampleRect r;
    r.i0 = std::max<int64_t>(0, ceilDiv(q.cx - q.r, unit));
    r.i1 = std::min<int64_t>(maxSample, floorDiv(q.cx + q.r, unit));
    r.k0 = std::max<int64_t>(0, ceilDiv(q.cz - q.r, unit));
    r.k1 = std::min<int64_t>(maxSample, floorDiv(q.cz + q.r, unit));
    return r;
}

SampleRect readRect(const QuantizedBrush& q, const SampleRect& affected, int64_t maxSample)
{
    if (affected.empty() || q.op != TerrainBrush::Smooth) return affected;
    SampleRect r;
    r.i0 = std::max<int64_t>(0, affected.i0 - 1);
    r.k0 = std::max<int64_t>(0, affected.k0 - 1);
    r.i1 = std::min<int64_t>(maxSample, affected.i1 + 1);
    r.k1 = std::min<int64_t>(maxSample, affected.k1 + 1);
    return r;
}

int64_t weight(const QuantizedBrush& q, int spacing, int64_t i, int64_t k)
{
    const int64_t unit = int64_t(spacing) * 256;
    const int64_t dx = i * unit - q.cx, dz = k * unit - q.cz;
    const int64_t d2 = dx * dx + dz * dz;
    const int64_t r2 = q.r * q.r;
    if (d2 >= r2) return -1;
    // d2 < r2 <= 2^30: (d2 << 16) < 2^46, t2 in [0, 65535]
    const int64_t t2 = floorDiv(d2 * 65536, r2);
    const int64_t u  = 65536 - t2;
    return (u * u) >> 16;
}

int32_t evaluateSample(const QuantizedBrush& q, int spacing, const SampleRect& read, const int32_t* pre, int64_t i,
                       int64_t k)
{
    const int64_t stride = read.width();
    const int32_t h      = pre[size_t((k - read.k0) * stride + (i - read.i0))];
    const int64_t w      = weight(q, spacing, i, k);
    if (w < 0) return h;
    int64_t v = h;
    switch (q.op)
    {
        case TerrainBrush::Add:
            v = int64_t(h) + ((q.s * w) >> 16);
            break;
        case TerrainBrush::Subtract:
            v = int64_t(h) - ((q.s * w) >> 16);
            break;
        case TerrainBrush::Flatten:
        {
            const int64_t g = (q.f * w) >> 16;
            v               = int64_t(h) + roundShift16((q.cy - int64_t(h)) * g);
            break;
        }
        case TerrainBrush::Smooth:
        {
            const int64_t g = (q.f * w) >> 16;
            int64_t sum     = 0;
            for (int64_t dk = -1; dk <= 1; ++dk)
                for (int64_t di = -1; di <= 1; ++di)
                {
                    // Indices clamped to the space: the read rectangle holds every in-space neighbour
                    const int64_t ii = std::min(std::max(i + di, read.i0), read.i1);
                    const int64_t kk = std::min(std::max(k + dk, read.k0), read.k1);
                    sum += pre[size_t((kk - read.k0) * stride + (ii - read.i0))];
                }
            v = int64_t(h) + floorDiv(2 * (sum - 9 * int64_t(h)) * g + 9 * 65536, 18 * 65536);
            break;
        }
        default:
            break;
    }
    if (v == h) return h;
    return int32_t(std::min<int64_t>(std::max<int64_t>(v, -kTerrainHeightLimit), kTerrainHeightLimit));
}

uint64_t evaluate(const QuantizedBrush& q, int spacing, const SampleRect& affected, const SampleRect& read,
                  const int32_t* pre, int32_t* out)
{
    uint64_t changed     = 0;
    const int64_t stride = read.width();
    for (int64_t k = affected.k0; k <= affected.k1; ++k)
        for (int64_t i = affected.i0; i <= affected.i1; ++i)
        {
            const int32_t h = pre[size_t((k - read.k0) * stride + (i - read.i0))];
            const int32_t n = evaluateSample(q, spacing, read, pre, i, k);
            out[size_t((k - affected.k0) * affected.width() + (i - affected.i0))] = n;
            changed += n != h ? 1 : 0;
        }
    return changed;
}
}  // namespace BrushKernel

}  // namespace Engine
}  // namespace Saiga
