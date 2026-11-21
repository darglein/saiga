/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// SampleRect (docs/engine_design.md §12.2; S22): an inclusive rectangle of base-grid samples of a space (the brush
// kernel's rectangles, the changed boxes of terrain edits, the visual dirty queue).

#pragma once

#include <cstdint>

namespace Saiga
{
namespace Engine
{
// An inclusive rectangle of space samples (empty when i1 < i0 or k1 < k0)
struct SampleRect
{
    int64_t i0 = 0, k0 = 0, i1 = -1, k1 = -1;
    bool empty() const { return i1 < i0 || k1 < k0; }
    int64_t width() const { return empty() ? 0 : i1 - i0 + 1; }
    int64_t height() const { return empty() ? 0 : k1 - k0 + 1; }
    bool contains(int64_t i, int64_t k) const { return i >= i0 && i <= i1 && k >= k0 && k <= k1; }
};

}  // namespace Engine
}  // namespace Saiga
