/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// TrigTable (docs/engine_design.md §12.1, R-CONTENT-1; S23a): sine and cosine for a circle of 2^17 steps in Q30, from a
// quarter-wave table (32769 entries) filled once by an integer Taylor series, so every compiler has the same table
// without a committed source file. A uint16_t yaw (65536 = one turn) gives sin / cos(yaw) at index 2 * yaw and the
// half angle of a yaw quaternion at index yaw.

#pragma once

#include "saiga/export.h"

#include <cstdint>
#include <vector>

namespace Saiga
{
namespace Engine
{
constexpr uint32_t kTrigSteps     = uint32_t(1) << 17;  // one turn
constexpr uint32_t kTrigQuarter   = uint32_t(1) << 15;
constexpr int64_t kTrigOne        = int64_t(1) << 30;  // 1.0 in Q30
constexpr uint32_t kTrigTableSize = kTrigQuarter + 1;

// The quarter-wave table: sin(i * (pi / 2) / 32768) in Q30, i in [0, 32768] (filled at the first call; thread safe)
SAIGA_ENGINE_API const std::vector<int32_t>& trigQuarterTable();

// sin(a * 2 pi / 2^17) in Q30 (a modulo 2^17)
SAIGA_ENGINE_API int32_t sinQ30(uint32_t a17);
// cos(a) = sin(a + 2^15): exact symmetries by construction
SAIGA_ENGINE_API int32_t cosQ30(uint32_t a17);

}  // namespace Engine
}  // namespace Saiga
