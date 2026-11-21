/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The exact conversions of generated integers to simulation values (docs/engine_design.md §12.1, R-CONTENT-1; S23a):
// the only place generated integers become floating point. Positions: int to double, a power-of-two scale and an
// addition (exact below 2^53 / 256); a yaw rotation from the trig table (the half angle), normalized in double (the
// table's norm error stays out of saved and replicated state; the hashed data stays the integer yaw); scale / 1024.

#pragma once

#include "saiga/core/math/ChunkId.h"
#include "saiga/core/math/Quaternion.h"
#include "saiga/core/math/Types.h"
#include "saiga/export.h"

#include <cstdint>

namespace Saiga
{
namespace Engine
{
// A space position: x, z in 1/256 m, y in height units
SAIGA_ENGINE_API Vec3 contentPosition(int64_t x256, int32_t yHeight, int64_t z256);
// A chunk-relative position (x, z in 1/256 m of the chunk) in space coordinates
SAIGA_ENGINE_API Vec3 contentPosition(const ChunkId& chunk, int32_t x256, int32_t yHeight, int32_t z256);
// The rotation by a yaw (65536 = one turn) about +y: (cos(yaw / 2), 0, sin(yaw / 2), 0) from the Q30 table, scaled
// exactly (its norm is within 2^-28 of 1; no libm, R-CONTENT-1)
SAIGA_ENGINE_API Quat contentRotation(uint16_t yaw);
SAIGA_ENGINE_API double contentScale(uint16_t scale);

}  // namespace Engine
}  // namespace Saiga
