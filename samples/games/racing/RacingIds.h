/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/engine/ecs/Ids.h"

namespace Saiga
{
namespace Game
{
// The racing demo's component IDs
enum class RacingComponent : uint8_t
{
    VehicleControls = Engine::kFirstGameComponent,
    VehicleState,
    LapState,
    DriverState,
};

}  // namespace Game
}  // namespace Saiga
