/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/engine/ecs/Ids.h"

#include "games/common/DrivingComponents.h"

namespace Saiga
{
namespace Game
{
// The racing demo's component IDs
enum class RacingComponent : uint8_t
{
    VehicleControls = kFirstDemoComponent,  // after the common components (games/common/DrivingComponents.h)
    VehicleState,
    LapState,
    DriverState,
};

}  // namespace Game
}  // namespace Saiga
