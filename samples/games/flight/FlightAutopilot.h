/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "games/flight/FlightModel.h"

namespace Saiga
{
namespace Game
{
/**
 * Flies a circuit around a center at a height above the ground (the demo mode of the flight simulator): banks
 * towards the circle's tangent (corrected towards the radius), pitches to hold the height, throttles to hold the
 * speed, yaws against side slip.
 */
struct FlightAutopilot
{
    Vec3 center    = Vec3(0, 0, 0);
    double radius  = 2500;
    double height  = 300;
    double speed   = 65;
    double maxBank = 40.0 * kPi / 180.0;
    // Counter clockwise seen from above
    bool counterClockwise = true;

    AircraftControls controls(const FlightModel& model) const;
};

}  // namespace Game
}  // namespace Saiga
