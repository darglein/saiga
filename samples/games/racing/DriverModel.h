/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "games/racing/Track.h"
#include "games/racing/Vehicle.h"

namespace Saiga
{
namespace Game
{
/**
 * The AI driver model of the racing demo (the RaceDriver game object uses it): steers towards a point ahead on the
 * track (the look ahead grows with the speed) and limits the speed by the curvature ahead (corner speed v =
 * sqrt(lateral acceleration / curvature), reachable within the braking distance).
 */
struct DriverModel
{
    double lookAhead = 10, maxSpeed = 30, lateralAcceleration = 7.0;
    // Deceleration assumed for the braking distance (m/s^2)
    double braking = 5.0;
    // Arc length of the car on the track (updated by controls)
    double progress = -1;

    VehicleControls controls(const Vehicle& car, const Track& track);
};

}  // namespace Game
}  // namespace Saiga
