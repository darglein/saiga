/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "games/platformer/CharacterController.h"

namespace Saiga
{
namespace Game
{
/**
 * The demo runner of the platformer: runs towards the level's waypoints in order and jumps at ledges (when the ground
 * 0.6 m ahead is missing or lower than the next waypoint requires). Restarts at the first waypoint after a respawn.
 */
struct PlatformerDemo
{
    size_t next = 0;

    CharacterInput input(const CharacterController& character, const Level& level)
    {
        return input(character, level, level.waypoints);
    }
    // Towards the given waypoints (next indexes them) over the level's boxes (S21: a route segment of one space)
    CharacterInput input(const CharacterController& character, const Level& level, const std::vector<Vec3>& waypoints);
    bool reachedGoal(const CharacterController& character, const Level& level) const;
};

}  // namespace Game
}  // namespace Saiga
