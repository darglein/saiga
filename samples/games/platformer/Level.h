/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "games/common/GameMath.h"

#include <vector>

namespace Saiga
{
namespace Game
{
struct LevelBox
{
    Vec3 min, max;
    Vec3 color;
};

/**
 * The platformer level: axis aligned boxes (ground, platforms, walls), a start and a goal. demoLevel() is the one
 * level of the tech demo; waypoints are the top centers of the platforms in order (used by the demo runner).
 */
struct Level
{
    std::vector<LevelBox> boxes;
    Vec3 start = Vec3(0, 0, 0);
    Vec3 goal  = Vec3(0, 0, 0);
    std::vector<Vec3> waypoints;
    // Below this height the character respawns
    double killHeight = -12;

    // A platform with its top at top (center), half extents hx, hz and the given thickness
    void addPlatform(const Vec3& top, double hx, double hz, const Vec3& color, double thickness = 0.5);
    // Highest box top at (x, z) that is at most maxY (and in the box's xz range); -inf if there is none
    double supportHeight(double x, double z, double maxY) const;

    static Level demoLevel();
};

}  // namespace Game
}  // namespace Saiga
