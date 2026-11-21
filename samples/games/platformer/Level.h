/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/ChunkId.h"

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
 * The platformer level of one space: axis aligned boxes (ground, platforms, walls), a start and a goal. demoLevel() is
 * the outdoor level of the tech demo (space 0); waypoints are the top centers of the platforms in order (used by the
 * demo runner). cellarLevel() (S21) is the enclosed cellar (space 1) around its site origin.
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
    // The cellar (S21): a floor, low walls and two platforms around the site origin of the enclosed space
    // (SpaceTemplate::enclosedOrigin(4) = (640, 0, 640))
    static Level cellarLevel();
};

// A waypoint of the demo route (S21): a position in a space; a door is used (the interact action) when reached
struct DemoWaypoint
{
    SpaceId space = 0;
    Vec3 position = Vec3::Zero();
    bool door     = false;
};

// The doors of the demo (S21): the entrance on the start island (space 0) and the exit in the cellar (space 1)
struct CellarDoors
{
    static constexpr int32_t kCellarChunks = 4;
    static Vec3 origin();           // the cellar's site origin (640, 0, 640)
    static Vec3 entrance();         // on the start island
    static Vec3 entranceArrival();  // next to the entrance (back from the cellar)
    static Vec3 cellarArrival();    // in the cellar
    static Vec3 exit();             // in the cellar
};

// The routes of the demo runner: 0 the outdoor level's waypoints (space 0), 1 through the cellar and to the goal
std::vector<DemoWaypoint> outdoorRoute(const Level& outdoor);
std::vector<DemoWaypoint> cellarRoute(const Level& outdoor);

}  // namespace Game
}  // namespace Saiga
