/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The level of the platformer tech demo (boxes, start, goal, waypoints).

#include "games/platformer/Level.h"

#include <limits>

namespace Saiga
{
namespace Game
{
void Level::addPlatform(const Vec3& top, double hx, double hz, const Vec3& color, double thickness)
{
    boxes.push_back(
        {Vec3(top(0) - hx, top(1) - thickness, top(2) - hz), Vec3(top(0) + hx, top(1), top(2) + hz), color});
}

double Level::supportHeight(double x, double z, double maxY) const
{
    double best = -std::numeric_limits<double>::infinity();
    for (auto& b : boxes)
    {
        if (x < b.min(0) || x > b.max(0) || z < b.min(2) || z > b.max(2)) continue;
        if (b.max(1) <= maxY) best = std::max(best, b.max(1));
    }
    return best;
}

Level Level::demoLevel()
{
    Level l;
    Vec3 grass(0.45, 0.62, 0.35), stone(0.62, 0.6, 0.56), wood(0.62, 0.45, 0.3), gold(0.85, 0.7, 0.25);
    // Start island with a wall and a step
    l.boxes.push_back({Vec3(-6, -2, -6), Vec3(6, 0, 6), grass});
    l.boxes.push_back({Vec3(-5, 0, -1), Vec3(-4, 1.6, 3), stone});
    l.boxes.push_back({Vec3(2.5, 0, 1.5), Vec3(4.5, 0.4, 3.5), stone});

    // A climb of platforms (gaps of 1.5 - 2.3 m, rises of up to 0.8 m: the jump reaches about 1.6 m and 4 m)
    l.addPlatform(Vec3(0, 0.8, -9.5), 1.5, 1.5, stone);
    l.addPlatform(Vec3(0, 1.6, -14), 1.5, 1.5, stone);
    l.addPlatform(Vec3(4, 2.4, -17), 1.5, 1.5, stone);
    // A long bridge and a thin platform
    l.addPlatform(Vec3(10, 2.4, -17), 3, 0.75, wood, 0.3);
    l.addPlatform(Vec3(16, 3.2, -17), 1.2, 1.2, stone, 0.15);
    l.addPlatform(Vec3(16, 4.0, -22), 1.5, 1.5, stone);
    // Goal island
    l.addPlatform(Vec3(16, 4.0, -28), 2.5, 2.5, gold, 1.0);

    l.start     = Vec3(0, 0, 2);
    l.goal      = Vec3(16, 4.0, -29);
    l.waypoints = {Vec3(0, 0.8, -9.5), Vec3(0, 1.6, -14),  Vec3(4, 2.4, -17),   Vec3(10, 2.4, -17),
                   Vec3(16, 3.2, -17), Vec3(16, 4.0, -22), Vec3(16, 4.0, -28.5)};
    return l;
}

}  // namespace Game
}  // namespace Saiga
