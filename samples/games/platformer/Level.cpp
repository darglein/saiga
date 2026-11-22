/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The level of the platformer tech demo (boxes, start, goal, waypoints).

#include "games/platformer/Level.h"

#include "saiga/engine/world/SpaceTemplate.h"

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

Vec3 CellarDoors::origin()
{
    const double o = double(Engine::SpaceTemplate::enclosedOrigin(kCellarChunks));
    return Vec3(o, 0, o);
}
Vec3 CellarDoors::entrance()
{
    // On the start island, left of the start and away from the wall and the step
    return Vec3(-2.5, 0, 4.5);
}
Vec3 CellarDoors::entranceArrival()
{
    return Vec3(-1.5, 0, 3);
}
Vec3 CellarDoors::cellarArrival()
{
    return origin() + Vec3(0, 0, 3);
}
Vec3 CellarDoors::exit()
{
    return origin() + Vec3(-2, 0, -21);
}

Level Level::cellarLevel()
{
    Level l;
    const Vec3 o = CellarDoors::origin();
    Vec3 floor(0.32, 0.3, 0.28), wall(0.42, 0.38, 0.34), stone(0.55, 0.52, 0.48), wood(0.5, 0.36, 0.24);
    // The floor (12 x 32 m), side walls, a low back curb (the chase camera behind the hero looks over it) and the end
    // wall behind the exit
    l.boxes.push_back({o + Vec3(-6, -1, -26), o + Vec3(6, 0, 6), floor});
    l.boxes.push_back({o + Vec3(-7, 0, -26), o + Vec3(-6, 2.2, 6), wall});
    l.boxes.push_back({o + Vec3(6, 0, -26), o + Vec3(7, 2.2, 6), wall});
    l.boxes.push_back({o + Vec3(-7, 0, 6), o + Vec3(7, 0.4, 7), wall});
    l.boxes.push_back({o + Vec3(-7, 0, -27), o + Vec3(7, 3.5, -26), wall});
    // Crates along the walls
    l.boxes.push_back({o + Vec3(-5.6, 0, -3), o + Vec3(-4.4, 1.2, -1.8), wood});
    l.boxes.push_back({o + Vec3(4.2, 0, -16), o + Vec3(5.6, 0.9, -14.6), wood});
    // Two platforms to climb over (as outdoors: rises of 0.8 m, gaps the jump clears)
    l.addPlatform(o + Vec3(0, 0.8, -6), 1.5, 1.5, stone);
    l.addPlatform(o + Vec3(2, 1.6, -10.5), 1.5, 1.5, stone);
    l.start     = CellarDoors::cellarArrival();
    l.goal      = CellarDoors::exit();
    l.waypoints = {o + Vec3(0, 0.8, -6), o + Vec3(2, 1.6, -10.5), o + Vec3(2, 0, -16)};
    return l;
}

std::vector<DemoWaypoint> outdoorRoute(const Level& outdoor)
{
    std::vector<DemoWaypoint> route;
    for (const Vec3& p : outdoor.waypoints) route.push_back(DemoWaypoint{0, p, false});
    return route;
}

std::vector<DemoWaypoint> cellarRoute(const Level& outdoor)
{
    // The entrance, the cellar's platforms, the exit, then the outdoor level from its first platform
    std::vector<DemoWaypoint> route;
    route.push_back(DemoWaypoint{0, CellarDoors::entrance(), true});
    for (const Vec3& p : Level::cellarLevel().waypoints) route.push_back(DemoWaypoint{1, p, false});
    route.push_back(DemoWaypoint{1, CellarDoors::exit(), true});
    for (const DemoWaypoint& w : outdoorRoute(outdoor)) route.push_back(w);
    return route;
}

}  // namespace Game
}  // namespace Saiga
