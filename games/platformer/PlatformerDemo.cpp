/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Demo runner of the platformer: follows the waypoints, jumps at ledges.

#include "games/platformer/PlatformerDemo.h"

namespace Saiga
{
namespace Game
{
CharacterInput PlatformerDemo::input(const CharacterController& c, const Level& level)
{
    CharacterInput in;
    const auto& s = c.state;
    if (next >= level.waypoints.size()) return in;

    Vec3 target = level.waypoints[next];
    Vec3 d(target(0) - s.position(0), 0, target(2) - s.position(2));
    double dist = d.norm();
    if (dist < 0.6 && s.grounded && std::abs(s.position(1) - target(1)) < 0.05)
    {
        next++;
        if (next >= level.waypoints.size()) return in;
        target = level.waypoints[next];
        d      = Vec3(target(0) - s.position(0), 0, target(2) - s.position(2));
        dist   = d.norm();
    }
    in.move = dist > 1e-6 ? Vec3(d / dist) : Vec3(0, 0, 0);
    // Slow down when close, so the waypoint is not overshot
    if (dist < 1.5) in.move *= std::max(dist / 1.5, 0.3);

    // Jump at a ledge: the ground ahead is missing, or the target is higher than the current ground
    in.jumpHeld = true;
    if (s.grounded)
    {
        Vec3 ahead     = s.position + in.move.normalized() * 0.6;
        double support = level.supportHeight(ahead(0), ahead(2), s.position(1) + 0.01);
        bool ledge     = support < s.position(1) - 0.3;
        bool higher    = target(1) > s.position(1) + 0.2 && dist < 3.2;
        in.jumpPressed = ledge || higher;
    }
    return in;
}

bool PlatformerDemo::reachedGoal(const CharacterController& c, const Level& level) const
{
    const auto& p = c.state.position;
    return c.state.grounded && std::abs(p(1) - level.goal(1)) < 0.05 &&
           Vec3(p(0) - level.goal(0), 0, p(2) - level.goal(2)).norm() < 2.0;
}

}  // namespace Game
}  // namespace Saiga
