/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "games/common/GameMath.h"

#include <cstdint>
#include <vector>

namespace Saiga
{
namespace Game
{
/**
 * A closed race track: a Catmull-Rom spline through control points in the xz plane, sampled by arc length, on the
 * ground. Checkpoints (gates across the track) are placed at equal distances; checkpoint 0 is the start/finish line.
 */
class Track
{
   public:
    double width = 10;
    // The asphalt lies this far above the terrain (so it does not intersect the terrain's triangles); the curbs slope
    // down to the terrain over curbWidth on both sides
    double lift = 0.15, curbWidth = 0.8;

    // control: points in the xz plane (y is ignored), at least 4
    void build(const std::vector<Vec3>& control, int checkpoints = 10);

    double length() const { return totalLength; }
    // Position (y = 0) and unit tangent at arc length s (wraps around)
    Vec3 position(double s) const;
    Vec3 tangent(double s) const;
    // Curvature (1/m) at arc length s
    double curvature(double s) const;
    // Arc length of the closest sample to p, searched near hint (or globally if hint < 0)
    double closest(const Vec3& p, double hint = -1) const;
    // Height of the track surface above the terrain at p (lift on the asphalt, sloping to 0 over the curbs, 0 beyond);
    // hint: as for closest, updated to the arc length found. The track mesh and the car's ground use it, so the wheels
    // touch the asphalt that is drawn.
    double surfaceOffset(const Vec3& p, double& hint) const;

    struct Checkpoint
    {
        Vec3 center, tangent;
        double s;
    };
    const std::vector<Checkpoint>& checkpoints() const { return gates; }

    // A loop around center with the given radius and a little radial variation (deterministic for the seed)
    static std::vector<Vec3> loop(const Vec3& center, double radius, int points, uint32_t seed);
    // Searches the world (x, z in [margin, size - margin]) for the center of a loop(center, radius, points, seed) that
    // stays on land (above minHeight) and on gentle slopes; penalty (optional) adds a cost per sample (every 8 m) of
    // the track, e.g. for trees. Returns the best center found.
    static Vec3 findSite(const GroundHeight& ground, double size, double radius, double minHeight, int points,
                         uint32_t seed, const std::function<double(const Vec3&)>& penalty = nullptr);

   private:
    std::vector<Vec3> samples;
    std::vector<double> distances;
    double totalLength = 0;
    std::vector<Checkpoint> gates;
    double step = 1;
};

}  // namespace Game
}  // namespace Saiga
