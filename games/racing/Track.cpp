/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Race track of the racing tech demo: closed spline sampled by arc length, checkpoints, site search.

#include "games/racing/Track.h"

#include <cmath>
#include <cstdint>
#include <limits>

namespace Saiga
{
namespace Game
{
namespace
{
Vec3 catmullRom(const Vec3& p0, const Vec3& p1, const Vec3& p2, const Vec3& p3, double t)
{
    double t2 = t * t, t3 = t2 * t;
    return 0.5 * ((2.0 * p1) + (p2 - p0) * t + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 +
                  (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3);
}

double hash01(uint32_t x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return (x >> 8) * (1.0 / 16777216.0);
}
}  // namespace

void Track::build(const std::vector<Vec3>& control, int checkpointCount)
{
    samples.clear();
    distances.clear();
    gates.clear();
    totalLength = 0;
    step        = 1;
    int n       = int(control.size());
    if (n < 4) return;
    auto P = [&](int i) { return Vec3(control[(i % n + n) % n](0), 0, control[(i % n + n) % n](2)); };
    // Dense samples, then resample by arc length (1 m)
    std::vector<Vec3> dense;
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < 64; ++k) dense.push_back(catmullRom(P(i - 1), P(i), P(i + 1), P(i + 2), k / 64.0));
    std::vector<double> d(dense.size() + 1, 0);
    for (size_t i = 0; i < dense.size(); ++i) d[i + 1] = d[i] + (dense[(i + 1) % dense.size()] - dense[i]).norm();
    // Regression: identical control points gave a zero length, a zero sample step and NaN positions. Degenerate
    // geometry leaves the track empty.
    if (!(std::isfinite(d.back()) && d.back() > 1e-6)) return;
    totalLength = d.back();
    int count   = std::max(16, int(totalLength / 1.0));
    step        = totalLength / count;
    size_t j    = 0;
    for (int i = 0; i < count; ++i)
    {
        double s = i * step;
        while (j + 1 < d.size() - 1 && d[j + 1] < s) j++;
        double t = (s - d[j]) / std::max(d[j + 1] - d[j], 1e-12);
        samples.push_back(dense[j] + (dense[(j + 1) % dense.size()] - dense[j]) * t);
        distances.push_back(s);
    }
    for (int c = 0; c < checkpointCount; ++c)
    {
        double s = totalLength * c / checkpointCount;
        gates.push_back({position(s), tangent(s), s});
    }
}

Vec3 Track::position(double s) const
{
    if (samples.empty()) return Vec3(0, 0, 0);
    s = std::fmod(s, totalLength);
    if (s < 0) s += totalLength;
    double f = s / step;
    int i    = int(f) % int(samples.size());
    int k    = (i + 1) % int(samples.size());
    double t = f - std::floor(f);
    return samples[i] + (samples[k] - samples[i]) * t;
}

Vec3 Track::tangent(double s) const
{
    Vec3 t = position(s + step) - position(s - step);
    return t.norm() > 1e-12 ? Vec3(t.normalized()) : Vec3(0, 0, -1);
}

double Track::curvature(double s) const
{
    double h = 5.0;
    Vec3 a = tangent(s - h), b = tangent(s + h);
    return std::abs(signedAngleXZ(a, b)) / (2 * h);
}

double Track::surfaceOffset(const Vec3& p, double& hint) const
{
    if (samples.empty()) return 0;
    bool hinted = hint >= 0;
    hint        = closest(p, hint);
    Vec3 c      = position(hint);
    double d    = std::hypot(p(0) - c(0), p(2) - c(2));
    double h    = width * 0.5;
    if (hinted && d > h)
    {
        // Regression: next to another section of the track (e.g. two parallel straights) the hinted local search
        // stayed on the curb of the first section while p was on the asphalt of the second. Off the asphalt the
        // global closest sample decides.
        double g  = closest(p);
        Vec3 cg   = position(g);
        double dg = std::hypot(p(0) - cg(0), p(2) - cg(2));
        if (dg < d)
        {
            hint = g;
            d    = dg;
        }
    }
    if (d <= h) return lift;
    if (d >= h + curbWidth) return 0;
    return lift * (1 - (d - h) / curbWidth);
}

double Track::closest(const Vec3& p, double hint) const
{
    if (samples.empty()) return 0;
    int n       = int(samples.size());
    auto search = [&](int from, int to, int& bi)
    {
        double best = std::numeric_limits<double>::infinity();
        for (int i = from; i < to; ++i)
        {
            int k    = (i % n + n) % n;
            double d = Vec3(p(0) - samples[k](0), 0, p(2) - samples[k](2)).squaredNorm();
            if (d < best)
            {
                best = d;
                bi   = i;
            }
        }
        return best;
    };
    int bi = 0;
    if (hint >= 0 && n > 80)
    {
        int c       = int(std::fmod(hint, totalLength) / step);
        double best = search(c - 40, c + 40, bi);
        // Regression: a stale hint (e.g. after a respawn far away) found a wrong sample at the edge of the window, so
        // the lifted asphalt was missed. A best sample at the window edge or off the track falls back to the global
        // search.
        double off = width * 0.5 + curbWidth;
        if (bi > c - 40 && bi < c + 39 && best <= off * off) return distances[(bi % n + n) % n];
    }
    search(0, n, bi);
    return distances[bi];
}

std::vector<Vec3> Track::loop(const Vec3& center, double radius, int points, uint32_t seed)
{
    std::vector<Vec3> c;
    for (int i = 0; i < points; ++i)
    {
        double a = 2 * kPi * i / points;
        double r = radius * (0.75 + 0.45 * hash01(seed * 7919u + uint32_t(i)));
        // An oval with a little variation
        c.push_back(center + Vec3(std::cos(a) * r * 1.3, 0, std::sin(a) * r * 0.8));
    }
    return c;
}

Vec3 Track::findSite(const GroundHeight& ground, double size, double radius, double minHeight, int points,
                     uint32_t seed, const std::function<double(const Vec3&)>& penalty)
{
    Vec3 best       = Vec3(size / 2, 0, size / 2);
    double bestCost = std::numeric_limits<double>::infinity();
    double margin   = radius * 1.8;
    for (double z = margin; z <= size - margin; z += radius * 0.5)
    {
        for (double x = margin; x <= size - margin; x += radius * 0.5)
        {
            Track t;
            t.build(loop(Vec3(x, 0, z), radius, points, seed), 1);
            double maxSlope = 0, sum = 0, sum2 = 0, extra = 0, prev = 0;
            bool valid = true;
            int n      = 0;
            for (double s = 0; s < t.length() && valid; s += 8, ++n)
            {
                Vec3 p   = t.position(s);
                double h = ground(p(0), p(2));
                valid    = h >= minHeight;
                if (n > 0) maxSlope = std::max(maxSlope, std::abs(h - prev) / 8);
                prev = h;
                sum += h;
                sum2 += h * h;
                if (penalty) extra += penalty(p);
            }
            if (!valid || n == 0) continue;
            double var  = sum2 / n - (sum / n) * (sum / n);
            double cost = maxSlope * 100 + std::sqrt(std::max(var, 0.0)) + extra;
            if (cost < bestCost)
            {
                bestCost = cost;
                best     = Vec3(x, 0, z);
            }
        }
    }
    return best;
}

}  // namespace Game
}  // namespace Saiga
