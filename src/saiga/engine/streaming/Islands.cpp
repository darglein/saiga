/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/streaming/Islands.h"

#include "saiga/engine/components/PendingTransition.h"
#include "saiga/engine/components/Player.h"
#include "saiga/engine/world/World.h"

#include <algorithm>
#include <numeric>
#include <vector>

namespace Saiga
{
namespace Engine
{
namespace
{
struct Root
{
    Entity entity;
    SpaceId space = 0;
    Vec3 position = Vec3::Zero();
};

// atTargets: a root with a pending arrival counts at its target (the hard limit must hold for accepted arrivals, which
// complete in any order)
std::vector<Root> playerRoots(const World& w, bool atTargets)
{
    std::vector<Root> roots;
    Registry& reg = const_cast<Registry&>(w.registry());
    reg.view<const Player>().each(
        [&](Entity e, const Player&)
        {
            const Entity r = w.rootOf(e);
            for (const Root& x : roots)
                if (x.entity == r) return;
            if (atTargets && reg.has<PendingTransition>(r))
            {
                const PendingTransition& p = reg.read<PendingTransition>(r);
                roots.push_back({r, p.target, p.position});
                return;
            }
            const Transform& t = reg.read<Transform>(r);
            roots.push_back({r, t.space(), t.position()});
        });
    std::sort(roots.begin(), roots.end(), [](const Root& a, const Root& b) { return a.entity.index < b.entity.index; });
    return roots;
}

int components(const World& w, const std::vector<Root>& roots)
{
    std::vector<size_t> parent(roots.size());
    std::iota(parent.begin(), parent.end(), size_t(0));
    auto find = [&](size_t i)
    {
        while (parent[i] != i) i = parent[i] = parent[parent[i]];
        return i;
    };
    for (size_t i = 0; i < roots.size(); ++i)
        for (size_t j = i + 1; j < roots.size(); ++j)
        {
            if (roots[i].space != roots[j].space || roots[i].space >= w.spaceCount()) continue;
            const double r  = 2 * w.spaceTemplate(roots[i].space).simulationRadius;
            const double dx = roots[i].position.x() - roots[j].position.x();
            const double dz = roots[i].position.z() - roots[j].position.z();
            if (dx * dx + dz * dz <= r * r) parent[find(i)] = find(j);
        }
    int n = 0;
    for (size_t i = 0; i < roots.size(); ++i) n += find(i) == i ? 1 : 0;
    return n;
}
}  // namespace

int Islands::count(const World& world)
{
    return components(world, playerRoots(world, false));
}

int Islands::countAccepted(const World& world)
{
    return components(world, playerRoots(world, true));
}

int Islands::countWith(const World& world, Entity moved, SpaceId space, const Vec3& position)
{
    std::vector<Root> roots = playerRoots(world, true);
    bool found              = false;
    for (Root& r : roots)
        if (r.entity == moved)
        {
            r.space    = space;
            r.position = position;
            found      = true;
        }
    if (!found) roots.push_back({moved, space, position});
    return components(world, roots);
}

}  // namespace Engine
}  // namespace Saiga
