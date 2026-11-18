/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// transformSystem (phase 6): the Transform hierarchy and the list of changed transforms; the engine's system list.

#include "saiga/engine/world/TransformSystem.h"

#include "saiga/engine/components/LocalTransform.h"
#include "saiga/engine/components/Parent.h"
#include "saiga/engine/world/World.h"

#include <algorithm>
#include <utility>

namespace Saiga
{
namespace Engine
{
namespace
{
// Depth of a child: number of Parent links up to a root
int depthOf(World& w, Entity e)
{
    int depth = 0;
    while (const Parent* p = w.tryRead<Parent>(e))
    {
        depth++;
        SAIGA_ASSERT(depth < 1000, "Transform hierarchy: a Parent cycle (or deeper than 1000 levels)");
        SAIGA_ASSERT(w.alive(p->entity), "Transform hierarchy: a Parent refers to a destroyed entity");
        e = p->entity;
    }
    return depth;
}

void rebuildHierarchy(World& w)
{
    std::vector<std::pair<int, Entity>> children;
    w.viewAll<const Parent>().each([&](Entity e, const Parent&) { children.push_back({0, e}); });
    for (auto& c : children) c.first = depthOf(w, c.second);
    // Parents before children; inside a depth by index (deterministic)
    std::sort(children.begin(), children.end(), [](const std::pair<int, Entity>& a, const std::pair<int, Entity>& b)
              { return a.first != b.first ? a.first < b.first : a.second.index < b.second.index; });
    w.hierarchyOrder.clear();
    for (auto& c : children) w.hierarchyOrder.push_back(c.second);
    w.hierarchyDirty = false;
}
}  // namespace

void transformSystem(World& w, double)
{
    if (w.hierarchyDirty) rebuildHierarchy(w);

    for (Entity child : w.hierarchyOrder)
    {
        const Entity parentEntity = w.read<Parent>(child).entity;
        const Transform& parent   = w.read<Transform>(parentEntity);
        LocalTransform identity;
        const LocalTransform* local = w.tryRead<LocalTransform>(child);
        if (!local) local = &identity;

        const Vec3 position = parent.position() + parent.rotation() * parent.scale().cwiseProduct(local->position);
        const Quat rotation = parent.rotation() * local->rotation;
        const Vec3 scale    = parent.scale().cwiseProduct(local->scale);

        Transform& t = w.get<Transform>(child);
        if (parent.wasTeleported())
            t.teleport(position, rotation);
        else
        {
            t.setPosition(position);
            t.setRotation(rotation);
        }
        t.setScale(scale);
    }

    // The entities changed in this or the previous step (changes of phases 7 and 8 of the last step), sorted by index
    std::vector<uint32_t>& changed = w.changedTransformsForUpdate();
    changed.clear();
    w.viewAll<const Transform>().each(
        [&](Entity e, const Transform& t)
        {
            if (t.changedInThisStep() || t.changedInLastStep()) changed.push_back(e.index);
        });
}

void registerEngineSystems(SystemList& systems)
{
    systems.add(Phase::Transform, "transform", transformSystem, RunOn::Both);
}

}  // namespace Engine
}  // namespace Saiga
