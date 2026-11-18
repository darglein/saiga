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
    // add<Parent> rejects cycles, so a chain has fewer links than there are slots: the bound only guards a broken
    // registry (direct fills), it never cuts a valid chain
    const int max = int(w.registry().slotCount());
    int depth     = 0;
    while (const Parent* p = w.tryRead<Parent>(e))
    {
        depth++;
        SAIGA_ASSERT(depth < max, "Transform hierarchy: a Parent cycle");
        SAIGA_ASSERT(w.alive(p->entity), "Transform hierarchy: a Parent refers to a destroyed entity");
        if (depth >= max || !w.alive(p->entity)) break;
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

namespace
{
// The changed list: entities changed in this or the previous step, sorted by index
void buildChangedList(World& w)
{
    std::vector<uint32_t>& changed = w.changedTransformsForUpdate();
    changed.clear();
    w.viewAll<const Transform>().each(
        [&](Entity e, const Transform& t)
        {
            if (t.changedInThisStep() || t.changedInLastStep()) changed.push_back(e.index);
        });
}
}  // namespace

bool updateChildren(World& w, bool teleportsOnly)
{
    if (w.hierarchyDirty) rebuildHierarchy(w);

    bool any = false;
    for (Entity child : w.hierarchyOrder)
    {
        const Entity parentEntity = w.read<Parent>(child).entity;
        const Transform& parent   = w.read<Transform>(parentEntity);
        Transform& t              = w.get<Transform>(child);
        // A teleport of the parent in this step, or after the child's last update in the previous step (phase 8)
        const bool teleport = parent.wasTeleported() ||
                              std::binary_search(w.pendingChildTeleports.begin(), w.pendingChildTeleports.end(), child);
        // Teleports only: every child of a teleported parent is recomputed (a second teleport in the step reaches it)
        if (teleportsOnly && !teleport) continue;

        LocalTransform identity;
        const LocalTransform* local = w.tryRead<LocalTransform>(child);
        if (!local) local = &identity;
        const Vec3 position = parent.position() + parent.rotation() * parent.scale().cwiseProduct(local->position);
        const Quat rotation = parent.rotation() * local->rotation;
        const Vec3 scale    = parent.scale().cwiseProduct(local->scale);

        if (teleport)
            t.teleport(position, rotation);
        else
        {
            t.setPosition(position);
            t.setRotation(rotation);
        }
        t.setScale(scale);
        any = true;
    }
    return any;
}

void collectLateChildTeleports(World& w)
{
    // After phase 8: a child whose parent was teleported but whose pose does not match the parent's any more was not
    // updated after that teleport; it is teleported in the next phase 6
    w.pendingChildTeleports.clear();
    if (w.hierarchyDirty) rebuildHierarchy(w);
    for (Entity child : w.hierarchyOrder)
    {
        const Transform& parent = w.read<Transform>(w.read<Parent>(child).entity);
        if (!parent.wasTeleported()) continue;
        LocalTransform identity;
        const LocalTransform* local = w.tryRead<LocalTransform>(child);
        if (!local) local = &identity;
        const Vec3 position = parent.position() + parent.rotation() * parent.scale().cwiseProduct(local->position);
        if (w.read<Transform>(child).position() != position) w.pendingChildTeleports.push_back(child);
    }
    std::sort(w.pendingChildTeleports.begin(), w.pendingChildTeleports.end());
}

void propagateTeleportsBeforeExtract(World& w)
{
    // Teleports made in phase 7 (streaming) reach the children before extract reads them; the changed list is rebuilt
    // if a child was teleported
    if (updateChildren(w, true)) buildChangedList(w);
}

void transformSystem(World& w, double)
{
    updateChildren(w, false);
    w.pendingChildTeleports.clear();
    buildChangedList(w);
}

void registerEngineSystems(SystemList& systems)
{
    systems.add(Phase::Transform, "transform", transformSystem, RunOn::Both);
}

}  // namespace Engine
}  // namespace Saiga
