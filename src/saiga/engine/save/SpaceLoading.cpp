/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// spaceLoadSystem and spaceUnloadSystem (phase 7, host; S25) and the World's space unload and load
// (docs/engine_design.md §12.4): a streaming world's space without streaming centres stores its space-state entity in
// its space record once its roots are unloaded; it loads again when it gets a centre.

#include "saiga/engine/core/Log.h"
#include "saiga/engine/save/ChunkStore.h"
#include "saiga/engine/save/EntityGroup.h"
#include "saiga/engine/save/EntityUnloading.h"
#include "saiga/engine/save/GroupLoader.h"
#include "saiga/engine/save/Systems.h"
#include "saiga/engine/world/World.h"

namespace Saiga
{
namespace Engine
{
void spaceLoadSystem(World& world, double)
{
    world.entityUnloading().loadSpaces();
}

void spaceUnloadSystem(World& world, double)
{
    world.entityUnloading().unloadSpaces();
}

bool World::spaceLoaded(SpaceId space) const
{
    return !store->spaceUnloaded(space);
}

void World::unloadSpaceNow(SpaceId space)
{
    checkNotParallel("unloadSpaceNow");
    if (deferring())
    {
        record(makeCommand([space](World& w) { w.unloadSpaceNow(space); }));
        return;
    }
    if (space >= spaceCount() || store->spaceUnloaded(space)) return;
    const Entity e = spaceStateEntity(space);
    if (e.valid())
    {
        // A state entity with children (not a single-entity group) keeps its space loaded
        if (childCount(e) > 0) return;
        const uint32_t table = store->currentTable(*this);
        GroupWriteResult res;
        writeGroup(*this, e, *store->tables()[table], steps, res);
        StoredGroup g;
        g.schema = table;
        g.hash   = fnv1a64Bytes(res.bytes.data(), res.bytes.size());
        g.bytes  = std::make_shared<const std::vector<uint8_t>>(std::move(res.bytes));
        // The space record holds one group: a stored group that failed to reload earlier (kept, G14) is replaced by the
        // state the space had now, which superseded it (only groups[0] is ever loaded)
        std::vector<StoredGroup> previous;
        store->takeGroups(spaceRecordKey(space), previous);
        store->addGroup(spaceRecordKey(space), std::move(g));
        unloadRecords = true;
        try
        {
            destroyOne(e);
        }
        catch (...)
        {
            unloadRecords = false;
            throw;
        }
        unloadRecords      = false;
        spaceStates[space] = Entity{};
    }
    store->setSpaceUnloaded(space, true);
    store->statsForUpdate().spacesUnloaded++;
}

void World::loadSpaceNow(SpaceId space)
{
    checkNotParallel("loadSpaceNow");
    if (deferring())
    {
        record(makeCommand([space](World& w) { w.loadSpaceNow(space); }));
        return;
    }
    if (space >= spaceCount() || !store->spaceUnloaded(space)) return;
    std::vector<StoredGroup> groups;
    store->takeGroups(spaceRecordKey(space), groups);
    bool anySpace = false;
    for (size_t id = 0; id < kMaxComponents; ++id) anySpace = anySpace || reg.isSpaceState(ComponentId(id));
    Entity e;
    if (!groups.empty())
    {
        GroupLoader loader(*this);
        const StoredGroup& g                             = groups[0];
        const std::shared_ptr<const SchemaTable> current = store->tables()[store->currentTable(*this)];
        if (g.schema < store->tables().size() &&
            loader.stage(g, *store->tables()[g.schema], *current, spaceRecordKey(space), true))
        {
            std::vector<std::pair<Entity, double>> objects;
            loader.commit(objects);
            e = loader.root();
        }
        else
        {
            // The space loads with default state; the group stays stored (G14)
            SAIGA_LOG("save", "space %d: its stored state entity is not reloaded: %s", int(space),
                      loader.errorMessage().c_str());
            store->statsForUpdate().failedGroups++;
            store->putBack(spaceRecordKey(space), std::move(groups));
        }
    }
    if (anySpace)
    {
        if (!e.valid())
            e = makeStateEntity(Registry::kSpaceState, space, nullptr);
        else
            for (size_t id = 1; id < kMaxComponents; ++id)
                if (reg.isSpaceState(ComponentId(id)) && !reg.entityMask(e).test(ComponentId(id)))
                    addStateComponent(e, ComponentId(id), nullptr);
        if (spaceStates.size() < spaceCount()) spaceStates.resize(spaceCount());
        spaceStates[space] = e;
    }
    store->setSpaceUnloaded(space, false);
    store->statsForUpdate().spacesReloaded++;
}

}  // namespace Engine
}  // namespace Saiga
