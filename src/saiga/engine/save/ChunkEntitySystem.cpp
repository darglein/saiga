/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// chunkEntitySystem (phase 7, host; S25) and the World's unload and reload commands (docs/engine_design.md §9.6).

#include "saiga/engine/core/Log.h"
#include "saiga/engine/ecs/EntityLog.h"
#include "saiga/engine/gameplay/Behaviour.h"
#include "saiga/engine/save/ChunkStore.h"
#include "saiga/engine/save/EntityGroup.h"
#include "saiga/engine/save/EntityUnloading.h"
#include "saiga/engine/save/GroupLoader.h"
#include "saiga/engine/save/Systems.h"
#include "saiga/engine/streaming/ChunkResidency.h"
#include "saiga/engine/world/World.h"

namespace Saiga
{
namespace Engine
{
void chunkEntitySystem(World& world, double)
{
    world.entityUnloading().chunkPass();
}

void World::setLoadedTransform(Entity e, const Transform& t)
{
    Transform& d = reg.get<Transform>(e);
    d            = t;
    // A new entity at its pose: no interpolation from an earlier one
    d.changedThisStep = 1;
    d.teleported      = 1;
    d.teleports++;
}

void World::unloadNow(Entity root, const ChunkId& chunk)
{
    checkNotParallel("unloadNow");
    if (deferring())
    {
        record(makeCommand([root, chunk](World& w) { w.unloadNow(root, chunk); }));
        return;
    }
    // Re-checked at the flush: still an unloadable root on a chunk that is not kept (its position now)
    if (!reg.alive(root) || !unloading->unloadable(root)) return;
    const Transform& t = reg.read<Transform>(root);
    const ChunkId into = chunkAt(t.space(), t.position().x(), t.position().z());
    if (config.streaming && residencyPtr->kept(into)) return;
    ChunkStore::Stats& stats = store->statsForUpdate();
    const uint32_t table     = store->currentTable(*this);
    GroupWriteResult res;
    writeGroup(*this, root, *store->tables()[table], steps, res);
    StoredGroup g;
    g.schema = table;
    g.hash   = fnv1a64Bytes(res.bytes.data(), res.bytes.size());
    g.bytes  = std::make_shared<const std::vector<uint8_t>>(std::move(res.bytes));
    store->addGroup(into, std::move(g));
    stats.groupsUnloaded++;
    stats.entitiesUnloaded += res.entities.size();
    if (res.droppedReferences > 0)
    {
        stats.droppedReferences += res.droppedReferences;
        SAIGA_LOG_E("save", root,
                    "unloaded into chunk (%d, %d, %d): %llu Entity references out of its group became null",
                    int(into.space), into.x, into.z, (unsigned long long)res.droppedReferences);
    }
    // Removed like destroy, without onDestroy: the structural log's records are unloaded
    for (Entity x : res.entities)
        if (Behaviour* b = reg.tryGet<Behaviour>(x)) b->destroying = true;
    unloadRecords = true;
    try
    {
        destroyOne(root);
    }
    catch (...)
    {
        unloadRecords = false;
        throw;
    }
    unloadRecords = false;
    (void)chunk;
}

void World::reloadChunkNow(const ChunkId& chunk)
{
    checkNotParallel("reloadChunkNow");
    if (deferring())
    {
        record(makeCommand([chunk](World& w) { w.reloadChunkNow(chunk); }));
        return;
    }
    std::vector<StoredGroup> groups;
    // A group only on disk (a failed read): kept for a later reload
    if (!store->takeGroups(chunk, groups) || groups.empty()) return;
    const std::shared_ptr<const SchemaTable> current = store->tables()[store->currentTable(*this)];
    ChunkStore::Stats& stats                         = store->statsForUpdate();
    std::vector<StoredGroup> failed;
    std::vector<std::pair<Entity, double>> objects;
    for (size_t k = 0; k < groups.size(); ++k)
    {
        const StoredGroup& g = groups[k];
        GroupLoader loader(*this);
        const bool ok =
            g.schema < store->tables().size() && loader.stage(g, *store->tables()[g.schema], *current, chunk, false);
        if (!ok)
        {
            // Kept (G14: a game patch may add the missing migration); retried at the next reload
            SAIGA_LOG(
                "save", "chunk (%d, %d, %d): stored group %zu is not reloaded: %s", int(chunk.space), chunk.x, chunk.z,
                k, g.schema < store->tables().size() ? loader.errorMessage().c_str() : "its schema table is unknown");
            stats.failedGroups++;
            failed.push_back(g);
            continue;
        }
        loader.commit(objects);
        stats.groupsReloaded++;
        stats.entitiesReloaded += loader.createdCount();
    }
    store->putBack(chunk, std::move(failed));
    startReloadedObjects(objects);
}

void World::startReloadedObjects(const std::vector<std::pair<Entity, double>>& objects)
{
    if (objects.empty()) return;
    ChunkStore::Stats& stats = store->statsForUpdate();
    auto failure             = [&](Entity e, const std::string& message)
    {
        SAIGA_LOG_E("save", e, "reload: %s (removed with its descendants)", message.c_str());
        stats.reloadHookErrors++;
        const int phaseBefore = loadHookPhase;
        loadHookPhase         = 0;
        abandonSpawn(e);
        loadHookPhase = phaseBefore;
    };
    // Pass 1: every object exists before any hook runs
    for (const auto& [e, elapsed] : objects)
    {
        if (!reg.alive(e)) continue;
        Behaviour& b                  = reg.get<Behaviour>(e);
        const ClassList::ClassInfo* c = classes.find(b.classId);
        try
        {
            b.object        = c->factory();
            b.object->owner = this;
            b.object->self  = e;
        }
        catch (const std::exception& x)
        {
            failure(e, std::string("the factory threw: ") + x.what());
        }
    }
    // Pass 2 configure, pass 3 onLoad (the load rules), pass 4 onCatchUp(elapsed) (like onSpawn: structural changes
    // apply at once)
    for (int pass = 1; pass <= 3; ++pass)
    {
        loadHookPhase = pass <= 2 ? pass : 0;
        for (const auto& [e, elapsed] : objects)
        {
            if (!reg.alive(e) || !reg.has<Behaviour>(e) || !reg.read<Behaviour>(e).object) continue;
            GameObject* o  = reg.get<Behaviour>(e).object.get();
            loadHookEntity = e;
            hookDepth++;
            std::string error;
            try
            {
                if (pass == 1)
                    o->configure();
                else if (pass == 2)
                    o->onLoad();
                else
                    o->onCatchUp(elapsed);
            }
            catch (const std::exception& x)
            {
                error = x.what();
                if (error.empty()) error = "an exception";
            }
            catch (...)
            {
                error = "an exception";
            }
            hookDepth--;
            if (!error.empty())
                failure(e, std::string(pass == 1 ? "configure" : (pass == 2 ? "onLoad" : "onCatchUp")) +
                               " threw: " + error);
        }
    }
    loadHookPhase  = 0;
    loadHookEntity = Entity{};
    releaseRetired();
}

}  // namespace Engine
}  // namespace Saiga
