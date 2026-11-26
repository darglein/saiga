/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/save/EntityUnloading.h"

#include "saiga/engine/components/Parent.h"
#include "saiga/engine/components/PendingTransition.h"
#include "saiga/engine/components/Player.h"
#include "saiga/engine/content/ChunkRecords.h"
#include "saiga/engine/content/GenerationPipeline.h"
#include "saiga/engine/content/GeneratorList.h"
#include "saiga/engine/core/Log.h"
#include "saiga/engine/ecs/EntityLog.h"
#include "saiga/engine/net/NetOwner.h"
#include "saiga/engine/save/ChunkStore.h"
#include "saiga/engine/save/EntityGroup.h"
#include "saiga/engine/save/SaveDirectory.h"
#include "saiga/engine/streaming/ChunkResidency.h"
#include "saiga/engine/world/World.h"

#include <algorithm>

namespace Saiga
{
namespace Engine
{
namespace
{
constexpr uint8_t kStateFlags = Registry::kWorldState | Registry::kSpaceState;
// The cells of a chunk per axis (256 / 32)
constexpr int32_t kCellsPerChunk = 8;
}  // namespace

EntityUnloading::EntityUnloading(World& world) : w(world) {}

EntityUnloading::~EntityUnloading()
{
    if (consumer >= 0) w.structuralLog().removeConsumer(consumer);
    if (listener >= 0) w.residency().removeListener(listener);
    if (hook >= 0) w.removeStepStartHook(hook);
}

void EntityUnloading::attach()
{
    if (consumer >= 0) return;
    consumer = w.structuralLog().addConsumer();
    ResidencyListener l;
    l.onReadyChanged = [this](const ChunkId& c, bool ready)
    {
        if (ready)
            readyChunks.insert(c);
        else
            readyChunks.erase(c);
    };
    l.onKeptChanged = [this](const ChunkId& c, bool kept)
    {
        if (!kept)
            keptLost.insert(c);
        else
            keptLost.erase(c);
    };
    l.onResident = [this](const ChunkId& c) { residentChunks.insert(c); };
    // The one rule of when bytes leave memory (Scope 3): a chunk that leaves drops the bytes of its groups that are in
    // the current record
    l.onUnload = [this](const ChunkId& c) { w.chunkStore().dropBytes(c); };
    listener   = w.residency().addListener(std::move(l));
    // PartStored: a chunk whose groups are only on disk becomes Resident with them read
    ResidencyPartInfo part;
    part.bit    = PartStored;
    part.wanted = [this](const ChunkId& c) { return w.chunkStoreView().needsRead(c); };
    part.build  = [this](const ChunkId& c)
    {
        if (SaveDirectory* d = w.saveDirectory()) d->readStoredSync(c);
    };
    part.request = [this](const ChunkId& c, int priority, std::function<void()> done)
    {
        if (SaveDirectory* d = w.saveDirectory())
            d->requestStoredRead(c, priority, std::move(done));
        else
            done();
    };
    w.residency().addPart(std::move(part));
    hook = w.addStepStartHook([this]() { stepStart(); });
}

void EntityUnloading::onLoad()
{
    readyChunks.clear();
    keptLost.clear();
    residentChunks.clear();
    candidates.clear();
    largeRoots.clear();
    if (consumer >= 0) w.structuralLog().consume(consumer, [](const StructuralLog::Record&) {});
    fullUnload = consumer >= 0;
    fullReload = consumer >= 0;
}

bool EntityUnloading::unloadable(Entity root) const
{
    const Registry& reg = w.registry();
    if (!reg.alive(root) || (reg.flags(root) & kStateFlags) || reg.has<Parent>(root)) return false;
    // Client-owned groups stay (G16): their owner simulates them
    if (w.ownerOf(root) != 0) return false;
    if (w.isPlayerRoot(root)) return false;
    if (reg.has<PendingTransition>(root)) return false;
    if (!w.hasChildren(root)) return true;
    const std::vector<Entity> sub = w.subtree(root);
    if (sub.size() > kMaxGroupEntities) return false;
    for (Entity x : sub)
        if (reg.has<PendingTransition>(x)) return false;
    return true;
}

void EntityUnloading::consumeLog()
{
    if (consumer < 0) return;
    w.structuralLog().consume(consumer,
                              [&](const StructuralLog::Record& r)
                              {
                                  const ComponentId c = r.component;
                                  const bool created  = c == Transform::id && r.added;
                                  // An exclusion that ended: a removed Player, PendingTransition, NetOwner or Parent
                                  const bool released = !r.added && (c == Player::id || c == PendingTransition::id ||
                                                                     c == NetOwner::id || c == Parent::id);
                                  if (created || released) candidates.insert(r.entity.index);
                              });
}

std::vector<ChunkId> EntityUnloading::reloadList() const
{
    const ChunkStore& store = w.chunkStoreView();
    std::set<ChunkId> chunks;
    for (const ChunkId& c : readyChunks)
        if (store.hasGroups(c)) chunks.insert(c);
    if (fullReload)
        for (const auto& [c, e] : store.chunks())
            if (!e.groups.empty() && store.hasGroups(c)) chunks.insert(c);
    std::vector<ChunkId> out;
    for (const ChunkId& c : chunks)
        if (w.residency().ready(c)) out.push_back(c);
    return out;
}

std::vector<ChunkId> EntityUnloading::recordList() const
{
    std::set<ChunkId> chunks = residentChunks;
    if (fullReload)
        for (const ChunkStatus& s : w.residency().chunkStatuses())
            if (s.state == ChunkState::Resident) chunks.insert(s.chunk);
    std::vector<ChunkId> out;
    for (const ChunkId& c : chunks)
    {
        if (c.space >= w.spaceCount() || w.chunkRecord(c) || w.residency().state(c) != ChunkState::Resident) continue;
        // The spaces whose chunks can have records (the load's condition): a generator with slots or a plan with sites
        const SpaceGeneration& g      = w.generation().space(c.space);
        const GeneratorList::Entry* e = w.generators().find(g.generatorId);
        if (g.plan->sites.empty() && (!e || e->slots.empty())) continue;
        out.push_back(c);
    }
    return out;
}

void EntityUnloading::writeRecord(const ChunkId& c)
{
    // Scope 12: {current placement version, no tombstones}; the effective record does not change, so no
    // onRecordChanged (nothing derived changes)
    if (w.chunkRecord(c)) return;
    w.contentRecordsForUpdate().reserve(c, w.effectivePlacementVersion(c), 0);
    w.chunkStore().markDirty(c);
    w.chunkStore().statsForUpdate().autoRecords++;
}

bool EntityUnloading::spaceHasCentre(SpaceId space) const
{
    for (const StreamingCentre& c : w.residency().centres())
        if (c.space == space) return true;
    return false;
}

bool EntityUnloading::spaceHasRoot(SpaceId space) const
{
    // A scan, only while a loaded space has no centre (transitional: its roots unload in the same step)
    const Registry& reg = w.registry();
    for (uint32_t i = 1; i < reg.slotCount(); ++i)
    {
        const Entity e = reg.aliveAt(i);
        if (!e.valid() || (reg.flags(e) & kStateFlags) || reg.has<Parent>(e)) continue;
        if (reg.read<Transform>(e).space() == space) return true;
    }
    return false;
}

void EntityUnloading::stepStart()
{
    if (!w.isHost() || consumer < 0) return;
    for (SpaceId s : std::set<SpaceId>(w.chunkStoreView().unloadedSpaces()))
        if (spaceHasCentre(s)) w.loadSpaceNow(s);
    for (const ChunkId& c : reloadList()) w.reloadChunkNow(c);
    for (const ChunkId& c : recordList()) writeRecord(c);
    readyChunks.clear();
    residentChunks.clear();
    fullReload = false;
}

void EntityUnloading::loadSpaces()
{
    if (consumer < 0) return;
    for (SpaceId s : w.chunkStoreView().unloadedSpaces())
        if (spaceHasCentre(s)) w.recordOrRun([s](World& world) { world.loadSpaceNow(s); });
}

void EntityUnloading::chunkPass()
{
    if (consumer < 0) return;
    // 1. The reloads of the chunks that became ready inside the step and the records of the chunks that became
    // resident (commands: applied at the flush)
    for (const ChunkId& c : reloadList()) w.recordOrRun([c](World& world) { world.reloadChunkNow(c); });
    for (const ChunkId& c : recordList()) w.recordOrRun([this, c](World&) { writeRecord(c); });
    // 2. The unload pass over what changed
    consumeLog();
    const Registry& reg        = w.registry();
    std::set<uint32_t> examine = candidates;
    for (uint32_t i : w.changedTransforms()) examine.insert(i);
    for (uint32_t i : largeRoots) examine.insert(i);
    const SpatialGrid& grid = w.spatialGrid();
    for (const ChunkId& c : keptLost)
        for (int32_t z = 0; z < kCellsPerChunk; ++z)
            for (int32_t x = 0; x < kCellsPerChunk; ++x)
                grid.forEachInCell(c.space, c.x * kCellsPerChunk + x, c.z * kCellsPerChunk + z,
                                   [&](const SpatialGrid::Item& it) { examine.insert(it.entity.index); });
    if (fullUnload)
        for (uint32_t i = 1; i < reg.slotCount(); ++i)
            if (reg.aliveAt(i).valid()) examine.insert(i);
    std::set<uint32_t> roots;
    for (uint32_t i : examine)
    {
        const Entity e = reg.aliveAt(i);
        if (e.valid()) roots.insert(w.rootOf(e).index);
    }
    ChunkStore::Stats& stats = w.chunkStore().statsForUpdate();
    std::set<uint32_t> large;
    for (uint32_t i : roots)
    {
        const Entity r = reg.aliveAt(i);
        if (!r.valid() || (reg.flags(r) & kStateFlags)) continue;
        stats.unloadCandidates++;
        const Transform& t = reg.read<Transform>(r);
        if (t.space() >= w.spaceCount()) continue;
        const ChunkId chunk = w.chunkAt(t.space(), t.position().x(), t.position().z());
        if (w.residency().kept(chunk)) continue;
        if (!unloadable(r))
        {
            if (w.hasChildren(r) && w.subtree(r).size() > kMaxGroupEntities)
            {
                if (!largeRoots.count(i))
                    SAIGA_LOG_E("save", r, "not unloaded: its subtree has more than %u entities", kMaxGroupEntities);
                large.insert(i);
            }
            continue;
        }
        if (w.residency().state(chunk) == ChunkState::None) stats.farUnloads++;
        w.unloadNow(r, chunk);
    }
    // 3. The accumulated state is consumed at the flush (an aborted phase keeps it for the next pass)
    w.recordOrRun(
        [this, large](World&)
        {
            readyChunks.clear();
            residentChunks.clear();
            keptLost.clear();
            candidates.clear();
            largeRoots = large;
            fullReload = false;
            fullUnload = false;
        });
}

void EntityUnloading::unloadSpaces()
{
    if (consumer < 0) return;
    for (size_t i = 0; i < w.spaceCount(); ++i)
    {
        const SpaceId s = SpaceId(i);
        if (w.chunkStoreView().spaceUnloaded(s) || spaceHasCentre(s) || spaceHasRoot(s)) continue;
        w.recordOrRun([s](World& world) { world.unloadSpaceNow(s); });
    }
}

}  // namespace Engine
}  // namespace Saiga
