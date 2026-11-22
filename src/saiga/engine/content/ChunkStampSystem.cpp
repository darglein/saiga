/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// chunkStampSystem (docs/engine_design.md §12.1; S23a): phase 7, host. Reads Player, Transform, the plans and the chunk
// records; spawns the prefab entities (commands, R-SYS-2) with their component values (after configure / onSpawn, the
// world API's merge path) and SpaceLink targets, and records the stamped chunks.

#include "saiga/core/math/ChunkId.h"
#include "saiga/engine/api/ComponentJson.h"
#include "saiga/engine/components/Player.h"
#include "saiga/engine/components/SpaceLink.h"
#include "saiga/engine/components/Transform.h"
#include "saiga/engine/content/ChunkRecords.h"
#include "saiga/engine/content/ContentConvert.h"
#include "saiga/engine/content/GenerationPipeline.h"
#include "saiga/engine/content/Systems.h"
#include "saiga/engine/core/Log.h"
#include "saiga/engine/ecs/EntityLog.h"
#include "saiga/engine/world/World.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <set>

namespace Saiga
{
namespace Engine
{
namespace
{
// The SpaceLink of a prefab entity: "link" -> the instance of (this site, link) with the arrival in the target's
// coordinates; "parent" -> the creating space, the arrival relative to the creating site's origin, rotation and base
// height (exact)
bool resolveLink(const GenerationPipeline& g, SpaceId space, const Site& site, const PrefabSpaceLink& l, SpaceLink& out)
{
    if (!l.toParent)
    {
        for (size_t i = 0; i < g.spaceCount(); ++i)
        {
            const SpaceInstance& inst = g.space(SpaceId(i)).instance;
            if (inst.origin == SpaceInstance::FromSite && inst.parent == space && inst.siteKey == site.key &&
                inst.link == l.link)
            {
                out.target          = SpaceId(i);
                out.arrival         = contentPosition(l.x, l.y, l.z);
                out.arrivalRotation = contentRotation(l.arrivalYaw);
                return true;
            }
        }
        return false;
    }
    const SpaceInstance& self = g.space(space).instance;
    if (self.origin != SpaceInstance::FromSite) return false;
    const WorldPlan& parentPlan = *g.space(self.parent).plan;
    for (const Site& s : parentPlan.sites)
        if (s.key == self.siteKey)
        {
            int64_t rx = 0, rz = 0;
            rotateQuarter(l.x, l.z, s.quarterTurns, rx, rz);
            out.target          = self.parent;
            out.arrival         = contentPosition(s.x * kPosUnitsPerMetre + rx, int32_t(int64_t(s.baseHeight) + l.y),
                                                  s.z * kPosUnitsPerMetre + rz);
            out.arrivalRotation = contentRotation(uint16_t(uint32_t(l.arrivalYaw) + uint32_t(s.quarterTurns) * 16384u));
            return true;
        }
    return false;
}

// One stamp entry, resolved when the stamp is recorded
struct StampSpawn
{
    const ClassList::ClassInfo* info;
    Vec3 position;
    Quat rotation;
    std::shared_ptr<const json11::Json> components;
    std::optional<SpaceLink> link;
    std::string what;
};

void applyStampComponents(World& world, Entity e, const StampSpawn& s)
{
    ChunkRecords::Stats& stats = world.contentRecordsForUpdate().statsForUpdate();
    stats.entities++;
    for (const auto& kv : s.components->object_items())
    {
        const int found = world.registry().findComponent(kv.first);
        std::string error;
        if (!world.alive(e))
            error = "the entity was destroyed by its class";
        else if (found < 0)
            error = "unknown component";
        else if (mergeComponentValues(world, e, ComponentId(found), kv.second, error))
            continue;
        stats.failedComponents++;
        SAIGA_LOG_E("content", e, "stamp %s: %s not applied: %s", s.what.c_str(), kv.first.c_str(), error.c_str());
    }
    if (s.link && world.alive(e))
    {
        if (world.has<SpaceLink>(e))
            world.get<SpaceLink>(e) = *s.link;
        else
            world.add<SpaceLink>(e, *s.link);
    }
}

void stampChunk(World& w, const ChunkId& chunk, const std::vector<StampRecord>& list)
{
    const GenerationPipeline& g = w.generation();
    const SpaceGeneration& sp   = g.space(chunk.space);
    std::vector<StampSpawn> spawns;
    spawns.reserve(list.size());
    for (const StampRecord& r : list)
    {
        const Site& site           = sp.plan->sites[r.siteIndex];
        const PrefabEntity& entity = sp.prefabs[r.siteIndex]->entities[r.entity];
        StampSpawn s;
        s.info = w.classList().find(r.className);
        SAIGA_ASSERT(s.info, "chunk stamping: the class " + r.className + " was checked in phase B");
        s.position   = contentPosition(r.x, r.y, r.z);
        s.rotation   = contentRotation(r.yaw);
        s.components = std::make_shared<const json11::Json>(entity.components);
        s.what       = r.prefab + " entity " + std::to_string(r.entity);
        if (entity.spaceLink)
        {
            SpaceLink l;
            if (resolveLink(g, chunk.space, site, *entity.spaceLink, l))
                s.link = l;
            else
                SAIGA_LOG("content", "stamp: the space_link of %s (site %u) has no target space", r.className.c_str(),
                          r.siteKey);
        }
        spawns.push_back(std::move(s));
    }
    // One command for the whole chunk (everything applies at the same flush): the spawns run at once inside it, the
    // stamped entry after them. A spawn that throws (configure, onSpawn) rolls back the chunk's earlier spawns (without
    // onDestroy) before the exception propagates, so the chunk stays unstamped and is stamped whole later (PR review:
    // the stamped entry used to apply first and stayed with only part of the entities)
    const uint16_t current = sp.current;
    w.recordOrRun(
        [chunk, current, spawns = std::move(spawns)](World& world)
        {
            std::vector<Entity> spawned;
            spawned.reserve(spawns.size());
            try
            {
                // Applied at once (no command is recorded while commands are applied or between steps)
                for (const StampSpawn& s : spawns)
                    world.spawnByClassDeferred(*s.info, chunk.space, s.position, s.rotation,
                                               [&s, &spawned](World& ww, Entity x)
                                               {
                                                   spawned.push_back(x);
                                                   applyStampComponents(ww, x, s);
                                               });
            }
            catch (...)
            {
                ChunkRecords::Stats& stats = world.contentRecordsForUpdate().statsForUpdate();
                for (Entity e : spawned)
                {
                    world.rollbackSpawn(e);
                    stats.entities--;
                }
                throw;
            }
            world.contentRecordsForUpdate().markStamped(chunk, current);
        });
}
}  // namespace

void chunkStampSystem(World& w, double)
{
    const GenerationPipeline& g = w.generation();
    // The chunks within fullRadius of a Player (the chunk square intersects the disc), per space, in ChunkId order
    std::set<ChunkId> chunks;
    w.view<const Player, const Transform>().each(
        [&](Entity, const Player&, const Transform& t)
        {
            const SpaceId space = t.space();
            if (space >= g.spaceCount()) return;
            const SpaceGeneration& sp = g.space(space);
            if (sp.plan->sitesByChunk.empty()) return;
            const double r   = sp.settings.fullRadius;
            const Vec3 p     = t.position();
            const int64_t n  = sp.settings.sizeChunks;
            const int64_t x0 = std::max<int64_t>(0, int64_t(std::floor((p.x() - r) / 256)));
            const int64_t z0 = std::max<int64_t>(0, int64_t(std::floor((p.z() - r) / 256)));
            const int64_t x1 = std::min<int64_t>(n - 1, int64_t(std::floor((p.x() + r) / 256)));
            const int64_t z1 = std::min<int64_t>(n - 1, int64_t(std::floor((p.z() + r) / 256)));
            for (int64_t cz = z0; cz <= z1; ++cz)
                for (int64_t cx = x0; cx <= x1; ++cx)
                {
                    const double dx = std::max({double(cx) * 256 - p.x(), 0.0, p.x() - double(cx + 1) * 256});
                    const double dz = std::max({double(cz) * 256 - p.z(), 0.0, p.z() - double(cz + 1) * 256});
                    if (dx * dx + dz * dz > r * r) continue;
                    ChunkId c;
                    c.space = space;
                    c.x     = int32_t(cx);
                    c.z     = int32_t(cz);
                    if (sp.plan->sitesByChunk.count(c)) chunks.insert(c);
                }
        });
    const ChunkRecords& records = w.contentRecords();
    for (const ChunkId& c : chunks)
    {
        if (records.isStamped(c)) continue;
        const std::vector<StampRecord>& list = g.stampList(c);
        if (list.empty()) continue;
        stampChunk(w, c, list);
    }
}

}  // namespace Engine
}  // namespace Saiga
