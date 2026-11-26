/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// ContentCache (docs/engine_design.md §12.1, §17.2; S23a): the shared cache of generated content, one per setup
// (WorldSetup::sharedContent). Thread safe (one mutex; work runs outside the lock). Plans and prefabs are never
// evicted; ChunkBase and ChunkPlacement entries are evicted LRU by a byte budget (default 256 MB).
//
// Keys: the generator's registration identity (name, version(), slot table hash; a second identity under the same name
// on one cache stops the program), then plans by (name, pinned version, seed, template view hash, SpaceId), ChunkBase
// by (name, pinned version, plan hash, chunk) and ChunkPlacement by (name, placement version, plan hash, chunk). The
// point grids of a space's composed height (S20's terrain pyramid through TerrainHeightSource::cachedGrid; never
// evicted, like plans) by (name, pinned version, plan hash, SpaceId, samples per side, spacing).
//
// The in-flight rule (D8): a streaming-pool job registers its key as in flight only when it starts executing (a
// cancelled queued job leaves nothing behind); another pool job asking for an in-flight key waits for that generation
// (running jobs always finish); a synchronous requester (the main thread) never waits: on a miss it generates inline,
// also while a pool job generates the same key. The first insert wins; with SAIGA_ASSERTS a second result's hashes
// must equal the inserted one's.

#pragma once

#include "saiga/engine/content/ChunkContent.h"
#include "saiga/engine/content/PrefabStore.h"
#include "saiga/engine/content/WorldPlan.h"
#include "saiga/export.h"

#include <cstdint>
#include <functional>
#include <future>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>

namespace Saiga
{
namespace Engine
{
struct GeneratorIdentity
{
    std::string name;
    uint16_t version  = 0;
    uint64_t slotHash = 0;  // names and flags
};

class SAIGA_ENGINE_API ContentCache
{
   public:
    explicit ContentCache(size_t byteBudget = size_t(256) << 20);
    ContentCache(const ContentCache&)            = delete;
    ContentCache& operator=(const ContentCache&) = delete;

    PrefabStore& prefabs() { return prefabStore; }

    // The first world using the cache records each name's identity; another identity under a known name stops the
    // program (SAIGA_EXIT_ERROR): the cache would serve one generator's chunks to another
    void registerIdentity(const GeneratorIdentity& identity);

    using PlanKey = std::tuple<std::string, uint16_t, uint64_t, uint64_t, SpaceId>;
    // The plan of the key, built by make on a miss (outside the lock; the first insert wins)
    std::shared_ptr<const WorldPlan> plan(const PlanKey& key,
                                          const std::function<std::shared_ptr<const WorldPlan>()>& make);

    // A point grid of a space's composed height (the terrain pyramid), made on a miss (outside the lock; the first
    // insert wins)
    using GridKey = std::tuple<std::string, uint16_t, uint64_t, SpaceId, int, int64_t>;
    std::shared_ptr<const std::vector<int32_t>> grid(const GridKey& key,
                                                     const std::function<std::vector<int32_t>()>& make);

    using ChunkKey = std::tuple<std::string, uint16_t, uint64_t, ChunkId>;  // (generator, version, plan hash, chunk)
    // Synchronous: a hit, or generated inline (never a wait)
    std::shared_ptr<const ChunkBase> base(const ChunkKey& key, const std::function<ChunkBase()>& make);
    std::shared_ptr<const ChunkPlacement> placement(const ChunkKey& key, const std::function<ChunkPlacement()>& make);
    // A pool job's form: a hit; a running generation of the same key (waited for); or generated here with the key
    // registered as in flight while it runs
    std::shared_ptr<const ChunkPlacement> placementInJob(const ChunkKey& key,
                                                         const std::function<ChunkPlacement()>& make);
    // The same for bases (S24, G17: the residency's job generates the base, another job of the same chunk waits for it)
    std::shared_ptr<const ChunkBase> baseInJob(const ChunkKey& key, const std::function<ChunkBase()>& make);
    // A cached entry or null (tests)
    std::shared_ptr<const ChunkPlacement> findPlacement(const ChunkKey& key) const;
    std::shared_ptr<const ChunkBase> findBase(const ChunkKey& key) const;
    bool inFlight(const ChunkKey& key) const;

    struct Stats
    {
        uint64_t baseGenerations = 0, placementGenerations = 0, planBuilds = 0, gridBuilds = 0;
        uint64_t hits = 0, evictions = 0, droppedDuplicates = 0, waits = 0;
        size_t bytes = 0, entries = 0;
    };
    Stats stats() const;
    // Chunk generations (bases and placements): the stamp lists need none
    uint64_t generations() const;
    size_t byteBudget() const { return budget; }

   private:
    struct ChunkEntry
    {
        std::shared_ptr<const ChunkBase> base;
        std::shared_ptr<const ChunkPlacement> placement;
        size_t bytes = 0;
        std::list<std::pair<bool, ChunkKey>>::iterator lru;
    };
    void touch(ChunkEntry& e);
    void insertLocked(bool placementKind, const ChunkKey& key, ChunkEntry entry);
    void evictLocked();

    mutable std::mutex mutex;
    size_t budget = 0;
    PrefabStore prefabStore;
    std::map<std::string, GeneratorIdentity> identities;
    std::map<PlanKey, std::shared_ptr<const WorldPlan>> plans;
    std::map<GridKey, std::shared_ptr<const std::vector<int32_t>>> grids;
    std::map<ChunkKey, ChunkEntry> bases, placements;
    // Most recent first: (placement kind, key)
    std::list<std::pair<bool, ChunkKey>> lru;
    std::map<ChunkKey, std::shared_future<std::shared_ptr<const ChunkPlacement>>> running;
    std::map<ChunkKey, std::shared_future<std::shared_ptr<const ChunkBase>>> runningBases;
    Stats statistics;
};

}  // namespace Engine
}  // namespace Saiga
