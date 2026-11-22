/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/content/ContentCache.h"

#include "saiga/core/util/assert.h"

namespace Saiga
{
namespace Engine
{
namespace
{
size_t bytesOf(const ChunkBase& b)
{
    return sizeof(ChunkBase) + (b.heights ? b.heights->size() * sizeof(int32_t) : 0);
}
size_t bytesOf(const ChunkPlacement& p)
{
    return sizeof(ChunkPlacement) + p.instances.size() * sizeof(TierOneInstance) + p.grass.density.size();
}
}  // namespace

ContentCache::ContentCache(size_t byteBudget) : budget(byteBudget) {}

void ContentCache::registerIdentity(const GeneratorIdentity& identity)
{
    std::lock_guard<std::mutex> lock(mutex);
    auto it = identities.find(identity.name);
    if (it == identities.end())
    {
        identities[identity.name] = identity;
        return;
    }
    if (it->second.version != identity.version || it->second.slotHash != identity.slotHash)
        SAIGA_EXIT_ERROR("ContentCache: generator '" + identity.name + "' is registered with version " +
                         std::to_string(identity.version) + " and another slot table or version (" +
                         std::to_string(it->second.version) +
                         ") is already using this cache; worlds of different generator builds need their own cache "
                         "(WorldSetup::sharedContent)");
}

std::shared_ptr<const WorldPlan> ContentCache::plan(const PlanKey& key,
                                                    const std::function<std::shared_ptr<const WorldPlan>()>& make)
{
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = plans.find(key);
        if (it != plans.end()) return it->second;
    }
    std::shared_ptr<const WorldPlan> p = make();
    std::lock_guard<std::mutex> lock(mutex);
    statistics.planBuilds++;
    return plans.emplace(key, std::move(p)).first->second;
}

void ContentCache::touch(ChunkEntry& e)
{
    lru.splice(lru.begin(), lru, e.lru);
}

void ContentCache::insertLocked(bool placementKind, const ChunkKey& key, ChunkEntry entry)
{
    lru.push_front({placementKind, key});
    entry.lru = lru.begin();
    statistics.bytes += entry.bytes;
    (placementKind ? placements : bases).emplace(key, std::move(entry));
    evictLocked();
}

void ContentCache::evictLocked()
{
    // Keep at least the newest entry
    while (statistics.bytes > budget && lru.size() > 1)
    {
        const auto [kind, key] = lru.back();
        auto& map              = kind ? placements : bases;
        auto it                = map.find(key);
        if (it != map.end())
        {
            statistics.bytes -= it->second.bytes;
            map.erase(it);
        }
        lru.pop_back();
        statistics.evictions++;
    }
}

std::shared_ptr<const ChunkBase> ContentCache::base(const ChunkKey& key, const std::function<ChunkBase()>& make)
{
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = bases.find(key);
        if (it != bases.end())
        {
            statistics.hits++;
            touch(it->second);
            return it->second.base;
        }
    }
    auto b = std::make_shared<const ChunkBase>(make());
    std::lock_guard<std::mutex> lock(mutex);
    statistics.baseGenerations++;
    auto it = bases.find(key);
    if (it != bases.end())
    {
        // The first insert wins
        SAIGA_ASSERT(it->second.base->heightsHash == b->heightsHash, "ContentCache: two generations of one key differ");
        statistics.droppedDuplicates++;
        return it->second.base;
    }
    ChunkEntry e;
    e.base  = b;
    e.bytes = bytesOf(*b);
    insertLocked(false, key, std::move(e));
    return b;
}

std::shared_ptr<const ChunkPlacement> ContentCache::placement(const ChunkKey& key,
                                                              const std::function<ChunkPlacement()>& make)
{
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = placements.find(key);
        if (it != placements.end())
        {
            statistics.hits++;
            touch(it->second);
            return it->second.placement;
        }
    }
    auto p = std::make_shared<const ChunkPlacement>(make());
    std::lock_guard<std::mutex> lock(mutex);
    statistics.placementGenerations++;
    auto it = placements.find(key);
    if (it != placements.end())
    {
        SAIGA_ASSERT(it->second.placement->hashes.total == p->hashes.total,
                     "ContentCache: two generations of one key differ");
        statistics.droppedDuplicates++;
        return it->second.placement;
    }
    ChunkEntry e;
    e.placement = p;
    e.bytes     = bytesOf(*p);
    insertLocked(true, key, std::move(e));
    return p;
}

std::shared_ptr<const ChunkPlacement> ContentCache::placementInJob(const ChunkKey& key,
                                                                   const std::function<ChunkPlacement()>& make)
{
    std::promise<std::shared_ptr<const ChunkPlacement>> promise;
    {
        std::unique_lock<std::mutex> lock(mutex);
        auto it = placements.find(key);
        if (it != placements.end())
        {
            statistics.hits++;
            touch(it->second);
            return it->second.placement;
        }
        auto r = running.find(key);
        if (r != running.end())
        {
            // Attach to the running generation
            std::shared_future<std::shared_ptr<const ChunkPlacement>> f = r->second;
            statistics.waits++;
            lock.unlock();
            return f.get();
        }
        // In flight from now on (the job is executing)
        running.emplace(key, promise.get_future().share());
    }
    std::shared_ptr<const ChunkPlacement> result;
    try
    {
        result = placement(key, make);
    }
    catch (...)
    {
        std::lock_guard<std::mutex> lock(mutex);
        running.erase(key);
        promise.set_exception(std::current_exception());
        throw;
    }
    {
        std::lock_guard<std::mutex> lock(mutex);
        running.erase(key);
    }
    promise.set_value(result);
    return result;
}

std::shared_ptr<const ChunkPlacement> ContentCache::findPlacement(const ChunkKey& key) const
{
    std::lock_guard<std::mutex> lock(mutex);
    auto it = placements.find(key);
    return it == placements.end() ? nullptr : it->second.placement;
}

std::shared_ptr<const ChunkBase> ContentCache::findBase(const ChunkKey& key) const
{
    std::lock_guard<std::mutex> lock(mutex);
    auto it = bases.find(key);
    return it == bases.end() ? nullptr : it->second.base;
}

bool ContentCache::inFlight(const ChunkKey& key) const
{
    std::lock_guard<std::mutex> lock(mutex);
    return running.count(key) != 0;
}

ContentCache::Stats ContentCache::stats() const
{
    std::lock_guard<std::mutex> lock(mutex);
    Stats s   = statistics;
    s.entries = bases.size() + placements.size();
    return s;
}

uint64_t ContentCache::generations() const
{
    std::lock_guard<std::mutex> lock(mutex);
    return statistics.baseGenerations + statistics.placementGenerations;
}

}  // namespace Engine
}  // namespace Saiga
