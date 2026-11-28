/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// GenerationPipeline (docs/engine_design.md §12.1, §15.3; S23a): one per World; it owns the generator list, the pinned
// versions and every space instance's plan, all computed eagerly at construction and immutable afterwards.
//
// Phase A (build, pure, no World; first in the World constructor, also used by checkContentHashes and a load with the
// saved pins): the generators and their registration checks, the template views, the prefab files of every plan, the
// space list (the overworld, the generator's listed instances, the sites' instances, one per unreferenced template),
// every plan with its validation, sitesByChunk and the plan hashes. Phase B (validateEntities, last in the World
// constructor after renderAssets): the prefab entities' classes, component values and RenderInstance meshes.
//
// height() composes the sites' terrain patches into the generator's natural height (a cheap, lock-free point query,
// safe in parallelEach); chunkBase / chunkPlacement generate synchronously through the ContentCache (a hit or inline
// generation, never a wait); contentSource gives pool jobs a copyable value (never the World, D8); stampList /
// staticList come from the plan alone (no chunk generation).

#pragma once

#include "saiga/core/math/TerrainSurface.h"
#include "saiga/engine/content/ChunkContent.h"
#include "saiga/engine/content/ContentCache.h"
#include "saiga/engine/content/GeneratorList.h"
#include "saiga/engine/content/Prefab.h"
#include "saiga/engine/content/gen/SitePatch.h"
#include "saiga/engine/world/SpaceTemplate.h"
#include "saiga/export.h"

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace Saiga
{
namespace Engine
{
struct WorldSetup;
class World;
class StreamingPool;

// The versions of a build: by generator name the pinned (plan / terrain) version and the current placement version;
// a missing name takes the generator's version()
struct GenerationPins
{
    std::map<std::string, uint16_t> pinned, current;
};

// One space instance of the world with everything generation reads (immutable; pool jobs hold it by shared_ptr)
struct SpaceGeneration
{
    SpaceId id = 0;
    SpaceInstance instance;
    SpaceTemplate settings;  // the instance's template
    GeneratorId generatorId = 0;
    std::shared_ptr<const Generator> generator;
    std::string generatorName;
    uint16_t pinned = 1, current = 1;
    std::shared_ptr<const WorldPlan> plan;
    std::vector<std::shared_ptr<const Prefab>> prefabs;           // per site
    std::vector<std::shared_ptr<const SiteContent>> siteContent;  // per site
    uint64_t planHash = 0;
    bool hasTerrain   = false;
    bool openWorld    = true;
};

struct GenerationPlans
{
    std::string error;  // non-empty: the build failed (the first error)
    bool ok() const { return error.empty(); }

    uint64_t seed = 0;
    std::string dataDirectory;
    std::shared_ptr<const GeneratorList> generators;
    std::vector<SpaceTemplate> templates;
    std::vector<std::shared_ptr<const SpaceGeneration>> spaces;  // by SpaceId
    // By generator name
    std::map<std::string, uint16_t> pinned, current;
    uint64_t combinedHash = 0;
    // Some plan has sites (chunkStampSystem is registered, also when nonLegacy); some template's generator is not
    // legacyTerrain
    bool anySites = false, nonLegacy = false;
};

// A copyable value for pool jobs (D8): the space's generation data and the cache; never the World
class SAIGA_ENGINE_API ContentSource
{
   public:
    ContentSource() = default;
    ContentSource(std::shared_ptr<const SpaceGeneration> space, std::shared_ptr<ContentCache> cache);
    // Submits a pool job (in-flight registration when the job starts; a running generation of the same key is
    // attached to) whose completion runs at the pool's drain() on the main thread. Returns the job's ID
    uint64_t requestPlacement(StreamingPool& pool, ChunkId chunk, uint16_t placementVersion, int priority,
                              std::function<void(std::shared_ptr<const ChunkPlacement>)> onMainThread) const;
    // The pool form inside a job of another kind (S23b: the grass masks): a hit, a running generation (waited for), or
    // generated here
    std::shared_ptr<const ChunkPlacement> placementInJob(ChunkId chunk, uint16_t placementVersion) const;
    // The base in a pool job (S24, G17: the residency's job): a hit, a running generation (waited for), or generated
    // here with the key in flight
    std::shared_ptr<const ChunkBase> baseInJob(ChunkId chunk) const;
    bool valid() const { return sp != nullptr; }
    const SpaceGeneration& space() const { return *sp; }

   private:
    std::shared_ptr<const SpaceGeneration> sp;
    std::shared_ptr<ContentCache> cache;
};

class SAIGA_ENGINE_API GenerationPipeline
{
   public:
    // Phase A (pure): the plans of the setup with the given pins; plans.error names the first problem
    static GenerationPlans build(const WorldSetup& setup, const GenerationPins& pins, ContentCache& cache);
    // Phase B: the prefab entities against the World's classes, components and meshes; empty if valid
    static std::string validateEntities(const World& world, const GenerationPlans& plans);

    GenerationPipeline(GenerationPlans plans, std::shared_ptr<ContentCache> cache);

    const GenerationPlans& plans() const { return state; }
    // A host load with other pins (same space table): the plans are replaced (the stamp list caches dropped)
    void replacePlans(GenerationPlans plans);
    const GeneratorList& generators() const { return *state.generators; }
    ContentCache& cache() const { return *contentCache; }
    std::shared_ptr<ContentCache> sharedCache() const { return contentCache; }

    size_t spaceCount() const { return state.spaces.size(); }
    const SpaceGeneration& space(SpaceId space) const;
    std::shared_ptr<const SpaceGeneration> spacePtr(SpaceId space) const;
    // E6: the space's generator has terrain
    bool spaceHasTerrain(SpaceId space) const { return this->space(space).hasTerrain; }
    // Some template's generator is not legacyTerrain (GENR, GenerationState and the hash part; records come later)
    bool nonLegacy() const { return state.nonLegacy; }

    // The composed height (height units) at integer metres; 0 for a space without terrain. Const, lock-free,
    // allocation-free (parallelEach, pool jobs)
    int32_t height(SpaceId space, int64_t x, int64_t z) const;
    // S20's height source of the space (the composed height; empty without terrain)
    TerrainHeightSource heightSource(SpaceId space) const;

    std::shared_ptr<const ChunkBase> chunkBase(ChunkId chunk) const;
    std::shared_ptr<const ChunkPlacement> chunkPlacement(ChunkId chunk, uint16_t placementVersion) const;
    ContentSource contentSource(SpaceId space) const;
    // The stamp / static lists of a chunk from the plan only (cached per chunk)
    const std::vector<StampRecord>& stampList(ChunkId chunk) const;
    const std::vector<StaticPart>& staticList(ChunkId chunk) const;

   private:
    GenerationPlans state;
    std::shared_ptr<ContentCache> contentCache;
    mutable std::mutex listMutex;
    mutable std::map<ChunkId, std::unique_ptr<std::vector<StampRecord>>> stampLists;
    mutable std::map<ChunkId, std::unique_ptr<std::vector<StaticPart>>> staticLists;
};

// Pure helpers (the pipeline, ContentSource jobs, checkContentHashes)
SAIGA_ENGINE_API std::vector<StampRecord> computeStampList(const SpaceGeneration& space, ChunkId chunk);
SAIGA_ENGINE_API std::vector<StaticPart> computeStaticList(const SpaceGeneration& space, ChunkId chunk);
// inJob: the pool form of the cache (waits for a running generation of the same key; S24)
SAIGA_ENGINE_API std::shared_ptr<const ChunkBase> generateBase(ContentCache& cache, const SpaceGeneration& space,
                                                               ChunkId chunk, bool inJob = false);
// inJob: the pool form of the cache (waits for a running generation of the same key)
SAIGA_ENGINE_API std::shared_ptr<const ChunkPlacement> generatePlacement(ContentCache& cache,
                                                                         const SpaceGeneration& space, ChunkId chunk,
                                                                         uint16_t placementVersion, bool inJob = false);

}  // namespace Engine
}  // namespace Saiga
