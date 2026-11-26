/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/content/GenerationPipeline.h"

#include "saiga/engine/api/ComponentJson.h"
#include "saiga/engine/components/SpaceLink.h"
#include "saiga/engine/content/ContentHash.h"
#include "saiga/engine/content/gen/ChunkGeneration.h"
#include "saiga/engine/core/Hash.h"
#include "saiga/engine/core/StreamingPool.h"
#include "saiga/engine/net/NetOwner.h"
#include "saiga/engine/world/World.h"

#include <algorithm>
#include <set>

namespace Saiga
{
namespace Engine
{
namespace
{
GenerationTemplate viewOf(const SpaceTemplate& t, uint16_t index)
{
    GenerationTemplate v;
    v.name          = t.name;
    v.kind          = t.kind;
    v.templateIndex = index;
    v.baseSpacing   = t.baseSpacing;
    v.sizeChunks    = t.sizeChunks;
    v.seaLevel      = t.kind == SpaceKind::Enclosed ? std::nullopt : t.seaLevel;
    return v;
}

bool sameView(const GenerationTemplate& a, const GenerationTemplate& b)
{
    return a.name == b.name && a.kind == b.kind && a.templateIndex == b.templateIndex &&
           a.baseSpacing == b.baseSpacing && a.sizeChunks == b.sizeChunks && a.seaLevel == b.seaLevel;
}

uint64_t slotHash(const std::vector<GeneratorSlot>& slots)
{
    Hasher h(fnv1a64("saiga.content.slots.v1"));
    h.addU32(uint32_t(slots.size()));
    for (const GeneratorSlot& s : slots) h.addString(s.name).addU32(s.flags);
    return h.value();
}

// The closed integer bounds of a site's rotated footprint in space metres; false for an origin far outside every space
// (|x|, |z| > 2^40: the translation could overflow)
bool siteBounds(const Site& s, int64_t out[4])
{
    constexpr int64_t kOriginLimit = int64_t(1) << 40;
    if (s.x > kOriginLimit || s.x < -kOriginLimit || s.z > kOriginLimit || s.z < -kOriginLimit) return false;
    rotateBounds(s.footprint, s.quarterTurns, out);
    out[0] += s.x;
    out[2] += s.x;
    out[1] += s.z;
    out[3] += s.z;
    return true;
}

// sitesByChunk: every chunk the closed bounds of a site's rotated footprint intersect, clipped to the space
void fillSitesByChunk(WorldPlan& plan, SpaceId space)
{
    const int64_t size = plan.space.sizeChunks;
    for (uint32_t i = 0; i < uint32_t(plan.sites.size()); ++i)
    {
        const Site& s = plan.sites[i];
        if (s.quarterTurns > 3 || s.footprint[0] >= s.footprint[2] || s.footprint[1] >= s.footprint[3]) continue;
        int64_t b[4];
        if (!siteBounds(s, b)) continue;  // refused by checkPlan
        const int64_t cx0 = std::max<int64_t>(0, floorDiv(b[0], 256)), cz0 = std::max<int64_t>(0, floorDiv(b[1], 256));
        const int64_t cx1 = std::min<int64_t>(size - 1, floorDiv(b[2], 256));
        const int64_t cz1 = std::min<int64_t>(size - 1, floorDiv(b[3], 256));
        for (int64_t cz = cz0; cz <= cz1; ++cz)
            for (int64_t cx = cx0; cx <= cx1; ++cx)
            {
                ChunkId c;
                c.space = space;
                c.x     = int32_t(cx);
                c.z     = int32_t(cz);
                plan.sitesByChunk[c].push_back(i);
            }
    }
}

std::shared_ptr<const SiteContent> siteContentOf(const Prefab& p)
{
    auto c = std::make_shared<SiteContent>();
    if (p.patch)
    {
        auto d     = std::make_shared<PatchData>();
        d->originX = p.patch->originX;
        d->originZ = p.patch->originZ;
        d->sizeX   = p.patch->sizeX;
        d->sizeZ   = p.patch->sizeZ;
        d->heights = p.patch->heights;
        d->weights = p.patch->weights;
        c->patch   = std::move(d);
    }
    for (const PrefabRect& r : p.exclusion) c->exclusion.insert(c->exclusion.end(), {r.x0, r.z0, r.x1, r.z1});
    return c;
}

ChunkInputs inputsOf(const SpaceGeneration& s)
{
    ChunkInputs in;
    in.generator     = s.generator.get();
    in.pinnedVersion = s.pinned;
    in.plan          = s.plan.get();
    in.space         = s.id;
    in.sites         = &s.siteContent;
    in.openWorld     = s.openWorld;
    return in;
}

struct Builder
{
    const WorldSetup& setup;
    const GenerationPins& pins;
    ContentCache& cache;
    GenerationPlans out;

    bool fail(const std::string& e)
    {
        if (out.error.empty()) out.error = e;
        return false;
    }

    std::string spaceName(SpaceId id, const SpaceGeneration& s) const
    {
        return "space " + std::to_string(id) + " ('" + s.settings.name + "', generator '" + s.generatorName + "')";
    }

    // The plan of an instance: the generator's in/out plan() with seed and view set, through the cache
    std::shared_ptr<SpaceGeneration> makeSpace(SpaceId id, const SpaceInstance& instance)
    {
        auto s                        = std::make_shared<SpaceGeneration>();
        s->id                         = id;
        s->instance                   = instance;
        s->settings                   = out.templates[instance.templateIndex];
        s->generatorId                = s->settings.generator;
        const GeneratorList::Entry* e = out.generators->find(s->generatorId);
        s->generator                  = e->generator;
        s->generatorName              = e->generator->name();
        s->pinned                     = out.pinned[s->generatorName];
        s->current                    = out.current[s->generatorName];
        s->openWorld                  = s->settings.kind == SpaceKind::OpenWorld;
        s->hasTerrain                 = s->generator->hasTerrain() && s->openWorld;

        const GenerationTemplate view = viewOf(s->settings, instance.templateIndex);
        const Generator& g            = *s->generator;
        const uint16_t pinned         = s->pinned;
        const uint64_t seed           = instance.seed;
        std::string planError;
        s->plan = cache.plan(ContentCache::PlanKey{s->generatorName, pinned, seed, templateViewHash(view), id},
                             [&]()
                             {
                                 auto p   = std::make_shared<WorldPlan>();
                                 p->seed  = seed;
                                 p->space = view;
                                 g.plan(pinned, *p);
                                 if (p->seed != seed || !sameView(p->space, view))
                                     SAIGA_EXIT_ERROR(std::string("generator '") + g.name() +
                                                      "': plan() changed plan.seed or plan.space");
                                 if (!p->sitesByChunk.empty())
                                     SAIGA_EXIT_ERROR(std::string("generator '") + g.name() +
                                                      "': plan() filled sitesByChunk (the pipeline fills it)");
                                 fillSitesByChunk(*p, id);
                                 return std::shared_ptr<const WorldPlan>(std::move(p));
                             });
        return s;
    }

    // The prefabs and the validation of a plan (Scope 4)
    bool checkPlan(SpaceGeneration& s)
    {
        const WorldPlan& plan  = *s.plan;
        const std::string what = spaceName(s.id, s);
        const Generator& g     = *s.generator;
        if (s.id != 0 && !plan.spaces.empty())
            return fail(what + ": only the overworld's plan lists space instances (no nested instances)");
        if (g.legacyTerrain() && !plan.sites.empty())
            return fail(what + ": a legacyTerrain generator's plan has no sites");
        const int64_t size = int64_t(s.settings.sizeChunks) * 256;
        std::set<uint32_t> keys;
        std::vector<std::array<int64_t, 4>> bounds;
        std::map<const Prefab*, std::shared_ptr<const SiteContent>> contentOf;
        for (const Site& site : plan.sites)
        {
            const std::string sw = what + ", site " + std::to_string(site.key) + " ('" + site.prefab + "')";
            if (!keys.insert(site.key).second) return fail(sw + ": the site key is used twice");
            if (site.quarterTurns > 3) return fail(sw + ": quarterTurns must be 0..3");
            if (site.footprint[0] >= site.footprint[2] || site.footprint[1] >= site.footprint[3])
                return fail(sw + ": an empty footprint");
            if (site.baseHeight > kTerrainHeightLimit || site.baseHeight < -kTerrainHeightLimit)
                return fail(sw + ": baseHeight beyond kTerrainHeightLimit");
            std::array<int64_t, 4> b;
            if (!siteBounds(site, b.data()))
                return fail(sw + ": the origin (" + std::to_string(site.x) + ", " + std::to_string(site.z) +
                            ") leaves the space");
            if (b[0] < 0 || b[1] < 0 || b[2] > size || b[3] > size)
                return fail(sw + ": the rotated footprint [" + std::to_string(b[0]) + ", " + std::to_string(b[1]) +
                            ", " + std::to_string(b[2]) + ", " + std::to_string(b[3]) + "] leaves the space [0, " +
                            std::to_string(size) + "]^2");
            for (size_t k = 0; k < bounds.size(); ++k)
                if (b[0] <= bounds[k][2] && bounds[k][0] <= b[2] && b[1] <= bounds[k][3] && bounds[k][1] <= b[3])
                    return fail(sw + ": the footprint overlaps (or touches) site " + std::to_string(plan.sites[k].key) +
                                "'s");
            bounds.push_back(b);

            std::string error;
            std::shared_ptr<const Prefab> prefab =
                cache.prefabs().get(out.dataDirectory, site.prefab, site.prefabVersion, error);
            if (!prefab) return fail(sw + ": " + error);
            const PrefabRect& f = prefab->footprint;
            if (f.x0 < site.footprint[0] || f.z0 < site.footprint[1] || f.x1 > site.footprint[2] ||
                f.z1 > site.footprint[3])
                return fail(sw + ": the prefab's footprint does not fit the site's footprint");
            if (prefab->patch && !s.hasTerrain)
                return fail(sw + ": a terrain_patch in a space without terrain (" + prefab->file + ")");
            if (prefab->patch)
                for (int32_t h : prefab->patch->heights)
                {
                    const int64_t v = int64_t(site.baseHeight) + h;
                    if (v > kTerrainHeightLimit || v < -kTerrainHeightLimit)
                        return fail(sw + ": a patched height lies beyond kTerrainHeightLimit");
                }
            if (!prefab->spaces.empty() && s.id != 0)
                return fail(sw + ": a site of an instance space creates spaces (" + prefab->file +
                            "; no nested instances)");
            for (const std::string& t : prefab->spaces)
                if (std::none_of(out.templates.begin(), out.templates.end(),
                                 [&](const SpaceTemplate& st) { return st.name == t; }))
                    return fail(sw + ": the space template '" + t + "' does not exist (" + prefab->file + ")");
            for (const PrefabEntity& e : prefab->entities)
                if (e.spaceLink && e.spaceLink->toParent && s.instance.origin != SpaceInstance::FromSite)
                    return fail(sw + ": a space_link to \"parent\" in a space that no site created (" + prefab->file +
                                ")");
            s.prefabs.push_back(prefab);
            // One immutable SiteContent per prefab (a prefab placed many times shares its patch arrays)
            std::shared_ptr<const SiteContent>& content = contentOf[prefab.get()];
            if (!content) content = siteContentOf(*prefab);
            s.siteContent.push_back(content);
        }
        std::vector<uint64_t> prefabHashes;
        for (const auto& p : s.prefabs) prefabHashes.push_back(p->contentHash);
        s.planHash = planHash(plan, s.pinned, prefabHashes);
        return true;
    }

    void run()
    {
        out.seed          = setup.seed;
        out.dataDirectory = setup.dataDirectory;
        validateSpaceTemplates(setup.config.spaceTemplates);
        out.templates = setup.config.spaceTemplates;
        if (out.templates.empty()) out.templates.push_back(SpaceTemplate());
        out.generators = std::make_shared<const GeneratorList>(makeGeneratorList(setup.generators, out.templates));

        for (const auto& [id, e] : out.generators->entries())
        {
            const Generator& g     = *e.generator;
            const std::string name = g.name();
            cache.registerIdentity({name, g.version(), slotHash(e.slots)});
            auto pin              = pins.pinned.find(name);
            auto cur              = pins.current.find(name);
            const uint16_t pinned = pin == pins.pinned.end() ? g.version() : pin->second;
            const uint16_t curr   = cur == pins.current.end() ? g.version() : cur->second;
            if (pinned == 0 || pinned > g.version() || curr > g.version() || curr < pinned)
            {
                fail("generator '" + name + "': pinned v" + std::to_string(pinned) + " / current v" +
                     std::to_string(curr) + " (this build has v1..v" + std::to_string(g.version()) +
                     "; the current version is at least the pinned one)");
                return;
            }
            out.pinned[name]  = pinned;
            out.current[name] = curr;
        }
        for (const SpaceTemplate& t : out.templates)
        {
            const Generator& g = *out.generators->find(t.generator)->generator;
            if (t.kind == SpaceKind::Enclosed && g.hasTerrain())
                SAIGA_EXIT_ERROR("space template '" + t.name + "': an enclosed space has no terrain (its generator '" +
                                 g.name() +
                                 "' has terrain; use SpaceTemplate::enclosed(name, sizeChunks) with a "
                                 "generator without terrain)");
            out.nonLegacy = out.nonLegacy || !g.legacyTerrain();
        }

        // 1. The overworld
        SpaceInstance overworld;
        overworld.templateIndex = 0;
        overworld.seed          = setup.seed;
        overworld.origin        = SpaceInstance::Overworld;
        std::vector<std::shared_ptr<SpaceGeneration>> spaces;
        spaces.push_back(makeSpace(0, overworld));
        if (!checkPlan(*spaces[0])) return;

        std::vector<SpaceInstance> list{overworld};
        std::vector<bool> referenced(out.templates.size(), false);
        referenced[0] = true;
        // 2. The generator's listed instances
        for (const SpaceInstance& listed : spaces[0]->plan->spaces)
        {
            if (listed.templateIndex >= out.templates.size())
            {
                fail(spaceName(0, *spaces[0]) + ": a listed instance names the template index " +
                     std::to_string(listed.templateIndex) + " (" + std::to_string(out.templates.size()) +
                     " templates)");
                return;
            }
            SpaceInstance i = listed;
            i.parent        = 0;
            i.siteKey       = 0;
            i.link          = 0;
            i.origin        = SpaceInstance::Listed;
            list.push_back(i);
            referenced[i.templateIndex] = true;
        }
        // 3. The sites' instances (site order, then the prefab's spaces order)
        const WorldPlan& plan0 = *spaces[0]->plan;
        for (size_t si = 0; si < plan0.sites.size(); ++si)
        {
            const Site& site     = plan0.sites[si];
            const Prefab& prefab = *spaces[0]->prefabs[si];
            const uint64_t site0 = hashCombine(hashCombine(setup.seed, fnv1a64("site")), site.key);
            for (size_t link = 0; link < prefab.spaces.size(); ++link)
            {
                SpaceInstance i;
                for (size_t t = 0; t < out.templates.size(); ++t)
                    if (out.templates[t].name == prefab.spaces[link]) i.templateIndex = uint16_t(t);
                i.seed    = link == 0 ? site0 : hashCombine(site0, link);
                i.parent  = 0;
                i.siteKey = site.key;
                i.link    = uint8_t(link);
                i.origin  = SpaceInstance::FromSite;
                list.push_back(i);
                referenced[i.templateIndex] = true;
            }
        }
        // 4. One instance per unreferenced template, in template order
        for (size_t t = 1; t < out.templates.size(); ++t)
            if (!referenced[t])
            {
                SpaceInstance i;
                i.templateIndex = uint16_t(t);
                i.seed          = hashCombine(setup.seed, t);
                i.origin        = SpaceInstance::Unreferenced;
                list.push_back(i);
            }
        if (list.size() > 65535)
        {
            fail("the world plan lists " + std::to_string(list.size()) + " spaces (at most 65535)");
            return;
        }
        for (size_t i = 1; i < list.size(); ++i)
        {
            spaces.push_back(makeSpace(SpaceId(i), list[i]));
            if (!checkPlan(*spaces.back())) return;
        }

        std::vector<CombinedPlanEntry> combined;
        for (const auto& s : spaces)
        {
            out.anySites = out.anySites || !s->plan->sites.empty();
            combined.push_back({s->instance.templateIndex, s->generatorName, s->planHash});
            out.spaces.push_back(s);
        }
        out.combinedHash = combinedPlanHash(combined);
    }
};
}  // namespace

// ---- phase A / B

GenerationPlans GenerationPipeline::build(const WorldSetup& setup, const GenerationPins& pins, ContentCache& cache)
{
    Builder b{setup, pins, cache, {}};
    b.run();
    return std::move(b.out);
}

std::string GenerationPipeline::validateEntities(const World& world, const GenerationPlans& plans)
{
    const Registry& reg = world.registry();
    for (const auto& sp : plans.spaces)
        for (size_t si = 0; si < sp->prefabs.size(); ++si)
        {
            const Prefab& prefab = *sp->prefabs[si];
            for (size_t ei = 0; ei < prefab.entities.size(); ++ei)
            {
                const PrefabEntity& e  = prefab.entities[ei];
                const std::string what = "prefab '" + prefab.name + "' v" + std::to_string(prefab.version) + " (" +
                                         prefab.file + "), entity " + std::to_string(ei) + " ('" + e.className + "')";
                if (!world.classList().find(e.className)) return what + ": unknown class \"" + e.className + "\"";
                for (const auto& kv : e.components.object_items())
                {
                    const int found = reg.findComponent(kv.first);
                    if (found < 0) return what + ": unknown component \"" + kv.first + "\"";
                    const ComponentId id = ComponentId(found);
                    if (id == Parent::id || id == NetOwner::id || id == SpaceLink::id)
                        return what + ": " + kv.first +
                               " cannot be set by a prefab (Parent: no entity references; NetOwner: the host; "
                               "SpaceLink: space_link)";
                    std::string error;
                    if (!checkComponentValues(world, Entity{}, id, kv.second, error))
                        return what + ": " + kv.first + ": " + error;
                }
            }
        }
    return {};
}

// ---- the pipeline

GenerationPipeline::GenerationPipeline(GenerationPlans plans, std::shared_ptr<ContentCache> cache)
    : state(std::move(plans)), contentCache(std::move(cache))
{
    SAIGA_ASSERT(state.ok() && contentCache);
}

void GenerationPipeline::replacePlans(GenerationPlans plans)
{
    SAIGA_ASSERT(plans.ok());
    std::lock_guard<std::mutex> lock(listMutex);
    state = std::move(plans);
    stampLists.clear();
    staticLists.clear();
}

const SpaceGeneration& GenerationPipeline::space(SpaceId space) const
{
    SAIGA_ASSERT(space < state.spaces.size(), "GenerationPipeline: unknown space " + std::to_string(space));
    return *state.spaces[space];
}

std::shared_ptr<const SpaceGeneration> GenerationPipeline::spacePtr(SpaceId space) const
{
    SAIGA_ASSERT(space < state.spaces.size(), "GenerationPipeline: unknown space " + std::to_string(space));
    return state.spaces[space];
}

int32_t GenerationPipeline::height(SpaceId id, int64_t x, int64_t z) const
{
    const SpaceGeneration& s = space(id);
    if (!s.hasTerrain) return 0;
    return composedHeight(*s.generator, s.pinned, *s.plan, s.id, s.siteContent, x, z);
}

TerrainHeightSource GenerationPipeline::heightSource(SpaceId id) const
{
    std::shared_ptr<const SpaceGeneration> s = spacePtr(id);
    if (!s->hasTerrain) return TerrainHeightSource();
    TerrainHeightSource source;
    source.height = [s](int64_t x, int64_t z)
    { return composedHeight(*s->generator, s->pinned, *s->plan, s->id, s->siteContent, x, z); };
    // The pyramid once per plan in the cache (the worlds of a shared cache: a host and its clients, the scenario
    // variants, tests)
    std::shared_ptr<ContentCache> cache = contentCache;
    source.cachedGrid = [s, cache](int n, int64_t spacing, const std::function<std::vector<int32_t>()>& make)
    { return cache->grid(ContentCache::GridKey{s->generatorName, s->pinned, s->planHash, s->id, n, spacing}, make); };
    return source;
}

std::shared_ptr<const ChunkBase> generateBase(ContentCache& cache, const SpaceGeneration& space, ChunkId chunk,
                                              bool inJob)
{
    const ContentCache::ChunkKey key{space.generatorName, space.pinned, space.planHash, chunk};
    auto make = [&]() { return generateChunkBase(inputsOf(space), chunk); };
    return inJob ? cache.baseInJob(key, make) : cache.base(key, make);
}

std::shared_ptr<const ChunkPlacement> generatePlacement(ContentCache& cache, const SpaceGeneration& space,
                                                        ChunkId chunk, uint16_t placementVersion, bool inJob)
{
    const ContentCache::ChunkKey key{space.generatorName, placementVersion, space.planHash, chunk};
    auto make = [&]()
    {
        // In a job the base of a running generation of the same chunk is waited for (S24: the residency's job)
        std::shared_ptr<const ChunkBase> base = generateBase(cache, space, chunk, inJob);
        return generateChunkPlacement(inputsOf(space), *base, placementVersion,
                                      hashStamps(computeStampList(space, chunk)),
                                      hashStatics(computeStaticList(space, chunk)));
    };
    return inJob ? cache.placementInJob(key, make) : cache.placement(key, make);
}

std::shared_ptr<const ChunkBase> GenerationPipeline::chunkBase(ChunkId chunk) const
{
    return generateBase(*contentCache, space(chunk.space), chunk);
}

std::shared_ptr<const ChunkPlacement> GenerationPipeline::chunkPlacement(ChunkId chunk, uint16_t placementVersion) const
{
    return generatePlacement(*contentCache, space(chunk.space), chunk, placementVersion);
}

ContentSource GenerationPipeline::contentSource(SpaceId space) const
{
    return ContentSource(spacePtr(space), contentCache);
}

std::vector<StampRecord> computeStampList(const SpaceGeneration& space, ChunkId chunk)
{
    std::vector<StampRecord> out;
    const WorldPlan& plan = *space.plan;
    auto it               = plan.sitesByChunk.find(chunk);
    if (it == plan.sitesByChunk.end()) return out;
    for (uint32_t si : it->second)
    {
        const Site& site     = plan.sites[si];
        const Prefab& prefab = *space.prefabs[si];
        for (uint32_t ei = 0; ei < uint32_t(prefab.entities.size()); ++ei)
        {
            const PrefabEntity& e = prefab.entities[ei];
            int64_t rx = 0, rz = 0;
            rotateQuarter(e.x, e.z, site.quarterTurns, rx, rz);
            const int64_t x = site.x * kPosUnitsPerMetre + rx, z = site.z * kPosUnitsPerMetre + rz;
            if (floorShift(x, 16) != chunk.x || floorShift(z, 16) != chunk.z) continue;
            StampRecord r;
            r.siteIndex     = si;
            r.siteKey       = site.key;
            r.entity        = ei;
            r.prefab        = prefab.name;
            r.prefabVersion = prefab.version;
            r.className     = e.className;
            r.x             = x;
            r.z             = z;
            r.y             = int32_t(int64_t(site.baseHeight) + e.y);
            r.yaw           = uint16_t(uint32_t(e.yaw) + uint32_t(site.quarterTurns) * 16384u);
            out.push_back(std::move(r));
        }
    }
    // (site index, entity index): sitesByChunk lists the sites in order
    return out;
}

std::vector<StaticPart> computeStaticList(const SpaceGeneration& space, ChunkId chunk)
{
    std::vector<StaticPart> out;
    const WorldPlan& plan = *space.plan;
    auto it               = plan.sitesByChunk.find(chunk);
    if (it == plan.sitesByChunk.end()) return out;
    for (uint32_t si : it->second)
    {
        const Site& site     = plan.sites[si];
        const Prefab& prefab = *space.prefabs[si];
        for (uint32_t pi = 0; pi < uint32_t(prefab.statics.size()); ++pi)
        {
            const PrefabStatic& s = prefab.statics[pi];
            int64_t rx = 0, rz = 0;
            rotateQuarter(s.x, s.z, site.quarterTurns, rx, rz);
            const int64_t x = site.x * kPosUnitsPerMetre + rx, z = site.z * kPosUnitsPerMetre + rz;
            if (floorShift(x, 16) != chunk.x || floorShift(z, 16) != chunk.z) continue;
            StaticPart p;
            p.siteIndex = si;
            p.siteKey   = site.key;
            p.part      = pi;
            p.mesh      = s.mesh;
            p.x         = x;
            p.z         = z;
            p.y         = int32_t(int64_t(site.baseHeight) + s.y);
            p.yaw       = uint16_t(uint32_t(s.yaw) + uint32_t(site.quarterTurns) * 16384u);
            out.push_back(std::move(p));
        }
    }
    return out;
}

const std::vector<StampRecord>& GenerationPipeline::stampList(ChunkId chunk) const
{
    std::lock_guard<std::mutex> lock(listMutex);
    auto& entry = stampLists[chunk];
    if (!entry) entry = std::make_unique<std::vector<StampRecord>>(computeStampList(space(chunk.space), chunk));
    return *entry;
}

const std::vector<StaticPart>& GenerationPipeline::staticList(ChunkId chunk) const
{
    std::lock_guard<std::mutex> lock(listMutex);
    auto& entry = staticLists[chunk];
    if (!entry) entry = std::make_unique<std::vector<StaticPart>>(computeStaticList(space(chunk.space), chunk));
    return *entry;
}

// ---- ContentSource

ContentSource::ContentSource(std::shared_ptr<const SpaceGeneration> space, std::shared_ptr<ContentCache> cache)
    : sp(std::move(space)), cache(std::move(cache))
{
}

std::shared_ptr<const ChunkBase> ContentSource::baseInJob(ChunkId chunk) const
{
    return generateBase(*cache, *sp, chunk, true);
}

std::shared_ptr<const ChunkPlacement> ContentSource::placementInJob(ChunkId chunk, uint16_t placementVersion) const
{
    return generatePlacement(*cache, *sp, chunk, placementVersion, true);
}

uint64_t ContentSource::requestPlacement(StreamingPool& pool, ChunkId chunk, uint16_t placementVersion, int priority,
                                         std::function<void(std::shared_ptr<const ChunkPlacement>)> onMainThread) const
{
    auto result                                  = std::make_shared<std::shared_ptr<const ChunkPlacement>>();
    std::shared_ptr<const SpaceGeneration> space = sp;
    std::shared_ptr<ContentCache> c              = cache;
    return pool.submit([space, c, chunk, placementVersion, result](const std::atomic<bool>&)
                       { *result = generatePlacement(*c, *space, chunk, placementVersion, true); }, priority,
                       [result, onMainThread](bool cancelled)
                       {
                           if (!cancelled && onMainThread) onMainThread(*result);
                       });
}

}  // namespace Engine
}  // namespace Saiga
