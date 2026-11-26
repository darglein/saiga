/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// World::save, dumpJson and diff (docs/engine_design.md §15.1, S11). The format is described in SaveFormat.h.

#include "saiga/engine/content/ChunkRecords.h"
#include "saiga/engine/content/GenerationPipeline.h"
#include "saiga/engine/core/EngineVersion.h"
#include "saiga/engine/gameplay/Behaviour.h"
#include "saiga/engine/save/ChunkStore.h"
#include "saiga/engine/save/EntityGroup.h"
#include "saiga/engine/save/SaveDiff.h"
#include "saiga/engine/save/SaveDirectory.h"
#include "saiga/engine/save/SaveDirectoryFormat.h"
#include "saiga/engine/save/SaveFormat.h"
#include "saiga/engine/save/SaveReader.h"
#include "saiga/engine/save/SchemaTable.h"
#include "saiga/engine/terrain/TerrainDeltaCodec.h"
#include "saiga/engine/terrain/TerrainEditStore.h"
#include "saiga/engine/world/World.h"

#include <map>
#include <stdexcept>

namespace Saiga
{
namespace Engine
{
std::vector<uint8_t> World::save() const
{
    return saveBlob(false);
}

std::vector<uint8_t> World::saveWorldState() const
{
    return saveBlob(true);
}

std::vector<uint8_t> World::saveBlob(bool worldStateForm) const
{
    SAIGA_ASSERT(isHost(), "World::save on a client (its state comes from the host)");
    SAIGA_ASSERT(!inStep() && commands.empty(), "World::save inside a step");
    SAIGA_ASSERT(hookDepth == 0 && loadHookPhase == 0, "World::save inside a game hook");
    const SchemaTable schema = SchemaTable::fromWorld(*this);
    auto schemaIndex         = [&](SchemaKind kind, const char* name)
    {
        for (size_t i = 0; i < schema.types.size(); ++i)
            if (schema.types[i].kind == kind && schema.types[i].name == name) return uint32_t(i);
        SAIGA_EXIT_ERROR(std::string("save: ") + name + " is not in the schema");
        return uint32_t(0);
    };

    std::vector<uint8_t> out;
    out.insert(out.end(), std::begin(kSaveMagic), std::end(kSaveMagic));
    SectionWriter s(out);
    BinaryWriter& w = s.writer();
    w.write(kSaveFormatVersion);

    s.begin(kTagHeader);
    w.write(std::string(engineVersion()));
    w.write(config.stepsPerSecond);
    w.write(steps);
    w.write(time());
    s.end();

    s.begin(kTagSchema);
    schema.write(w);
    s.end();

    // Optional sections in table order, only with content (D7)
    if (spaceCount() > 1)
    {
        s.beginOptional(kTagSpaces, kSpaceTableVersion);
        w.write(uint32_t(spaceCount()));
        for (size_t i = 0; i < spaceCount(); ++i)
        {
            const SpaceTemplate& t = spaceTemplate(SpaceId(i));
            w.write(uint32_t(t.name.size()));
            for (char c : t.name) w.write(uint8_t(c));
            w.write(uint8_t(t.kind));
        }
        s.end();
    }

    // GENR (S23a): only when the world writes generation state (a non-legacy generator or chunk records)
    if (writesGenerationState())
    {
        const GenerationPlans& plans = pipeline->plans();
        s.beginOptional(kTagGeneration, kGenerationSectionVersion);
        w.write(plans.seed);
        w.write(uint32_t(plans.templates.size()));
        for (const SpaceTemplate& t : plans.templates)
            w.write(std::string(plans.generators->find(t.generator)->generator->name()));
        const std::vector<const GeneratorList::Entry*> gens = plans.generators->byName();
        w.write(uint32_t(gens.size()));
        for (const GeneratorList::Entry* g : gens)
        {
            const std::string name = g->generator->name();
            w.write(name);
            w.write(plans.pinned.at(name));
            w.write(plans.current.at(name));
            w.write(uint16_t(g->slots.size()));
            for (const GeneratorSlot& slot : g->slots)
            {
                w.write(slot.name);
                w.write(slot.flags);
            }
        }
        w.write(uint32_t(plans.spaces.size()));
        w.write(plans.combinedHash);
        s.end();
    }

    // TERR (S22): only with a chunk with terrain state (S25: not in the world-state form, the records hold it)
    if (!worldStateForm && terrainEditStore->chunkCount() > 0)
    {
        const TerrainEditStore& st = *terrainEditStore;
        s.beginOptional(kTagTerrain, kTerrainSectionVersion);
        uint32_t spacesWithState = 0;
        for (size_t i = 0; i < st.spaceCount(); ++i) spacesWithState += st.chunks(SpaceId(i)).empty() ? 0 : 1;
        w.write(spacesWithState);
        std::vector<int32_t> dense;
        std::vector<uint8_t> encoded;
        for (size_t i = 0; i < st.spaceCount(); ++i)
        {
            const SpaceId space = SpaceId(i);
            if (st.chunks(space).empty()) continue;
            w.write(space);
            w.write(uint8_t(st.spacing(space)));
            w.write(uint16_t(st.chunkSamples(space) + 1));
            w.write(uint32_t(st.chunks(space).size()));
            for (const auto& [c, e] : st.chunks(space))
            {
                w.write(c.x);
                w.write(c.z);
                w.write(e.sequence);
                const bool deltas = e.nonZeroSamples > 0;
                w.write(uint8_t(deltas ? 1 : 0));
                if (!deltas) continue;
                st.denseDelta(c, dense);
                encoded.clear();
                TerrainDeltaCodec::encode(dense.data(), dense.size(), true, encoded);
                w.write(uint64_t(encoded.size()));
                out.insert(out.end(), encoded.begin(), encoded.end());
            }
        }
        s.end();
    }
    // CREC (S23a): only with chunk records
    if (!worldStateForm && !chunkRecords->records().empty())
    {
        const ChunkRecords& rec = *chunkRecords;
        s.beginOptional(kTagChunkRecords, kChunkRecordSectionVersion);
        std::map<SpaceId, std::vector<std::pair<ChunkId, ChunkRecord>>> bySpace;
        for (const auto& [c, r] : rec.records()) bySpace[c.space].push_back({c, r});
        w.write(uint32_t(bySpace.size()));
        for (const auto& [space, list] : bySpace)
        {
            w.write(space);
            w.write(uint32_t(list.size()));
            for (const auto& [c, r] : list)
            {
                w.write(c.x);
                w.write(c.z);
                w.write(r.placementVersion);
                w.write(uint8_t(rec.isStamped(c) ? 1 : 0));
                // Format 2 (S23b): the tombstones
                w.write(uint32_t(r.tombstones.size()));
                for (const InstanceKey& k : r.tombstones)
                {
                    w.write(uint8_t(k.generatorSlot));
                    w.write(k.counter);
                }
            }
        }
        s.end();
    }
    // EGRP (S25): only with stored groups or unloaded spaces; disk-only groups are read through the file system
    // (blobs stay self-contained)
    if (!worldStateForm && store->hasContent())
    {
        s.beginOptional(kTagStoredGroups, kStoredGroupsSectionVersion);
        std::vector<uint8_t> schmBytes;
        {
            BinaryWriter sw(schmBytes);
            schema.write(sw);
        }
        // The store's tables in the blob's numbering: SCHM is 0, the others 1.. in first-use order
        const auto& tables = store->tables();
        std::vector<int64_t> number(tables.size(), -1);
        std::vector<uint32_t> extra;
        auto numberOf = [&](uint32_t t)
        {
            if (number[t] < 0)
            {
                std::vector<uint8_t> b;
                BinaryWriter tw(b);
                tables[t]->write(tw);
                if (b == schmBytes)
                    number[t] = 0;
                else
                {
                    extra.push_back(t);
                    number[t] = int64_t(extra.size());
                }
            }
            return uint32_t(number[t]);
        };
        auto bytesOf = [&](const ChunkId& c, const StoredGroup& g)
        {
            GroupBytes b = g.bytes;
            if (!b && saveDir) saveDir->readGroupForSave(c, g, b);
            SAIGA_ASSERT(b, "save: a stored group of " + std::to_string(c.space) + ", " + std::to_string(c.x) + ", " +
                                std::to_string(c.z) + " cannot be read");
            return b;
        };
        for (const auto& [c, e] : store->chunks())
            for (const StoredGroup& g : e.groups) numberOf(g.schema);
        w.write(uint32_t(extra.size()));
        for (uint32_t t : extra) tables[t]->write(w);
        auto writeGroupBytes = [&](const ChunkId& c, const StoredGroup& g)
        {
            const GroupBytes b = bytesOf(c, g);
            w.write(numberOf(g.schema));
            w.write(uint64_t(b->size()));
            out.insert(out.end(), b->begin(), b->end());
        };
        w.write(uint32_t(store->chunksWithGroups()));
        for (const auto& [c, e] : store->chunks())
        {
            if (isSpaceRecordKey(c) || e.groups.empty()) continue;
            w.write(c.space);
            w.write(c.x);
            w.write(c.z);
            w.write(uint32_t(e.groups.size()));
            for (const StoredGroup& g : e.groups) writeGroupBytes(c, g);
        }
        w.write(uint32_t(store->unloadedSpaces().size()));
        for (SpaceId sp : store->unloadedSpaces())
        {
            const ChunkEntry* e = store->entry(spaceRecordKey(sp));
            const bool group    = e && !e->groups.empty();
            w.write(sp);
            w.write(uint8_t(group ? 1 : 0));
            if (group) writeGroupBytes(spaceRecordKey(sp), e->groups[0]);
        }
        s.end();
    }

    s.begin(kTagEntities);
    const uint32_t slots = reg.slotCount();
    w.write(slots);
    for (uint32_t i = 1; i < slots; ++i)
    {
        const Registry::SlotRecord r = reg.slotRecord(i);
        SAIGA_ASSERT(r.state != SlotState::Reserved, "save with a reserved entity");
        w.write(r.generation);
        w.write(uint8_t(r.state));
        // S24 (G9): Simulated is derived (the load evaluates it): written set for every alive entity, so a saved byte
        // never depends on the centres or the ready set
        w.write(r.state == SlotState::Alive ? uint8_t(r.flags | Registry::kSimulated) : r.flags);
    }
    w.write(reg.freeList());
    s.end();

    s.begin(kTagComponents);
    uint32_t types = 0;
    for (size_t id = 0; id < kMaxComponents; ++id) types += reg.savedComponent(ComponentId(id)) ? 1 : 0;
    w.write(types);
    for (size_t id = 0; id < kMaxComponents; ++id)
    {
        if (!reg.savedComponent(ComponentId(id))) continue;
        const char* name = reg.componentInfo(ComponentId(id)).name;
        w.write(schemaIndex(SchemaKind::Component, name));
        const size_t countAt = out.size();
        w.write(uint64_t(0));
        const size_t lengthAt = out.size();
        w.write(uint64_t(0));
        const size_t start = out.size();
        uint64_t count     = 0;
        if (id == Transform::id)
        {
            for (uint32_t i = 1; i < slots; ++i)
            {
                if (reg.slotRecord(i).state != SlotState::Alive) continue;
                w.write(i);
                w.write(reg.transformAt(i));
                count++;
            }
        }
        else
        {
            const PoolBase* p = reg.poolById(ComponentId(id));
            for (const Entity& e : p->denseEntities())
            {
                w.write(e.index);
                p->appendBytes(e.index, out);
                count++;
            }
        }
        patchU64(out, countAt, count);
        patchU64(out, lengthAt, uint64_t(out.size() - start));
    }
    s.end();

    s.begin(kTagEvents);
    uint32_t eventTypes = 0;
    for (const auto& q : events.all()) eventTypes += (q && q->pendingCount() > 0) ? 1 : 0;
    w.write(eventTypes);
    for (const auto& q : events.all())
    {
        if (!q || q->pendingCount() == 0) continue;
        w.write(schemaIndex(SchemaKind::Event, q->name));
        w.write(uint64_t(q->pendingCount()));
        const size_t lengthAt = out.size();
        w.write(uint64_t(0));
        const size_t start = out.size();
        q->appendPending(out);
        patchU64(out, lengthAt, uint64_t(out.size() - start));
    }
    s.end();

    s.begin(kTagActions);
    w.write(uint64_t(actions.size()));
    for (const QueuedAction& a : actions)
    {
        w.write(a.peer);
        w.write(a.sequence);
        w.write(a.action.player);
        w.write(a.action.action);
        w.write(a.action.aim);
    }
    w.write(actionSequence);
    s.end();

    s.begin(kTagState);
    const Rng::State rs = rng.state();
    w.write(rs.state);
    w.write(rs.increment);
    w.write(persistentCounter);
    s.end();

    SAIGA_ASSERT(w.ok(), "save: " + w.error());
    return out;
}

std::string World::dumpJson(Entity e) const
{
    SAIGA_ASSERT(!inStep(), "World::dumpJson inside a step");
    auto entityJson = [&](Entity x)
    {
        json11::Json::object components;
        components["Transform"] = toJson(reg.read<Transform>(x));
        const ComponentMask& m  = reg.entityMask(x);
        for (size_t id = 1; id < kMaxComponents; ++id)
            if (m.test(ComponentId(id)) && reg.savedComponent(ComponentId(id)))
                components[reg.componentInfo(ComponentId(id)).name] = reg.poolById(ComponentId(id))->toJson(x.index);
        json11::Json::object o{
            {"entity", json11::Json::object{{"index", double(x.index)}, {"generation", double(x.generation)}}},
            {"flags", double(reg.flags(x))},
            {"components", components}};
        if (const Behaviour* b = reg.registered(Behaviour::id) ? reg.tryRead<Behaviour>(x) : nullptr)
            if (const ClassList::ClassInfo* c = classes.find(b->classId)) o["class"] = c->name;
        return json11::Json(o);
    };
    json11::Json::array list;
    if (e.valid())
    {
        if (reg.alive(e)) list.push_back(entityJson(e));
    }
    else
    {
        for (uint32_t i = 1; i < reg.slotCount(); ++i)
            if (Entity x = reg.aliveAt(i); x.valid()) list.push_back(entityJson(x));
    }
    json11::Json::object top{{"step", double(steps)}, {"time", time()}, {"entities", list}};
    // S25: the stored groups per chunk (count, root class, entities; not their components) and the unloaded spaces
    if (!e.valid() && store->hasContent())
    {
        json11::Json::array chunks;
        for (const auto& [c, en] : store->chunks())
        {
            if (isSpaceRecordKey(c) || en.groups.empty()) continue;
            json11::Json::array groups;
            for (const StoredGroup& g : en.groups)
            {
                GroupBytes b = g.bytes;
                if (!b && saveDir) saveDir->readGroupForSave(c, g, b);
                std::string cls;
                uint32_t n = 0;
                if (b && g.schema < store->tables().size()) describeGroup(*b, *store->tables()[g.schema], cls, n);
                groups.push_back(json11::Json::object{{"root_class", cls}, {"entities", double(n)}});
            }
            chunks.push_back(json11::Json::object{
                {"chunk", json11::Json::array{double(c.space), double(c.x), double(c.z)}}, {"groups", groups}});
        }
        top["stored_groups"] = chunks;
        json11::Json::array spaces;
        for (SpaceId s : store->unloadedSpaces()) spaces.push_back(double(s));
        top["unloaded_spaces"] = spaces;
    }
    return json11::Json(top).dump();
}

std::string World::diff(const World& a, const World& b, int maxLines)
{
    DiffFilter filter;
    filter.limit     = maxLines > 0 ? size_t(maxLines) : 0;
    const SaveDiff d = diffSaves(a.save(), b.save(), filter);
    if (!d.error.empty()) return "diff failed: " + d.error;
    std::string out;
    for (const SaveDifference& x : d.differences) out += x.describe() + "\n";
    if (d.total > d.differences.size()) out += "... and " + std::to_string(d.total - d.differences.size()) + " more\n";
    return out;
}

bool World::checkLoadHook(const char* operation, Entity e, ComponentId id) const
{
    // configure (phase 1) makes no structural changes; onLoad (phase 2) only adds or removes Saved::No components of
    // its own entity, so a load followed by save() gives the saved bytes
    const bool allowed = loadHookPhase == 2 && e == loadHookEntity && reg.registered(id) && !reg.savedComponent(id);
    // Refused in every build, catchable: the loader reports the hook's exception as a hook error (an assert would
    // stop the program before the load could report it)
    if (!allowed)
        throw std::logic_error(std::string(operation) +
                               " during a load: configure makes no structural changes, onLoad adds or removes only "
                               "Saved::No components of its own entity");
    return allowed;
}

}  // namespace Engine
}  // namespace Saiga
