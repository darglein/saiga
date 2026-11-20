/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// World::save, dumpJson and diff (docs/engine_design.md §15.1, S11). The format is described in SaveFormat.h.

#include "saiga/engine/core/EngineVersion.h"
#include "saiga/engine/gameplay/Behaviour.h"
#include "saiga/engine/save/SaveDiff.h"
#include "saiga/engine/save/SaveFormat.h"
#include "saiga/engine/save/SchemaTable.h"
#include "saiga/engine/world/World.h"

namespace Saiga
{
namespace Engine
{
std::vector<uint8_t> World::save() const
{
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

    s.begin(kTagEntities);
    const uint32_t slots = reg.slotCount();
    w.write(slots);
    for (uint32_t i = 1; i < slots; ++i)
    {
        const Registry::SlotRecord r = reg.slotRecord(i);
        SAIGA_ASSERT(r.state != SlotState::Reserved, "save with a reserved entity");
        w.write(r.generation);
        w.write(uint8_t(r.state));
        w.write(r.flags);
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
    return json11::Json(json11::Json::object{{"step", double(steps)}, {"time", time()}, {"entities", list}}).dump();
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
    SAIGA_ASSERT(allowed, std::string(operation) +
                              " during a load: configure makes no structural changes, onLoad adds or removes only "
                              "Saved::No components of its own entity");
    return allowed;
}

}  // namespace Engine
}  // namespace Saiga
