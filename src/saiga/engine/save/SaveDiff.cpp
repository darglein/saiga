/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// diffSaves: field-level comparison of two save blobs by name (S11).

#include "saiga/engine/save/SaveDiff.h"

#include "saiga/engine/ecs/Registry.h"
#include "saiga/engine/save/GenericDecode.h"
#include "saiga/engine/save/SaveFormat.h"
#include "saiga/engine/save/SaveReader.h"

#include <algorithm>
#include <map>
#include <set>
#include <tuple>

#include <unordered_map>

namespace Saiga
{
namespace Engine
{
namespace
{
using Json = json11::Json;

Json entityJson(Entity e)
{
    return Json(Json::object{{"index", double(e.index)}, {"generation", double(e.generation)}});
}

std::string short_(const Json& j)
{
    std::string s = j.dump();
    if (s.size() > 200) s = s.substr(0, 197) + "...";
    return s;
}

// The decoded components of one save: name -> entity index -> (bytes, JSON)
struct Value
{
    size_t begin = 0, end = 0;
    mutable Json json;  // decoded on demand (only values that differ)
    mutable bool decoded = false;
    bool present         = false;
};
// The values of one component type by entity index
// The values of one component type by entity index (sparse: a block's size, not the slot count, bounds the memory)
struct Values
{
    std::unordered_map<uint32_t, Value> byIndex;
    const Value* find(uint32_t index) const
    {
        auto it = byIndex.find(index);
        return it == byIndex.end() ? nullptr : &it->second;
    }
};
struct Decoded
{
    std::map<std::string, Values> components;
    std::map<std::string, const SchemaType*> types;
    std::map<std::string, std::vector<Json>> events;
};

bool decodeAll(const std::vector<uint8_t>& data, const ParsedSave& p, Decoded& out, std::string& error)
{
    SaveDecoder decoder(p.schema, nullptr, nullptr);
    for (const ParsedSave::Block& b : p.components)
    {
        const SchemaType& t = p.schema.types[b.schemaIndex];
        out.types[t.name]   = &t;
        // Empty blocks take no memory (a save lists every registered type)
        if (b.count == 0) continue;
        auto& values = out.components[t.name].byIndex;
        values.reserve(size_t(b.count));
        BinaryReader r(data.data() + b.begin, b.end - b.begin);
        for (uint64_t i = 0; i < b.count; ++i)
        {
            uint32_t index = 0;
            r.read(index);
            Value v;
            v.begin = b.begin + r.position();
            if (!r.ok() || !decoder.skip(t, r, error))
            {
                if (error.empty()) error = t.name + ": " + r.error();
                return false;
            }
            v.end     = b.begin + r.position();
            v.present = true;
            if (index >= p.slots.size())
            {
                error = t.name + ": the entity index " + std::to_string(index) + " is not a slot";
                return false;
            }
            if (!values.emplace(index, std::move(v)).second)
            {
                error = t.name + ": two values for entity index " + std::to_string(index);
                return false;
            }
        }
        if (!r.atEnd())
        {
            error = t.name + ": bytes left over";
            return false;
        }
    }
    for (const ParsedSave::Block& b : p.events)
    {
        const SchemaType& t = p.schema.types[b.schemaIndex];
        auto& list          = out.events[t.name];
        BinaryReader r(data.data() + b.begin, b.end - b.begin);
        for (uint64_t i = 0; i < b.count; ++i)
        {
            Json j;
            if (!decoder.decode(t, r, nullptr, j, error)) return false;
            list.push_back(j);
        }
    }
    return true;
}

// The JSON form of a value (decoded once)
const Json& jsonOf(const Value& v, const std::vector<uint8_t>& data, const ParsedSave& p, const SchemaType& t)
{
    if (!v.decoded)
    {
        SaveDecoder decoder(p.schema, nullptr, nullptr);
        BinaryReader r(data.data() + v.begin, v.end - v.begin);
        std::string error;
        if (!decoder.decode(t, r, nullptr, v.json, error)) v.json = Json(Json::object{{"<undecodable>", error}});
        v.decoded = true;
    }
    return v.json;
}

// SPCE: one difference per space id whose row differs ("space 1: 'cellar' (enclosed) vs 'cave' (enclosed)")
template <class Add>
void diffSpaceTables(const SpaceTable& a, const SpaceTable& b, Add& add)
{
    auto row = [](const SpaceTable& t, size_t i)
    {
        if (i >= t.rows.size()) return Json();
        return Json("'" + t.rows[i].name + "' (" + (t.rows[i].kind == SpaceKind::Enclosed ? "enclosed" : "open world") +
                    ")");
    };
    for (size_t i = 0; i < std::max(a.rows.size(), b.rows.size()); ++i)
    {
        const Json x = row(a, i), y = row(b, i);
        if (x == y) continue;
        SaveDifference d;
        d.kind  = "space";
        d.field = std::to_string(i);
        d.a     = x;
        d.b     = y;
        add(std::move(d));
    }
}

// GENR (S23a): the seed, the generator per template, per generator (by name) its versions and slot table, the space
// count and the combined plan hash ("generation generator test_terrain pinned: 1 != 2")
template <class Add>
void diffGeneration(const GenerationSection& a, const GenerationSection& b, Add& add)
{
    auto one = [&](const std::string& component, const std::string& field, const Json& x, const Json& y)
    {
        if (x == y) return;
        SaveDifference d;
        d.kind      = "generation";
        d.component = component;
        d.field     = field;
        d.a         = x;
        d.b         = y;
        add(std::move(d));
    };
    one("seed", "", Json(std::to_string(a.seed)), Json(std::to_string(b.seed)));
    for (size_t i = 0; i < std::max(a.templateGenerators.size(), b.templateGenerators.size()); ++i)
        one("template " + std::to_string(i), "generator",
            i < a.templateGenerators.size() ? Json(a.templateGenerators[i]) : Json(),
            i < b.templateGenerators.size() ? Json(b.templateGenerators[i]) : Json());
    std::map<std::string, const GenerationSection::Generator*> ga, gb;
    for (const auto& g : a.generators) ga[g.name] = &g;
    for (const auto& g : b.generators) gb[g.name] = &g;
    std::set<std::string> names;
    for (const auto& kv : ga) names.insert(kv.first);
    for (const auto& kv : gb) names.insert(kv.first);
    auto slots = [](const GenerationSection::Generator* g)
    {
        Json::array s;
        if (g)
            for (const auto& slot : g->slots) s.push_back(Json::array{slot.name, double(slot.flags)});
        return Json(s);
    };
    for (const std::string& n : names)
    {
        const auto* x = ga.count(n) ? ga[n] : nullptr;
        const auto* y = gb.count(n) ? gb[n] : nullptr;
        one("generator " + n, "pinned", x ? Json(double(x->pinned)) : Json(), y ? Json(double(y->pinned)) : Json());
        one("generator " + n, "current", x ? Json(double(x->current)) : Json(), y ? Json(double(y->current)) : Json());
        one("generator " + n, "slots", slots(x), slots(y));
    }
    one("spaces", "", Json(double(a.spaceCount)), Json(double(b.spaceCount)));
    one("combined_plan_hash", "", Json(std::to_string(a.combinedHash)), Json(std::to_string(b.combinedHash)));
}

// CREC (S23a, S23b): per chunk record, one-sided records and the version / stamped / tombstone differences ("record
// space 0 chunk (4, 5): ...")
template <class Add>
void diffChunkRecords(const ChunkRecordSection& a, const ChunkRecordSection& b, Add& add)
{
    using Key = std::tuple<SpaceId, int32_t, int32_t>;
    std::map<Key, const ChunkRecordSection::Record*> ra, rb;
    for (const auto& sp : a.spaces)
        for (const auto& r : sp.records) ra[Key(sp.space, r.x, r.z)] = &r;
    for (const auto& sp : b.spaces)
        for (const auto& r : sp.records) rb[Key(sp.space, r.x, r.z)] = &r;
    std::set<Key> keys;
    for (const auto& kv : ra) keys.insert(kv.first);
    for (const auto& kv : rb) keys.insert(kv.first);
    auto state = [](const ChunkRecordSection::Record* r)
    {
        if (!r) return Json();
        Json::array keys;
        for (const InstanceKey& k : r->tombstones)
            keys.push_back(Json::array{Json(double(k.generatorSlot)), Json(double(k.counter))});
        return Json(Json::object{
            {"placement_version", double(r->placementVersion)}, {"stamped", r->stamped}, {"tombstones", keys}});
    };
    for (const Key& k : keys)
    {
        const Json x = state(ra.count(k) ? ra[k] : nullptr), y = state(rb.count(k) ? rb[k] : nullptr);
        if (x == y) continue;
        SaveDifference d;
        d.kind      = "record";
        d.component = "space " + std::to_string(std::get<0>(k)) + " chunk (" + std::to_string(std::get<1>(k)) + ", " +
                      std::to_string(std::get<2>(k)) + ")";
        d.a = x;
        d.b = y;
        add(std::move(d));
    }
}

// TERR (S22): per chunk with terrain state, one-sided chunks, sequence differences and per chunk the number of
// differing delta samples with the first one ("terrain space 0 chunk (1, 2): ...")
template <class Add>
void diffTerrain(const TerrainSection& a, const TerrainSection& b, Add& add)
{
    using Key = std::tuple<SpaceId, int32_t, int32_t>;
    std::map<Key, std::pair<const TerrainSection::Chunk*, uint16_t>> ca, cb;
    for (const auto& sp : a.spaces)
        for (const auto& c : sp.chunks) ca[Key(sp.space, c.x, c.z)] = {&c, sp.N};
    for (const auto& sp : b.spaces)
        for (const auto& c : sp.chunks) cb[Key(sp.space, c.x, c.z)] = {&c, sp.N};
    std::set<Key> keys;
    for (const auto& kv : ca) keys.insert(kv.first);
    for (const auto& kv : cb) keys.insert(kv.first);
    auto state = [](const TerrainSection::Chunk* c)
    {
        if (!c) return Json();
        return Json(Json::object{{"sequence", double(c->sequence)}, {"deltas", c->hasDeltas}});
    };
    for (const Key& k : keys)
    {
        auto ia = ca.find(k), ib = cb.find(k);
        const TerrainSection::Chunk* x = ia == ca.end() ? nullptr : ia->second.first;
        const TerrainSection::Chunk* y = ib == cb.end() ? nullptr : ib->second.first;
        SaveDifference d;
        d.kind      = "terrain";
        d.component = "space " + std::to_string(std::get<0>(k)) + " chunk (" + std::to_string(std::get<1>(k)) + ", " +
                      std::to_string(std::get<2>(k)) + ")";
        if (!x || !y || x->sequence != y->sequence || x->hasDeltas != y->hasDeltas)
        {
            SaveDifference s = d;
            s.a              = state(x);
            s.b              = state(y);
            add(std::move(s));
            // A one-sided chunk is one difference; otherwise the deltas are compared too
            if (!x || !y) continue;
        }
        // The same N (equal spacing) unless the saves come from different setups (also without deltas: the layouts
        // are incompatible)
        const size_t N = ia->second.second;
        if (ib->second.second != ia->second.second)
        {
            d.a = Json("N " + std::to_string(ia->second.second));
            d.b = Json("N " + std::to_string(ib->second.second));
            add(std::move(d));
            continue;
        }
        if (!x->hasDeltas || !y->hasDeltas || x->deltas == y->deltas) continue;
        size_t count = 0, first = 0;
        for (size_t i = 0; i < x->deltas.size(); ++i)
            if (x->deltas[i] != y->deltas[i])
            {
                if (count == 0) first = i;
                count++;
            }
        d.field = std::to_string(count) + " samples differ, first (" + std::to_string(first % N) + ", " +
                  std::to_string(first / N) + ")";
        d.a = Json(double(x->deltas[first]));
        d.b = Json(double(y->deltas[first]));
        add(std::move(d));
    }
}

Json actionsJson(const ParsedSave& p)
{
    Json::array a;
    for (const ParsedSave::Action& x : p.actions)
        a.push_back(Json::object{{"peer", double(x.peer)},
                                 {"sequence", std::to_string(x.sequence)},
                                 {"player", entityJson(x.player)},
                                 {"action", double(x.action)},
                                 {"aim", Json::array{x.aim.x(), x.aim.y(), x.aim.z()}}});
    return a;
}
}  // namespace

Json SaveDifference::toJson() const
{
    Json::object o{{"kind", kind}};
    if (entity.valid() || kind == "flags" || kind.compare(0, 4, "only") == 0 || kind.compare(0, 9, "component") == 0 ||
        kind == "field")
        o["entity"] = entityJson(entity);
    if (!component.empty()) o["component"] = component;
    if (!field.empty()) o["field"] = field;
    if (!className.empty()) o["class"] = className;
    if (!position.is_null()) o["position"] = position;
    if (!a.is_null() || kind == "field") o["a"] = a;
    if (!b.is_null() || kind == "field") o["b"] = b;
    return o;
}

std::string SaveDifference::describe() const
{
    const std::string e = "entity " + std::to_string(entity.index) + ":" + std::to_string(entity.generation);
    if (kind == "field") return e + " " + component + "." + field + ": " + short_(a) + " != " + short_(b);
    if (kind == "only_in_a" || kind == "only_in_b")
        return e + " only in " + kind.substr(8) + (className.empty() ? "" : " (" + className + ")");
    if (kind == "component_only_in_a" || kind == "component_only_in_b")
        return e + " " + component + " only in " + kind.substr(18);
    if (kind == "terrain")
        return "terrain " + component + (field.empty() ? "" : " " + field) + ": " + short_(a) + " != " + short_(b);
    if (kind == "generation" || kind == "record")
        return kind + " " + component + (field.empty() ? "" : " " + field) + ": " + short_(a) + " != " + short_(b);
    if (kind == "space")
        return "space " + field + ": " + (a.is_string() ? a.string_value() : "none") + " vs " +
               (b.is_string() ? b.string_value() : "none");
    if (kind == "section_only_in_a" || kind == "section_only_in_b")
        return "section " + component + " only in " + kind.substr(16);
    if (kind == "flags")
        return "entity index " + std::to_string(entity.index) + " " + kind + ": " + short_(a) + " != " + short_(b);
    return kind + (component.empty() ? "" : " " + component) + ": " + short_(a) + " != " + short_(b);
}

SaveDiff diffSaves(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b, const DiffFilter& filter)
{
    SaveDiff result;
    std::map<std::string, double> kinds, perComponent;
    std::vector<SaveDifference> all;
    auto add = [&](SaveDifference d)
    {
        if (filter.hasEntity && !d.entity.valid() && d.kind != "flags") return;
        if (filter.hasEntity && (d.entity.index != filter.entityIndex ||
                                 (filter.hasGeneration && d.entity.generation != filter.entityGeneration)))
            return;
        if (!filter.components.empty() &&
            std::find(filter.components.begin(), filter.components.end(), d.component) == filter.components.end())
            return;
        kinds[d.kind] += 1;
        if (!d.component.empty()) perComponent[d.component] += 1;
        all.push_back(std::move(d));
    };
    if (a == b)
    {
        result.summary = Json::object{{"kinds", Json::object{}}, {"components", Json::object{}}};
        return result;
    }
    ParsedSave pa, pb;
    std::string error;
    if (!parseSave(a, pa, error))
    {
        result.error = "a: " + error;
        return result;
    }
    if (!parseSave(b, pb, error))
    {
        result.error = "b: " + error;
        return result;
    }
    Decoded da, db;
    if (!decodeAll(a, pa, da, error))
    {
        result.error = "a: " + error;
        return result;
    }
    if (!decodeAll(b, pb, db, error))
    {
        result.error = "b: " + error;
        return result;
    }

    // World-level state
    auto world = [&](const std::string& kind, const Json& x, const Json& y, const std::string& component = "")
    {
        if (x == y) return;
        SaveDifference d;
        d.kind      = kind;
        d.component = component;
        d.a         = x;
        d.b         = y;
        add(std::move(d));
    };
    world("step", Json(std::to_string(pa.steps)), Json(std::to_string(pb.steps)));
    world("time", Json(pa.time), Json(pb.time));
    world("rng", Json(Json::array{std::to_string(pa.rngState), std::to_string(pa.rngIncrement)}),
          Json(Json::array{std::to_string(pb.rngState), std::to_string(pb.rngIncrement)}));
    world("counter", Json(std::to_string(pa.persistentCounter)), Json(std::to_string(pb.persistentCounter)));
    {
        // With the generations: they decide the handles of future entities
        Json::array fa, fb;
        for (uint32_t i : pa.freeList)
            fa.push_back(Json::array{double(i), double(i < pa.slots.size() ? pa.slots[i].generation : 0)});
        for (uint32_t i : pb.freeList)
            fb.push_back(Json::array{double(i), double(i < pb.slots.size() ? pb.slots[i].generation : 0)});
        world("free_list", fa, fb);
    }
    world("actions", actionsJson(pa), actionsJson(pb));
    // Optional sections (D7): present on one side only, then one comparison per tag
    auto section = [&](uint32_t tag, bool inA, bool inB)
    {
        if (inA == inB) return;
        SaveDifference d;
        d.kind      = inA ? "section_only_in_a" : "section_only_in_b";
        d.component = tagName(tag);
        add(std::move(d));
    };
    section(kTagSpaces, pa.spaces.has_value(), pb.spaces.has_value());
    if (pa.spaces && pb.spaces) diffSpaceTables(*pa.spaces, *pb.spaces, add);
    section(kTagGeneration, pa.generation.has_value(), pb.generation.has_value());
    if (pa.generation && pb.generation) diffGeneration(*pa.generation, *pb.generation, add);
    section(kTagChunkRecords, pa.chunkRecords.has_value(), pb.chunkRecords.has_value());
    if (pa.chunkRecords || pb.chunkRecords)
        diffChunkRecords(pa.chunkRecords ? *pa.chunkRecords : ChunkRecordSection(),
                         pb.chunkRecords ? *pb.chunkRecords : ChunkRecordSection(), add);
    section(kTagTerrain, pa.terrain.has_value(), pb.terrain.has_value());
    {
        TerrainSection ta, tb;
        std::string e;
        if (pa.terrain && !parseTerrainSection(a.data() + pa.terrain->begin, pa.terrain->end - pa.terrain->begin,
                                               pa.terrain->version, ta, e))
        {
            result.error = "a: TERR: " + e;
            return result;
        }
        if (pb.terrain && !parseTerrainSection(b.data() + pb.terrain->begin, pb.terrain->end - pb.terrain->begin,
                                               pb.terrain->version, tb, e))
        {
            result.error = "b: TERR: " + e;
            return result;
        }
        diffTerrain(ta, tb, add);
    }
    {
        std::map<std::string, bool> names;
        for (auto& kv : da.events) names[kv.first] = true;
        for (auto& kv : db.events) names[kv.first] = true;
        for (auto& kv : names)
        {
            auto ia = da.events.find(kv.first), ib = db.events.find(kv.first);
            const Json x = ia == da.events.end() ? Json(Json::array{}) : Json(ia->second);
            const Json y = ib == db.events.end() ? Json(Json::array{}) : Json(ib->second);
            world("events", x, y, kv.first);
        }
    }

    // Per entity
    auto classOf = [](const std::vector<uint8_t>& data, const ParsedSave& p, const Decoded& d,
                      uint32_t index) -> std::string
    {
        auto it = d.components.find("Behaviour");
        if (it == d.components.end()) return "";
        const Value* v = it->second.find(index);
        if (!v) return "";
        const Json& j        = jsonOf(*v, data, p, *d.types.at("Behaviour"));
        const std::string* n = p.schema.className(uint32_t(j["classId"].number_value()));
        return n ? *n : "";
    };
    auto positionOf = [](const std::vector<uint8_t>& data, const ParsedSave& p, const Decoded& d,
                         uint32_t index) -> Json
    {
        auto it = d.components.find("Transform");
        if (it == d.components.end()) return Json();
        const Value* v = it->second.find(index);
        return v ? jsonOf(*v, data, p, *d.types.at("Transform"))["position"] : Json();
    };
    const size_t slots  = std::max(pa.slots.size(), pb.slots.size());
    const uint8_t alive = uint8_t(SlotState::Alive);
    for (uint32_t i = 1; i < slots; ++i)
    {
        const bool inA = i < pa.slots.size() && pa.slots[i].state == alive;
        const bool inB = i < pb.slots.size() && pb.slots[i].state == alive;
        if (!inA && !inB) continue;
        const Entity ea{i, inA ? pa.slots[i].generation : 0}, eb{i, inB ? pb.slots[i].generation : 0};
        if (inA != inB)
        {
            SaveDifference d;
            d.kind      = inA ? "only_in_a" : "only_in_b";
            d.entity    = inA ? ea : eb;
            d.className = inA ? classOf(a, pa, da, i) : classOf(b, pb, db, i);
            d.position  = inA ? positionOf(a, pa, da, i) : positionOf(b, pb, db, i);
            add(std::move(d));
            continue;
        }
        if (ea.generation != eb.generation)
        {
            // The slot was reused: two different entities, each reported with its class and position (a filter on
            // either handle finds its side)
            SaveDifference left;
            left.kind      = "only_in_a";
            left.entity    = ea;
            left.className = classOf(a, pa, da, i);
            left.position  = positionOf(a, pa, da, i);
            add(std::move(left));
            SaveDifference right;
            right.kind      = "only_in_b";
            right.entity    = eb;
            right.className = classOf(b, pb, db, i);
            right.position  = positionOf(b, pb, db, i);
            add(std::move(right));
            continue;
        }
        if (pa.slots[i].flags != pb.slots[i].flags)
        {
            SaveDifference d;
            d.kind   = "flags";
            d.entity = ea;
            d.a      = double(pa.slots[i].flags);
            d.b      = double(pb.slots[i].flags);
            add(std::move(d));
        }
        std::map<std::string, bool> names;
        for (auto& kv : da.components)
            if (kv.second.find(i)) names[kv.first] = true;
        for (auto& kv : db.components)
            if (kv.second.find(i)) names[kv.first] = true;
        for (auto& kv : names)
        {
            const std::string& name = kv.first;
            auto ca = da.components.find(name), cb = db.components.find(name);
            const Value* va = ca != da.components.end() ? ca->second.find(i) : nullptr;
            const Value* vb = cb != db.components.end() ? cb->second.find(i) : nullptr;
            if (!va || !vb)
            {
                SaveDifference d;
                d.kind      = va ? "component_only_in_a" : "component_only_in_b";
                d.entity    = ea;
                d.component = name;
                add(std::move(d));
                continue;
            }
            // Equal bytes of the same type: equal (no JSON comparison)
            const SchemaType* ta = da.types[name];
            const SchemaType* tb = db.types[name];
            // Equal bytes are equal values only with the same schema (nested structs included); a Behaviour's classId
            // is compared by class name (two builds may number the classes differently)
            std::string reason;
            const bool sameType =
                name != "Behaviour" && compareTypes(pa.schema, *ta, pb.schema, *tb, reason) == SchemaMatch::Same;
            if (sameType && va->end - va->begin == vb->end - vb->begin &&
                std::equal(a.begin() + std::ptrdiff_t(va->begin), a.begin() + std::ptrdiff_t(va->end),
                           b.begin() + std::ptrdiff_t(vb->begin)))
                continue;
            Json ja = jsonOf(*va, a, pa, *ta);
            Json jb = jsonOf(*vb, b, pb, *tb);
            if (name == "Behaviour")
            {
                auto byName = [](const Json& j, const ParsedSave& p)
                {
                    Json::object o         = j.object_items();
                    const std::string* cls = p.schema.className(uint32_t(j["classId"].number_value()));
                    o["classId"]           = cls ? Json(*cls) : j["classId"];
                    return Json(o);
                };
                ja = byName(ja, pa);
                jb = byName(jb, pb);
            }
            if (ja == jb) continue;
            std::map<std::string, bool> fields;
            for (auto& f : ja.object_items()) fields[f.first] = true;
            for (auto& f : jb.object_items()) fields[f.first] = true;
            for (auto& f : fields)
            {
                const Json x = ja[f.first], y = jb[f.first];
                if (x == y) continue;
                SaveDifference d;
                d.kind      = "field";
                d.entity    = ea;
                d.component = name;
                d.field     = f.first;
                d.a         = x;
                d.b         = y;
                add(std::move(d));
            }
        }
    }

    result.total       = all.size();
    const size_t begin = std::min(filter.after, all.size());
    const size_t end   = filter.limit == 0 ? all.size() : std::min(all.size(), begin + filter.limit);
    result.differences.assign(std::make_move_iterator(all.begin() + std::ptrdiff_t(begin)),
                              std::make_move_iterator(all.begin() + std::ptrdiff_t(end)));
    Json::object k, c;
    for (auto& kv : kinds) k[kv.first] = kv.second;
    for (auto& kv : perComponent) c[kv.first] = kv.second;
    result.summary = Json::object{{"kinds", k}, {"components", c}};
    return result;
}

}  // namespace Engine
}  // namespace Saiga
