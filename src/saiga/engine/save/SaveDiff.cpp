/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// diffSaves: field-level comparison of two save blobs by name (S11).

#include "saiga/engine/save/SaveDiff.h"

#include "saiga/engine/ecs/Registry.h"
#include "saiga/engine/save/GenericDecode.h"
#include "saiga/engine/save/SaveReader.h"

#include <algorithm>
#include <map>

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
struct Values
{
    std::vector<Value> byIndex;
    const Value* find(uint32_t index) const
    {
        return index < byIndex.size() && byIndex[index].present ? &byIndex[index] : nullptr;
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
            // Grown to the largest index present (not to the slot count of every block)
            if (index >= values.size()) values.resize(size_t(index) + 1);
            if (values[index].present)
            {
                error = t.name + ": two values for entity index " + std::to_string(index) + " (or not a slot)";
                return false;
            }
            values[index] = std::move(v);
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
    if (entity.valid() || kind == "generation" || kind == "flags" || kind.compare(0, 4, "only") == 0 ||
        kind.compare(0, 9, "component") == 0 || kind == "field")
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
    if (kind == "generation" || kind == "flags")
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
        if (filter.hasEntity && (!d.entity.valid() && d.kind != "generation" && d.kind != "flags")) return;
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
        Json::array fa, fb;
        for (uint32_t i : pa.freeList) fa.push_back(double(i));
        for (uint32_t i : pb.freeList) fb.push_back(double(i));
        world("free_list", fa, fb);
    }
    world("actions", actionsJson(pa), actionsJson(pb));
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
            bool sameType        = ta->version == tb->version && ta->fields.size() == tb->fields.size();
            for (size_t k = 0; sameType && k < ta->fields.size(); ++k)
                sameType = ta->fields[k].name == tb->fields[k].name && ta->fields[k].type == tb->fields[k].type &&
                           ta->fields[k].enumValues == tb->fields[k].enumValues;
            if (sameType && va->end - va->begin == vb->end - vb->begin &&
                std::equal(a.begin() + std::ptrdiff_t(va->begin), a.begin() + std::ptrdiff_t(va->end),
                           b.begin() + std::ptrdiff_t(vb->begin)))
                continue;
            const Json& ja = jsonOf(*va, a, pa, *ta);
            const Json& jb = jsonOf(*vb, b, pb, *tb);
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
