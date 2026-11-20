/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// SchemaTable: the saved types of a world, their binary form in a save and the comparison with the registered types.

#include "saiga/engine/save/SchemaTable.h"

#include "saiga/engine/world/World.h"

#include <set>

#include <unordered_set>

namespace Saiga
{
namespace Engine
{
namespace
{
bool scalarType(const std::string& t)
{
    static const std::set<std::string> scalars = {"bool",   "int8",         "int16",  "int32",  "int64", "uint8",
                                                  "uint16", "uint32",       "uint64", "double", "float", "string",
                                                  "Entity", "PersistentId", "enum",   "quatd",  "quatf"};
    if (scalars.count(t)) return true;
    // vec2d .. vec4f, mat2d .. mat4f
    if (t.size() == 5 && (t.compare(0, 3, "vec") == 0 || t.compare(0, 3, "mat") == 0) && t[3] >= '1' && t[3] <= '9' &&
        (t[4] == 'd' || t[4] == 'f'))
        return true;
    return false;
}

constexpr uint32_t kMaxSchemaEntries = 1 << 20;
constexpr size_t kMaxNameLength      = 256;

// A field type string of a saved schema: bounded length and container nesting (decoding recurses over it), array
// lengths below 2^31 (decoding is bounded by the input and the zero-byte budget)
bool validFieldType(const std::string& type)
{
    if (type.empty() || type.size() > kMaxNameLength) return false;
    std::string t = type;
    for (int nesting = 0;; ++nesting)
    {
        if (nesting > 8) return false;
        if (t.compare(0, 7, "vector<") == 0 && t.back() == '>')
            t = t.substr(7, t.size() - 8);
        else if (t.compare(0, 6, "array<") == 0 && t.back() == '>')
        {
            const size_t comma = t.rfind(',');
            if (comma == std::string::npos || comma < 7) return false;
            const std::string count = t.substr(comma + 1, t.size() - comma - 2);
            if (count.empty() || count.size() > 10) return false;
            for (char c : count)
                if (c < '0' || c > '9') return false;
            if (std::stoull(count) > 0x7FFFFFFFull) return false;
            t = t.substr(6, comma - 6);
        }
        else
        {
            // A scalar, math or struct name
            if (t.empty()) return false;
            for (char c : t)
                if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_'))
                    return false;
            return true;
        }
    }
}

template <class T>
bool readCount(BinaryReader& r, T& n, uint64_t max, const char* what, std::string& error)
{
    uint64_t v = 0;
    r.read(v);
    if (!r.ok()) return false;
    if (v > max)
    {
        error = std::string("schema: too many ") + what;
        return false;
    }
    n = T(v);
    return true;
}
}  // namespace

std::string structNameOf(const std::string& fieldType)
{
    std::string t = fieldType;
    for (;;)
    {
        if (t.compare(0, 7, "vector<") == 0 && t.back() == '>')
            t = t.substr(7, t.size() - 8);
        else if (t.compare(0, 6, "array<") == 0 && t.back() == '>')
        {
            const size_t comma = t.rfind(',');
            if (comma == std::string::npos || comma < 6) return "";
            t = t.substr(6, comma - 6);
        }
        else
            break;
    }
    return scalarType(t) ? "" : t;
}

SchemaTable SchemaTable::fromWorld(const World& world)
{
    SchemaTable table;
    const Registry& reg = world.registry();
    std::vector<SchemaType> nested;
    // Components in ID order (Transform first), then events in ID order, then the nested structs
    for (size_t id = 0; id < kMaxComponents; ++id)
    {
        if (!reg.savedComponent(ComponentId(id))) continue;
        if (id == Transform::id)
            table.types.push_back(describeType<Transform>(SchemaKind::Component, Transform::id));
        else
        {
            const PoolBase* p = reg.poolById(ComponentId(id));
            table.types.push_back(p->describe());
            p->describeNested(nested);
        }
    }
    for (const auto& q : world.events.all())
        if (q)
        {
            table.types.push_back(q->describe());
            q->describeNested(nested);
        }
    // Each nested struct once (two components may share one)
    for (SchemaType& t : nested)
    {
        bool seen = false;
        for (const SchemaType& u : table.types) seen = seen || (u.kind == SchemaKind::Struct && u.name == t.name);
        if (!seen) table.types.push_back(std::move(t));
    }
    for (const ClassList::ClassInfo& c : world.classList().all())
        if (c.factory) table.classes.push_back({uint32_t(c.id), c.name});
    table.buildIndex();
    return table;
}

void SchemaTable::buildIndex()
{
    index.clear();
    for (size_t i = 0; i < types.size(); ++i)
        index.emplace(std::to_string(int(types[i].kind)) + ":" + types[i].name, i);
}

const SchemaType* SchemaTable::find(SchemaKind kind, const std::string& name) const
{
    if (index.size() != types.size())
    {
        // Not indexed (types changed directly): a linear search
        for (const SchemaType& t : types)
            if (t.kind == kind && t.name == name) return &t;
        return nullptr;
    }
    auto it = index.find(std::to_string(int(kind)) + ":" + name);
    return it == index.end() ? nullptr : &types[it->second];
}

const std::string* SchemaTable::className(uint32_t id) const
{
    for (const auto& c : classes)
        if (c.first == id) return &c.second;
    return nullptr;
}

void SchemaTable::write(BinaryWriter& w) const
{
    w.write(uint64_t(types.size()));
    for (const SchemaType& t : types)
    {
        w.write(uint8_t(t.kind));
        w.write(t.name);
        w.write(t.id);
        w.write(t.version);
        w.write(uint64_t(t.fields.size()));
        for (const FieldInfo& f : t.fields)
        {
            w.write(f.name);
            w.write(f.type);
            w.write(f.enumValues);
        }
    }
    w.write(uint64_t(classes.size()));
    for (const auto& c : classes)
    {
        w.write(c.first);
        w.write(c.second);
    }
}

bool SchemaTable::read(BinaryReader& r, std::string& error)
{
    types.clear();
    classes.clear();
    std::unordered_set<std::string> seenTypes;
    std::unordered_set<uint32_t> seenClasses;
    size_t n = 0;
    if (!readCount(r, n, kMaxSchemaEntries, "types", error)) return false;
    for (size_t i = 0; i < n && r.ok(); ++i)
    {
        SchemaType t;
        uint8_t kind = 0;
        r.read(kind);
        if (r.ok() && kind > uint8_t(SchemaKind::Event))
        {
            error = "schema: unknown type kind " + std::to_string(kind);
            return false;
        }
        t.kind = SchemaKind(kind);
        r.read(t.name);
        if (r.ok() && (t.name.empty() || t.name.size() > kMaxNameLength))
        {
            error = "schema: a type name is empty or too long";
            return false;
        }
        r.read(t.id);
        r.read(t.version);
        size_t fields = 0;
        if (!readCount(r, fields, kMaxSchemaEntries, "fields", error)) return false;
        for (size_t k = 0; k < fields && r.ok(); ++k)
        {
            FieldInfo f;
            r.read(f.name);
            r.read(f.type);
            r.read(f.enumValues);
            if (r.ok() && (f.name.size() > kMaxNameLength || !validFieldType(f.type)))
            {
                error = "schema: " + t.name + " has a malformed field type";
                return false;
            }
            t.fields.push_back(std::move(f));
        }
        if (r.ok() && !seenTypes.insert(std::to_string(int(t.kind)) + ":" + t.name).second)
        {
            error = "schema: the type " + t.name + " is listed twice";
            return false;
        }
        types.push_back(std::move(t));
    }
    size_t nc = 0;
    if (r.ok() && !readCount(r, nc, kMaxSchemaEntries, "classes", error)) return false;
    for (size_t i = 0; i < nc && r.ok(); ++i)
    {
        std::pair<uint32_t, std::string> c;
        r.read(c.first);
        r.read(c.second);
        if (r.ok() && !seenClasses.insert(c.first).second)
        {
            error = "schema: the class ID " + std::to_string(c.first) + " is listed twice";
            return false;
        }
        classes.push_back(std::move(c));
    }
    buildIndex();
    if (!r.ok())
    {
        error = "schema: " + r.error();
        return false;
    }
    return true;
}

namespace
{
bool sameFields(const SchemaType& a, const SchemaType& b)
{
    if (a.fields.size() != b.fields.size()) return false;
    for (size_t i = 0; i < a.fields.size(); ++i)
        if (a.fields[i].name != b.fields[i].name || a.fields[i].type != b.fields[i].type ||
            a.fields[i].enumValues != b.fields[i].enumValues)
            return false;
    return true;
}

// Same version and fields, the nested structs too (visited guards self-containing structs)
bool sameRecursive(const SchemaTable& saved, const SchemaType& s, const SchemaTable& current, const SchemaType& c,
                   std::set<std::string>& visited)
{
    if (s.version != c.version || !sameFields(s, c)) return false;
    for (const FieldInfo& f : s.fields)
    {
        const std::string nestedName = structNameOf(f.type);
        if (nestedName.empty() || !visited.insert(nestedName).second) continue;
        const SchemaType* ns = saved.findStruct(nestedName);
        const SchemaType* nc = current.findStruct(nestedName);
        if (!ns || !nc || !sameRecursive(saved, *ns, current, *nc, visited)) return false;
    }
    return true;
}
}  // namespace

SchemaMatch compareTypes(const SchemaTable& saved, const SchemaType& savedType, const SchemaTable& current,
                         const SchemaType& currentType, std::string& reason)
{
    if (savedType.version > currentType.version)
    {
        reason = savedType.name + " was saved with version " + std::to_string(savedType.version) +
                 ", newer than the registered version " + std::to_string(currentType.version);
        return SchemaMatch::Incompatible;
    }
    if (savedType.version < currentType.version) return SchemaMatch::Migrate;
    std::set<std::string> visited;
    if (sameRecursive(saved, savedType, current, currentType, visited)) return SchemaMatch::Same;
    // The same version: a field that exists in both must keep its type (R-SAVE-1)
    for (const FieldInfo& f : savedType.fields)
        for (const FieldInfo& g : currentType.fields)
            if (f.name == g.name && f.type != g.type)
            {
                reason = savedType.name + "." + f.name + " changed its type from " + f.type + " to " + g.type +
                         " without a version increment (R-SAVE-1)";
                return SchemaMatch::Incompatible;
            }
    return SchemaMatch::Compatible;
}

}  // namespace Engine
}  // namespace Saiga
