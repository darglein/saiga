/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/content/Prefab.h"

#include "saiga/core/math/ChunkId.h"
#include "saiga/engine/content/gen/FixedMath.h"
#include "saiga/engine/core/Hash.h"
#include "saiga/engine/data/DataFile.h"

#include <cmath>
#include <set>

namespace Saiga
{
namespace Engine
{
using Json = json11::Json;

namespace
{
// Footprints and rectangles within +-2^20 m
constexpr int64_t kRectLimit = int64_t(1) << 20;

class Parser
{
   public:
    explicit Parser(const std::string& file) : file(file) {}
    std::string error;

    bool fail(const std::string& key, const std::string& message)
    {
        if (error.empty()) error = file + ": " + key + ": " + message;
        return false;
    }
    // Unknown keys are errors (R-DATA-1)
    bool keys(const Json& object, const std::string& where, const std::set<std::string>& allowed)
    {
        if (!object.is_object()) return fail(where, "expected an object");
        for (const auto& kv : object.object_items())
            if (!allowed.count(kv.first)) return fail(where.empty() ? kv.first : where + "." + kv.first, "unknown key");
        return true;
    }
    bool integer(const Json& v, const std::string& key, int64_t lo, int64_t hi, int64_t& out)
    {
        if (!v.is_number()) return fail(key, "expected an integer");
        const double d = v.number_value();
        if (!std::isfinite(d) || d != std::floor(d)) return fail(key, "expected an integer");
        if (d < double(lo) || d > double(hi))
            return fail(key, "out of range [" + std::to_string(lo) + ", " + std::to_string(hi) + "]");
        out = int64_t(d);
        return true;
    }
    // A coordinate in metres that must be an exact multiple of 1 / scale (scale a power of two: v * scale is exact)
    bool exact(const Json& v, const std::string& key, int64_t scale, int64_t limit, int64_t& out)
    {
        if (!v.is_number()) return fail(key, "expected a number");
        const double m = v.number_value() * double(scale);
        if (!std::isfinite(m) || std::abs(m) > double(limit)) return fail(key, "out of range");
        if (m != std::floor(m))
            return fail(key, "not an exact multiple of 1/" + std::to_string(scale) + " m (" + v.dump() + ")");
        out = int64_t(m);
        return true;
    }
    bool rect(const Json& v, const std::string& key, PrefabRect& out)
    {
        if (!v.is_array() || v.array_items().size() != 4) return fail(key, "expected [x0, z0, x1, z1]");
        int64_t r[4];
        for (int i = 0; i < 4; ++i)
            if (!integer(v[size_t(i)], key + "[" + std::to_string(i) + "]", -kRectLimit, kRectLimit, r[i]))
                return false;
        if (r[0] >= r[2] || r[1] >= r[3]) return fail(key, "an empty rectangle (x0 < x1 and z0 < z1)");
        out = {int32_t(r[0]), int32_t(r[1]), int32_t(r[2]), int32_t(r[3])};
        return true;
    }
    // x, y, z: x and z in 1/256 m, y in height units
    bool position(const Json& v, const std::string& key, int64_t& x, int32_t& y, int64_t& z)
    {
        if (!v.is_array() || v.array_items().size() != 3) return fail(key, "expected [x, y, z]");
        int64_t yy = 0;
        if (!exact(v[0], key + "[0]", kPosUnitsPerMetre, kRectLimit * kPosUnitsPerMetre, x) ||
            !exact(v[1], key + "[1]", kHeightUnitsPerMetre, kTerrainHeightLimit, yy) ||
            !exact(v[2], key + "[2]", kPosUnitsPerMetre, kRectLimit * kPosUnitsPerMetre, z))
            return false;
        y = int32_t(yy);
        return true;
    }
    bool yaw(const Json& object, const char* name, const std::string& key, uint16_t& out)
    {
        if (object[name].is_null()) return true;
        int64_t deg = 0;
        if (!integer(object[name], key, -3600, 3600, deg)) return false;
        out = prefabYaw(deg);
        return true;
    }

   private:
    std::string file;
};

bool inside(const PrefabRect& inner, const PrefabRect& outer)
{
    return inner.x0 >= outer.x0 && inner.z0 >= outer.z0 && inner.x1 <= outer.x1 && inner.z1 <= outer.z1;
}

// A point in 1/256 m inside the half-open footprint
bool insidePoint(int64_t x, int64_t z, const PrefabRect& f)
{
    return x >= int64_t(f.x0) * kPosUnitsPerMetre && x < int64_t(f.x1) * kPosUnitsPerMetre &&
           z >= int64_t(f.z0) * kPosUnitsPerMetre && z < int64_t(f.z1) * kPosUnitsPerMetre;
}
}  // namespace

uint16_t prefabYaw(int64_t degrees)
{
    // A full turn adds exactly 65536 steps (wrapped away): reduced first, so the product cannot overflow
    return uint16_t(uint64_t(floorDiv(floorMod(degrees, 360) * 65536 + 180, 360)) & 0xFFFF);
}

std::string prefabPath(const std::string& dataDirectory, const std::string& name, uint16_t version)
{
    const std::string base = (dataDirectory.empty() ? std::string(".") : dataDirectory) + "/prefabs/" + name;
    return version == 1 ? base + ".json" : base + ".v" + std::to_string(version) + ".json";
}

bool parsePrefab(const Json& json, const std::string& file, Prefab& out, std::string& error)
{
    Parser p(file);
    Prefab r;
    r.name    = out.name;
    r.version = out.version;
    r.file    = file;
    auto done = [&]()
    {
        error = p.error;
        return false;
    };
    if (!p.keys(
            json, "",
            {"format", "footprint", "terrain_patch", "exclusion", "entities", "static_geometry", "spaces", "pieces"}))
        return done();
    if (!json["pieces"].is_null())
    {
        p.fail("pieces", "placed pieces in prefabs come with S30");
        return done();
    }
    int64_t format = 0;
    if (json["format"].is_null())
    {
        p.fail("format", "missing (\"format\": 1)");
        return done();
    }
    if (!p.integer(json["format"], "format", 1, 1, format)) return done();
    if (json["footprint"].is_null())
    {
        p.fail("footprint", "missing");
        return done();
    }
    if (!p.rect(json["footprint"], "footprint", r.footprint)) return done();

    const Json& patch = json["terrain_patch"];
    if (!patch.is_null())
    {
        if (!p.keys(patch, "terrain_patch", {"origin", "size", "heights", "weights"})) return done();
        PrefabPatch tp;
        int64_t v[4];
        const Json& o = patch["origin"];
        const Json& s = patch["size"];
        if (!o.is_array() || o.array_items().size() != 2)
        {
            p.fail("terrain_patch.origin", "expected [x, z]");
            return done();
        }
        if (!s.is_array() || s.array_items().size() != 2)
        {
            p.fail("terrain_patch.size", "expected [sx, sz]");
            return done();
        }
        if (!p.integer(o[0], "terrain_patch.origin[0]", -kRectLimit, kRectLimit, v[0]) ||
            !p.integer(o[1], "terrain_patch.origin[1]", -kRectLimit, kRectLimit, v[1]) ||
            !p.integer(s[0], "terrain_patch.size[0]", 1, 4096, v[2]) ||
            !p.integer(s[1], "terrain_patch.size[1]", 1, 4096, v[3]))
            return done();
        tp.originX       = int32_t(v[0]);
        tp.originZ       = int32_t(v[1]);
        tp.sizeX         = int32_t(v[2]);
        tp.sizeZ         = int32_t(v[3]);
        const size_t n   = size_t(tp.sizeX) * size_t(tp.sizeZ);
        const Json& hs   = patch["heights"];
        const Json& ws   = patch["weights"];
        const auto count = [&](const Json& a) { return a.is_array() ? a.array_items().size() : size_t(0); };
        if (count(hs) != n)
        {
            p.fail("terrain_patch.heights", "expected " + std::to_string(n) + " values (size " +
                                                std::to_string(tp.sizeX) + " x " + std::to_string(tp.sizeZ) +
                                                "), got " + std::to_string(count(hs)));
            return done();
        }
        if (count(ws) != n)
        {
            p.fail("terrain_patch.weights", "expected " + std::to_string(n) + " values (size " +
                                                std::to_string(tp.sizeX) + " x " + std::to_string(tp.sizeZ) +
                                                "), got " + std::to_string(count(ws)));
            return done();
        }
        tp.heights.resize(n);
        tp.weights.resize(n);
        for (size_t i = 0; i < n; ++i)
        {
            int64_t h = 0, w = 0;
            if (!p.integer(hs[i], "terrain_patch.heights[" + std::to_string(i) + "]", -2 * int64_t(kTerrainHeightLimit),
                           2 * int64_t(kTerrainHeightLimit), h) ||
                !p.integer(ws[i], "terrain_patch.weights[" + std::to_string(i) + "]", 0, 256, w))
                return done();
            tp.heights[i] = int32_t(h);
            tp.weights[i] = int32_t(w);
        }
        const PrefabRect area{tp.originX, tp.originZ, tp.originX + tp.sizeX, tp.originZ + tp.sizeZ};
        if (!inside(area, r.footprint))
        {
            p.fail("terrain_patch", "the patch lies outside the footprint");
            return done();
        }
        r.patch = std::move(tp);
    }

    const Json& exclusion = json["exclusion"];
    if (!exclusion.is_null())
    {
        if (!exclusion.is_array())
        {
            p.fail("exclusion", "expected an array of [x0, z0, x1, z1]");
            return done();
        }
        for (size_t i = 0; i < exclusion.array_items().size(); ++i)
        {
            const std::string key = "exclusion[" + std::to_string(i) + "]";
            PrefabRect e;
            if (!p.rect(exclusion[i], key, e)) return done();
            if (!inside(e, r.footprint))
            {
                p.fail(key, "outside the footprint");
                return done();
            }
            r.exclusion.push_back(e);
        }
    }

    const Json& spaces = json["spaces"];
    if (!spaces.is_null())
    {
        if (!spaces.is_array() || spaces.array_items().size() > 255)
        {
            p.fail("spaces", "expected an array of at most 255 {\"template\": name}");
            return done();
        }
        for (size_t i = 0; i < spaces.array_items().size(); ++i)
        {
            const std::string key = "spaces[" + std::to_string(i) + "]";
            if (!p.keys(spaces[i], key, {"template"})) return done();
            if (!spaces[i]["template"].is_string() || spaces[i]["template"].string_value().empty())
            {
                p.fail(key + ".template", "expected a template name");
                return done();
            }
            r.spaces.push_back(spaces[i]["template"].string_value());
        }
    }

    const Json& entities = json["entities"];
    if (!entities.is_null())
    {
        if (!entities.is_array())
        {
            p.fail("entities", "expected an array");
            return done();
        }
        for (size_t i = 0; i < entities.array_items().size(); ++i)
        {
            const Json& e         = entities[i];
            const std::string key = "entities[" + std::to_string(i) + "]";
            if (!p.keys(e, key, {"class", "position", "yaw_deg", "components", "space_link"})) return done();
            PrefabEntity pe;
            if (!e["class"].is_string() || e["class"].string_value().empty())
            {
                p.fail(key + ".class", "expected a class name");
                return done();
            }
            pe.className = e["class"].string_value();
            if (!e["position"].is_null() && !p.position(e["position"], key + ".position", pe.x, pe.y, pe.z))
                return done();
            if (!insidePoint(pe.x, pe.z, r.footprint))
            {
                p.fail(key + ".position", "outside the footprint");
                return done();
            }
            if (!p.yaw(e, "yaw_deg", key + ".yaw_deg", pe.yaw)) return done();
            pe.components = e["components"].is_null() ? Json(Json::object{}) : e["components"];
            if (!pe.components.is_object())
            {
                p.fail(key + ".components", "expected an object {component: {fields}}");
                return done();
            }
            for (const auto& kv : pe.components.object_items())
                if (!kv.second.is_null() && !kv.second.is_object())
                {
                    p.fail(key + ".components." + kv.first, "expected an object of fields");
                    return done();
                }
            const Json& link = e["space_link"];
            if (!link.is_null())
            {
                const std::string lk = key + ".space_link";
                if (!p.keys(link, lk, {"to", "link", "arrival", "arrival_yaw_deg"})) return done();
                PrefabSpaceLink sl;
                const std::string to = link["to"].is_string() ? link["to"].string_value() : "";
                if (to != "link" && to != "parent")
                {
                    p.fail(lk + ".to", "expected \"link\" or \"parent\"");
                    return done();
                }
                sl.toParent = to == "parent";
                int64_t n   = 0;
                if (!link["link"].is_null() && !p.integer(link["link"], lk + ".link", 0, 254, n)) return done();
                sl.link = uint8_t(n);
                if (!sl.toParent && sl.link >= r.spaces.size())
                {
                    p.fail(lk + ".link", "the prefab lists " + std::to_string(r.spaces.size()) + " spaces");
                    return done();
                }
                if (!link["arrival"].is_null() && !p.position(link["arrival"], lk + ".arrival", sl.x, sl.y, sl.z))
                    return done();
                if (!p.yaw(link, "arrival_yaw_deg", lk + ".arrival_yaw_deg", sl.arrivalYaw)) return done();
                pe.spaceLink = sl;
            }
            r.entities.push_back(std::move(pe));
        }
    }

    const Json& statics = json["static_geometry"];
    if (!statics.is_null())
    {
        if (!statics.is_array())
        {
            p.fail("static_geometry", "expected an array");
            return done();
        }
        for (size_t i = 0; i < statics.array_items().size(); ++i)
        {
            const Json& e         = statics[i];
            const std::string key = "static_geometry[" + std::to_string(i) + "]";
            if (!p.keys(e, key, {"mesh", "position", "yaw_deg"})) return done();
            PrefabStatic ps;
            if (!e["mesh"].is_string() || e["mesh"].string_value().empty())
            {
                p.fail(key + ".mesh", "expected a mesh name");
                return done();
            }
            ps.mesh = e["mesh"].string_value();
            if (!e["position"].is_null() && !p.position(e["position"], key + ".position", ps.x, ps.y, ps.z))
                return done();
            if (!insidePoint(ps.x, ps.z, r.footprint))
            {
                p.fail(key + ".position", "outside the footprint");
                return done();
            }
            if (!p.yaw(e, "yaw_deg", key + ".yaw_deg", ps.yaw)) return done();
            r.statics.push_back(std::move(ps));
        }
    }
    r.contentHash = prefabContentHash(r);
    out           = std::move(r);
    return true;
}

bool readPrefab(const std::string& dataDirectory, const std::string& name, uint16_t version, Prefab& out,
                std::string& error)
{
    const std::string file = prefabPath(dataDirectory, name, version);
    // No '.' (also no "..", no extension): "a.v2" at v1 would name the file of "a" at v2
    if (name.empty() || name.find('.') != std::string::npos || name[0] == '/' || name.find('\\') != std::string::npos)
    {
        error = "prefab '" + name + "': not a valid prefab name";
        return false;
    }
    if (version == 0)
    {
        error = file + ": prefab versions start at 1";
        return false;
    }
    Json json;
    if (!readJsonFile(file, json, error))
    {
        error = "prefab '" + name + "' v" + std::to_string(version) + ": " + error;
        return false;
    }
    // Into a temporary: a failed parse leaves out unchanged
    Prefab parsed;
    parsed.name    = name;
    parsed.version = version;
    if (!parsePrefab(json, file, parsed, error)) return false;
    out = std::move(parsed);
    return true;
}

uint64_t prefabContentHash(const Prefab& p)
{
    Hasher h(fnv1a64("saiga.content.prefab.v1"));
    h.addU32(1);
    h.addI32(p.footprint.x0).addI32(p.footprint.z0).addI32(p.footprint.x1).addI32(p.footprint.z1);
    h.addU8(p.patch ? 1 : 0);
    if (p.patch)
    {
        const PrefabPatch& t = *p.patch;
        h.addI32(t.originX).addI32(t.originZ).addI32(t.sizeX).addI32(t.sizeZ);
        for (int32_t v : t.heights) h.addI32(v);
        for (int32_t v : t.weights) h.addI32(v);
    }
    h.addU32(uint32_t(p.exclusion.size()));
    for (const PrefabRect& e : p.exclusion) h.addI32(e.x0).addI32(e.z0).addI32(e.x1).addI32(e.z1);
    h.addU32(uint32_t(p.entities.size()));
    for (const PrefabEntity& e : p.entities)
    {
        h.addString(e.className).addI64(e.x).addI32(e.y).addI64(e.z).addU32(e.yaw);
        h.addU8(e.spaceLink ? 1 : 0);
        if (e.spaceLink)
        {
            const PrefabSpaceLink& l = *e.spaceLink;
            h.addU8(l.toParent ? 1 : 0).addU32(l.link).addI64(l.x).addI32(l.y).addI64(l.z).addU32(l.arrivalYaw);
        }
        // Component names, not values (released prefab files are never edited, §15.3)
        h.addU32(uint32_t(e.components.object_items().size()));
        for (const auto& kv : e.components.object_items()) h.addString(kv.first);
    }
    h.addU32(uint32_t(p.statics.size()));
    for (const PrefabStatic& s : p.statics) h.addString(s.mesh).addI64(s.x).addI32(s.y).addI64(s.z).addU32(s.yaw);
    h.addU32(uint32_t(p.spaces.size()));
    for (const std::string& s : p.spaces) h.addString(s);
    return h.value();
}

}  // namespace Engine
}  // namespace Saiga
