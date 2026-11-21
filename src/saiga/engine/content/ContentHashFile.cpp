/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/content/ContentHashFile.h"

#include "saiga/engine/content/GenerationPipeline.h"
#include "saiga/engine/content/gen/FixedMath.h"
#include "saiga/engine/content/gen/IntNoise.h"
#include "saiga/engine/content/gen/TrigTable.h"
#include "saiga/engine/core/Hash.h"
#include "saiga/engine/core/JobPool.h"
#include "saiga/engine/data/DataFile.h"
#include "saiga/engine/world/World.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <set>

namespace Saiga
{
namespace Engine
{
using Json = json11::Json;

bool parseU64(const std::string& text, bool hex, uint64_t& out)
{
    const char* b = text.data();
    const char* e = text.data() + text.size();
    if (hex)
    {
        if (text.size() < 3 || text[0] != '0' || (text[1] != 'x' && text[1] != 'X')) return false;
        b += 2;
    }
    if (b == e) return false;
    uint64_t v = 0;
    auto r     = std::from_chars(b, e, v, hex ? 16 : 10);
    if (r.ec != std::errc() || r.ptr != e) return false;
    out = v;
    return true;
}

std::string hexU64(uint64_t v)
{
    char buf[32];
    std::snprintf(buf, sizeof buf, "0x%016llx", static_cast<unsigned long long>(v));
    return buf;
}

uint64_t toolkitTrigHash()
{
    Hasher h(fnv1a64("saiga.content.toolkit.trig.v1"));
    for (int32_t v : trigQuarterTable()) h.addI32(v);
    return h.value();
}

uint64_t toolkitNoiseHash()
{
    Hasher h(fnv1a64("saiga.content.toolkit.noise.v1"));
    for (int64_t j = 0; j < 256; ++j)
        for (int64_t i = 0; i < 256; ++i)
        {
            const int64_t x = (i - 128) * 997, z = (j - 128) * 1009;
            h.addI32(fbm(1, x, z, 12, 5));
            h.addI32(ridged(2, x, z, 11, 4, 2));
            h.addI32(gradientNoise(3, x, z, 9));
            h.addI32(valueNoise(4, x, z, 7));
        }
    return h.value();
}

uint64_t toolkitSmoothHash()
{
    Hasher h(fnv1a64("saiga.content.toolkit.smooth.v1"));
    for (int64_t t = 0; t <= 65536; ++t) h.addI32(int32_t(smoothQ16(t)));
    return h.value();
}

std::string ContentHashReport::summary() const
{
    std::string s = std::string(ok ? "content hashes reproduce" : "content hashes differ") + " (" +
                    std::to_string(entries) + " entries, " + std::to_string(chunks) + " chunks, " +
                    std::to_string(seconds) + " s)";
    for (const std::string& p : problems) s += "\n  " + p;
    return s;
}

namespace
{
// Indented JSON: objects one key per line, arrays of scalars on one line
void pretty(const Json& j, int indent, std::string& out)
{
    const std::string pad(size_t(indent) * 2, ' '), inner(size_t(indent + 1) * 2, ' ');
    if (j.is_object())
    {
        if (j.object_items().empty())
        {
            out += "{}";
            return;
        }
        out += "{\n";
        size_t k = 0;
        for (const auto& kv : j.object_items())
        {
            out += inner + Json(kv.first).dump() + ": ";
            pretty(kv.second, indent + 1, out);
            out += ++k < j.object_items().size() ? ",\n" : "\n";
        }
        out += pad + "}";
        return;
    }
    if (j.is_array())
    {
        bool scalars = true;
        for (const Json& v : j.array_items()) scalars = scalars && !v.is_object() && !v.is_array();
        if (scalars)
        {
            out += "[";
            for (size_t k = 0; k < j.array_items().size(); ++k) out += (k ? ", " : "") + j.array_items()[k].dump();
            out += "]";
            return;
        }
        out += "[\n";
        for (size_t k = 0; k < j.array_items().size(); ++k)
        {
            out += inner;
            pretty(j.array_items()[k], indent + 1, out);
            out += k + 1 < j.array_items().size() ? ",\n" : "\n";
        }
        out += pad + "]";
        return;
    }
    out += j.dump();
}

struct Checker
{
    ContentHashReport& report;
    bool update;

    void problem(const std::string& p) { report.problems.push_back(p); }
    // Compares a hex value of the file with the actual one; returns the value to write
    Json compare(const Json& expected, uint64_t actual, const std::string& what)
    {
        uint64_t e = 0;
        if (!expected.is_string() || !parseU64(expected.string_value(), true, e))
            problem(what + ": expected a \"0x...\" string, got " + expected.dump());
        else if (e != actual)
            problem(what + ": expected " + expected.string_value() + ", actual " + hexU64(actual));
        return Json(hexU64(actual));
    }
};

bool parseChunkKey(const std::string& key, ChunkId& out)
{
    const size_t a = key.find(':'), b = key.find(':', a == std::string::npos ? a : a + 1);
    if (a == std::string::npos || b == std::string::npos) return false;
    int64_t s = 0, x = 0, z = 0;
    auto num = [](const std::string& t, int64_t& v)
    {
        auto r = std::from_chars(t.data(), t.data() + t.size(), v);
        return !t.empty() && r.ec == std::errc() && r.ptr == t.data() + t.size();
    };
    if (!num(key.substr(0, a), s) || !num(key.substr(a + 1, b - a - 1), x) || !num(key.substr(b + 1), z)) return false;
    if (s < 0 || s > 65535 || x < INT32_MIN || x > INT32_MAX || z < INT32_MIN || z > INT32_MAX) return false;
    out.space = SpaceId(s);
    out.x     = int32_t(x);
    out.z     = int32_t(z);
    return true;
}

// One setup entry; returns the updated entry
Json checkEntry(const WorldSetup& base, const Json& entry, int index, Checker& c)
{
    Json::object out    = entry.object_items();
    const std::string w = "setups[" + std::to_string(index) + "] (" + entry["name"].string_value() + ")";
    uint64_t seed       = 0;
    if (!entry["seed"].is_string() || !parseU64(entry["seed"].string_value(), false, seed))
    {
        c.problem(w + ".seed: expected a decimal string");
        return entry;
    }
    WorldSetup setup    = base;
    setup.seed          = seed;
    setup.sharedContent = nullptr;
    GenerationPins pins;
    std::map<std::string, std::vector<std::string>> slotLists;
    for (const auto& [name, g] : entry["generators"].object_items())
    {
        const int64_t plan  = int64_t(g["plan_version"].number_value()),
                      place = int64_t(g["placement_version"].number_value());
        if (plan < 1 || plan > 65535 || place < plan || place > 65535)
        {
            c.problem(w + ".generators." + name + ": invalid plan / placement versions");
            return entry;
        }
        pins.pinned[name]  = uint16_t(plan);
        pins.current[name] = uint16_t(place);
        slotLists[name];
        for (const Json& sl : g["slots"].array_items()) slotLists[name].push_back(sl.string_value());
    }
    ContentCache cache;
    const GenerationPlans plans = GenerationPipeline::build(setup, pins, cache);
    if (!plans.ok())
    {
        c.problem(w + ": phase A: " + plans.error);
        return entry;
    }
    // The generators of the file are the registered ones with their slot tables
    for (const auto& [name, slots] : slotLists)
    {
        const GeneratorList::Entry* g = plans.generators->findByName(name);
        if (!g)
        {
            c.problem(w + ": generator '" + name + "' is not registered");
            continue;
        }
        std::vector<std::string> actual;
        for (const GeneratorSlot& s : g->slots) actual.push_back(s.name);
        if (actual != slots) c.problem(w + ": generator '" + name + "': the slot table differs");
    }
    for (const auto& [id, g] : plans.generators->entries())
        if (id != 0 && !slotLists.count(g.generator->name()))
            c.problem(w + ": generator '" + std::string(g.generator->name()) + "' is not in the file");

    // Plans
    Json::object planOut;
    const Json& filePlans = entry["plans"];
    if (filePlans.object_items().size() != plans.spaces.size())
        c.problem(w + ".plans: the file lists " + std::to_string(filePlans.object_items().size()) +
                  " spaces, the setup has " + std::to_string(plans.spaces.size()));
    for (size_t i = 0; i < plans.spaces.size(); ++i)
    {
        const SpaceGeneration& sp = *plans.spaces[i];
        const std::string key     = std::to_string(i);
        const Json& e             = filePlans[key];
        if (e.is_object() && e["generator"].string_value() != sp.generatorName)
            c.problem(w + ".plans." + key + ": generator '" + e["generator"].string_value() + "', actual '" +
                      sp.generatorName + "'");
        planOut[key] = Json::object{
            {"generator", sp.generatorName},
            {"hash", c.compare(e["hash"], sp.planHash, w + ".plans." + key + " (" + sp.settings.name + ")")}};
    }
    out["plans"]              = planOut;
    out["combined_plan_hash"] = c.compare(entry["combined_plan_hash"], plans.combinedHash, w + ".combined_plan_hash");

    // The chunks to generate: the listed ones and the digest's
    std::vector<ChunkId> listed;
    for (const auto& kv : entry["chunks"].object_items())
    {
        ChunkId id;
        if (!parseChunkKey(kv.first, id) || id.space >= plans.spaces.size() || id.x < 0 || id.z < 0 ||
            id.x >= plans.spaces[id.space]->settings.sizeChunks || id.z >= plans.spaces[id.space]->settings.sizeChunks)
        {
            c.problem(w + ".chunks." + kv.first + ": not a chunk of the setup (\"space:x:z\" inside the space)");
            continue;
        }
        listed.push_back(id);
    }
    const Json& block = entry["digest_block"];
    int64_t b[4]      = {0, 0, 0, 0};
    if (!block.is_array() || block.array_items().size() != 4)
        c.problem(w + ".digest_block: expected [x0, z0, x1, z1]");
    else
        for (int k = 0; k < 4; ++k) b[k] = int64_t(block[size_t(k)].number_value());
    std::set<ChunkId> digestSet;
    const int64_t size0 = plans.spaces[0]->settings.sizeChunks;
    for (int64_t x = std::max<int64_t>(0, b[0]); x < std::min(b[2], size0); ++x)
        for (int64_t z = std::max<int64_t>(0, b[1]); z < std::min(b[3], size0); ++z)
        {
            ChunkId id;
            id.x = int32_t(x);
            id.z = int32_t(z);
            digestSet.insert(id);
        }
    std::vector<ChunkId> digestChunks(digestSet.begin(), digestSet.end());
    // Then every site chunk outside the block, once, in ChunkId order (sitesByChunk is ordered)
    for (const auto& kv : plans.spaces[0]->plan->sitesByChunk)
        if (!digestSet.count(kv.first)) digestChunks.push_back(kv.first);

    std::vector<ChunkId> all = listed;
    all.insert(all.end(), digestChunks.begin(), digestChunks.end());
    std::vector<std::shared_ptr<const ChunkPlacement>> results(all.size());
    JobPool pool(3);
    pool.parallelFor(0, int64_t(all.size()), 1,
                     [&](int64_t lo, int64_t hi)
                     {
                         for (int64_t k = lo; k < hi; ++k)
                         {
                             const SpaceGeneration& sp = *plans.spaces[all[size_t(k)].space];
                             results[size_t(k)]        = generatePlacement(cache, sp, all[size_t(k)], sp.current);
                         }
                     });
    c.report.chunks += int(all.size());

    Json::object chunkOut;
    for (size_t k = 0; k < listed.size(); ++k)
    {
        const ChunkId& id      = listed[k];
        const std::string key  = std::to_string(id.space) + ":" + std::to_string(id.x) + ":" + std::to_string(id.z);
        const Json& e          = entry["chunks"][key];
        const ContentHashes& h = results[k]->hashes;
        const std::string cw   = w + ".chunks." + key;
        chunkOut[key]          = Json::object{{"total", c.compare(e["total"], h.total, cw + " total")},
                                              {"heights", c.compare(e["heights"], h.heights, cw + " heights")},
                                              {"tier1", c.compare(e["tier1"], h.tier1, cw + " tier1")},
                                              {"grass", c.compare(e["grass"], h.grass, cw + " grass")},
                                              {"stamps", c.compare(e["stamps"], h.stamps, cw + " stamps")},
                                              {"statics", c.compare(e["statics"], h.statics, cw + " statics")}};
    }
    out["chunks"] = chunkOut;
    Hasher digest(fnv1a64("saiga.content.digest.v1"));
    digest.addU32(uint32_t(digestChunks.size()));
    for (size_t k = 0; k < digestChunks.size(); ++k)
        digest.addI32(digestChunks[k].x).addI32(digestChunks[k].z).addU64(results[listed.size() + k]->hashes.total);
    out["digest"] =
        c.compare(entry["digest"], digest.value(), w + ".digest (" + std::to_string(digestChunks.size()) + " chunks)");
    return out;
}
}  // namespace

ContentHashReport checkContentHashes(const WorldSetup& setup, const std::string& file, bool update)
{
    const auto start = std::chrono::steady_clock::now();
    ContentHashReport report;
    Checker c{report, update};
    auto finish = [&]()
    {
        report.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        return report;
    };
    std::error_code ec;
    if (!std::filesystem::exists(file, ec))
    {
        c.problem(file + ": the content hash file does not exist (commit a skeleton whose values are \"0x0\" first)");
        return finish();
    }
    Json root;
    std::string error;
    if (!readJsonFile(file, root, error))
    {
        c.problem(error);
        return finish();
    }
    if (root["format"].number_value() != 1 || !root["setups"].is_array())
    {
        c.problem(file + ": expected {\"format\": 1, \"setups\": [...]}");
        return finish();
    }
    Json::object out = root.object_items();
    if (root["toolkit"].is_object())
    {
        const Json& t = root["toolkit"];
        out["toolkit"] =
            Json::object{{"trig_table", c.compare(t["trig_table"], toolkitTrigHash(), "toolkit.trig_table")},
                         {"noise_grid", c.compare(t["noise_grid"], toolkitNoiseHash(), "toolkit.noise_grid")},
                         {"smooth_q16", c.compare(t["smooth_q16"], toolkitSmoothHash(), "toolkit.smooth_q16")}};
    }
    Json::array setups;
    for (size_t i = 0; i < root["setups"].array_items().size(); ++i)
    {
        setups.push_back(checkEntry(setup, root["setups"][i], int(i), c));
        report.entries++;
    }
    out["setups"] = setups;
    if (update)
    {
        std::string text;
        pretty(Json(out), 0, text);
        std::ofstream f(file, std::ios::binary | std::ios::trunc);
        f << text << "\n";
        if (!f)
            c.problem(file + ": cannot be written");
        else
            report.updated = true;
        c.problem(file + ": updated (an update run fails; commit the file in a dedicated commit with the reason)");
        report.ok = false;
        return finish();
    }
    report.ok = report.problems.empty();
    return finish();
}

}  // namespace Engine
}  // namespace Saiga
