/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/save/SaveDirectory.h"

#include "saiga/engine/content/ChunkRecords.h"
#include "saiga/engine/core/EngineVersion.h"
#include "saiga/engine/core/Log.h"
#include "saiga/engine/streaming/ChunkResidency.h"
#include "saiga/engine/terrain/TerrainEditStore.h"
#include "saiga/engine/terrain/TerrainSystem.h"
#include "saiga/engine/world/World.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <tuple>

namespace Saiga
{
namespace Engine
{
namespace
{
double nowSeconds()
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

bool allDigits(const std::string& s)
{
    if (s.empty() || s.size() > 19) return false;
    for (char c : s)
        if (c < '0' || c > '9') return false;
    return true;
}

bool endsWith(const std::string& s, const std::string& suffix)
{
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// world.<c>.sgw (kind 0), world.<c>.sgw.tmp (1), world.<c>.sgw.corrupt (2); false for other names
bool parseWorldName(const std::string& name, uint64_t& commit, int& kind)
{
    if (name.compare(0, 6, "world.") != 0) return false;
    std::string rest = name.substr(6);
    kind             = 0;
    if (endsWith(rest, ".sgw.tmp"))
    {
        kind = 1;
        rest = rest.substr(0, rest.size() - 8);
    }
    else if (endsWith(rest, ".sgw.corrupt"))
    {
        kind = 2;
        rest = rest.substr(0, rest.size() - 12);
    }
    else if (endsWith(rest, ".sgw"))
        rest = rest.substr(0, rest.size() - 4);
    else
        return false;
    if (!allDigits(rest)) return false;
    commit = std::strtoull(rest.c_str(), nullptr, 10);
    return commit > 0;
}

// <rx>.<rz>.<g>.sgr
bool parseRegionName(SpaceId space, const std::string& name, RegionKey& out)
{
    if (!endsWith(name, ".sgr")) return false;
    const std::string body = name.substr(0, name.size() - 4);
    const size_t a = body.find('.'), b = a == std::string::npos ? a : body.find('.', a + 1);
    if (b == std::string::npos) return false;
    const std::string x = body.substr(0, a), z = body.substr(a + 1, b - a - 1), g = body.substr(b + 1);
    auto isInt = [](const std::string& s) { return !s.empty() && allDigits(s[0] == '-' ? s.substr(1) : s); };
    if (!isInt(x) || !isInt(z) || !allDigits(g)) return false;
    out.space      = space;
    out.rx         = int32_t(std::strtol(x.c_str(), nullptr, 10));
    out.rz         = int32_t(std::strtol(z.c_str(), nullptr, 10));
    out.generation = uint32_t(std::strtoul(g.c_str(), nullptr, 10));
    return out.path() == "regions/" + std::to_string(space) + "/" + name;
}

using RegionTriple = std::tuple<SpaceId, int32_t, int32_t>;
RegionTriple triple(const RegionKey& k)
{
    return {k.space, k.rx, k.rz};
}

std::string chunkName(const ChunkId& c)
{
    if (isSpaceRecordKey(c)) return "the space record of space " + std::to_string(c.space);
    return "chunk (" + std::to_string(c.space) + ", " + std::to_string(c.x) + ", " + std::to_string(c.z) + ")";
}

// Reads one record: false with corrupt = true for a checksum, length or header mismatch
bool readRecord(SaveFileSystem& fs, const RecordLocation& loc, const ChunkId& key, RecordContent& out,
                std::string& error, bool* corrupt = nullptr)
{
    if (corrupt) *corrupt = true;
    std::vector<uint8_t> bytes(loc.length);
    if (!fs.readAt(loc.file.path(), loc.offset, loc.length, bytes.data(), &error)) return false;
    RecordHeader h;
    if (!decodeRecordHeader(bytes.data(), bytes.size(), h, error)) return false;
    if (h.key != key || h.commit != loc.commit || kRecordHeaderSize + size_t(h.payloadLength) + 8 != bytes.size() ||
        h.kind != (isSpaceRecordKey(key) ? kRecordKindSpace : kRecordKindChunk))
    {
        error = chunkName(key) + ": the record does not match its index entry";
        return false;
    }
    uint64_t stored = 0;
    for (int i = 0; i < 8; ++i) stored |= uint64_t(bytes[bytes.size() - 8 + size_t(i)]) << (8 * i);
    if (fnv1a64Bytes(bytes.data(), bytes.size() - 8) != stored)
    {
        error = chunkName(key) + ": the record checksum failed";
        return false;
    }
    // The checksum passed: a decode error is a format (version) error, not corruption
    if (corrupt) *corrupt = false;
    return decodeRecord(bytes.data(), bytes.size(), out, error);
}
}  // namespace

struct SaveDirectory::FrozenChunk
{
    ChunkId key;
    bool space         = false;
    uint32_t sequence  = 0;
    bool hasDeltas     = false;
    uint64_t deltaHash = 0;
    TerrainEditStore::FrozenTerrain terrain;
    bool hasRecord            = false;
    uint16_t placementVersion = 0;
    bool stamped              = false;
    std::vector<InstanceKey> tombstones;
    std::vector<FrozenGroup> groups;
    std::optional<RecordLocation> oldRecord;
    bool spaceUnloaded = false;
};

struct SaveDirectory::Snapshot
{
    uint64_t commit = 0, storeGeneration = 0;
    int zstdLevel = 3;
    std::vector<uint8_t> blob;
    bool writeHeader = false;
    std::string engineVersion;
    std::vector<std::shared_ptr<const SchemaTable>> tables;
    std::map<ChunkId, IndexEntry> index;
    std::map<RegionKey, uint64_t> files, backupFiles;
    std::vector<FrozenChunk> frozen;
};

struct SaveDirectory::Result
{
    bool ok = false, pointReached = false;
    std::string error;
    std::vector<std::string> log;
    std::map<ChunkId, IndexEntry> index;
    std::map<ChunkId, std::vector<uint64_t>> recordGroups;
    std::map<RegionKey, uint64_t> files;
    uint64_t failedGroups = 0, recordsWritten = 0, bytesWritten = 0, compactions = 0;
    // Disk-only groups whose old record could not be read: omitted from the new record, gone with this commit (G14)
    std::set<uint64_t> omitted;
    double ms = 0;
};

SaveDirectory::SaveDirectory(World& world, const std::string& path, const SaveDirectoryOptions& options,
                             std::shared_ptr<SaveFileSystem> fileSystem)
    : w(world), path(path), options(options), fs(std::move(fileSystem)), lane(world)
{
}

SaveDirectory::~SaveDirectory()
{
    if (locked) fs->unlock();
}

bool SaveDirectory::synchronousIo() const
{
    return options.synchronousIo ? *options.synchronousIo : w.streamingSettings().synchronous;
}

SaveStatus SaveDirectory::status() const
{
    SaveStatus s;
    s.open           = true;
    s.path           = path;
    s.committed      = committedCommit;
    s.inFlight       = inFlightCommit;
    s.lastCommitStep = lastCommitStep;
    s.lastCommitTime = lastCommitTime;
    s.lastError      = lastError;
    s.regionFiles    = files.size();
    s.regionBytes    = regionBytes;
    for (const auto& [c, e] : w.chunkStoreView().chunks())
    {
        s.dirtyChunks += e.dirty ? 1 : 0;
        if (e.record) s.liveBytes += e.record->length;
    }
    return s;
}

// ---- reads

void SaveDirectory::readStoredSync(const ChunkId& chunk)
{
    ChunkStore& store   = w.chunkStore();
    const ChunkEntry* e = store.entry(chunk);
    if (!e || !e->record || !store.needsRead(chunk)) return;
    RecordContent content;
    std::string error;
    store.statsForUpdate().recordsRead++;
    store.statsForUpdate().syncReads++;
    if (readRecord(*fs, *e->record, chunk, content, error))
    {
        store.fillBytes(chunk, content.groups);
        return;
    }
    size_t n = 0;
    for (const StoredGroup& g : e->groups) n += g.bytes ? 0 : 1;
    store.statsForUpdate().failedGroups += n;
    SAIGA_LOG("save", "%s: its record cannot be read (%s): %zu stored groups are not reloaded",
              chunkName(chunk).c_str(), error.c_str(), n);
}

void SaveDirectory::requestStoredRead(const ChunkId& chunk, int priority, std::function<void()> done)
{
    const ChunkEntry* e = w.chunkStoreView().entry(chunk);
    if (!e || !e->record)
    {
        done();
        return;
    }
    struct ReadResult
    {
        bool ok = false;
        RecordContent content;
        std::string error;
    };
    auto result                               = std::make_shared<ReadResult>();
    const RecordLocation loc                  = *e->record;
    const std::shared_ptr<SaveFileSystem> fsp = fs;
    const uint64_t generation                 = w.chunkStoreView().generation();
    lane.submit(
        priority, true,
        [fsp, loc, chunk, result]() { result->ok = readRecord(*fsp, loc, chunk, result->content, result->error); },
        [this, chunk, result, generation, done]()
        {
            ChunkStore& store = w.chunkStore();
            if (store.generation() == generation)
            {
                store.statsForUpdate().recordsRead++;
                if (result->ok)
                    store.fillBytes(chunk, result->content.groups);
                else if (const ChunkEntry* en = store.entry(chunk))
                {
                    size_t n = 0;
                    for (const StoredGroup& g : en->groups) n += g.bytes ? 0 : 1;
                    store.statsForUpdate().failedGroups += n;
                    SAIGA_LOG("save", "%s: its record cannot be read (%s): %zu stored groups are not reloaded",
                              chunkName(chunk).c_str(), result->error.c_str(), n);
                }
            }
            done();
        },
        synchronousIo());
}

bool SaveDirectory::readGroupForSave(const ChunkId& chunk, const StoredGroup& group, GroupBytes& out) const
{
    const ChunkEntry* e = w.chunkStoreView().entry(chunk);
    if (!e || !e->record || group.recordIndex < 0) return false;
    RecordContent content;
    std::string error;
    const_cast<World&>(w).chunkStore().statsForUpdate().syncReads++;
    if (!readRecord(*fs, *e->record, chunk, content, error) || size_t(group.recordIndex) >= content.groups.size())
        return false;
    out = content.groups[size_t(group.recordIndex)].bytes;
    return out && fnv1a64Bytes(out->data(), out->size()) == group.hash;
}

// ---- commits

std::shared_ptr<SaveDirectory::Snapshot> SaveDirectory::freeze(uint64_t c, bool)
{
    ChunkStore& store               = w.chunkStore();
    auto snap                       = std::make_shared<Snapshot>();
    snap->commit                    = c;
    snap->storeGeneration           = store.generation();
    snap->zstdLevel                 = options.zstdLevel;
    snap->blob                      = w.saveWorldState();
    snap->tables                    = store.tables();
    snap->writeHeader               = !headerWritten || snap->tables.size() > headerTables;
    snap->engineVersion             = engineVersion();
    snap->files                     = files;
    snap->backupFiles               = backupFiles;
    const TerrainEditStore& terrain = w.terrainEdits();
    const ChunkRecords& records     = w.contentRecords();
    for (auto& [key, e] : store.entriesForUpdate())
    {
        if (e.indexed)
        {
            IndexEntry ie;
            ie.kind     = isSpaceRecordKey(key) ? kRecordKindSpace : kRecordKindChunk;
            ie.key      = key;
            ie.location = e.record;
            ie.summary  = e.summary;
            snap->index.emplace(key, std::move(ie));
        }
        if (!e.dirty) continue;
        FrozenChunk f;
        f.key   = key;
        f.space = isSpaceRecordKey(key);
        if (f.space)
            f.spaceUnloaded = store.spaceUnloaded(key.space);
        else
        {
            if (const TerrainChunkEdits* te = terrain.entry(key))
            {
                f.sequence  = te->sequence;
                f.hasDeltas = te->nonZeroSamples > 0;
                if (f.hasDeltas)
                {
                    f.deltaHash = terrain.contentHash(key);
                    terrain.freeze(key, f.terrain);
                }
            }
            if (const ChunkRecord* r = records.find(key))
            {
                f.hasRecord        = true;
                f.placementVersion = r->placementVersion;
                f.tombstones       = r->tombstones;
            }
            f.stamped = records.isStamped(key);
        }
        for (const StoredGroup& g : e.groups)
        {
            FrozenGroup fg;
            fg.id       = g.id;
            fg.schema   = g.schema;
            fg.hash     = g.hash;
            fg.bytes    = g.bytes;
            fg.oldIndex = g.recordIndex;
            f.groups.push_back(std::move(fg));
        }
        f.oldRecord = e.record;
        e.dirty     = false;
        snap->frozen.push_back(std::move(f));
    }
    return snap;
}

void SaveDirectory::writeCommit(const Snapshot& snap, SaveFileSystem& fs, Result& result)
{
    const double start = nowSeconds();
    std::string e;
    auto fail = [&](const std::string& what)
    {
        result.ok    = false;
        result.error = what + (e.empty() ? "" : ": " + e);
    };
    result.index = snap.index;
    // The current generation of each region with a file in the table, the live bytes per file
    std::map<RegionTriple, RegionKey> current;
    for (const auto& [k, end] : snap.files) current[triple(k)] = k;
    std::map<RegionKey, uint64_t> live;
    for (const auto& [key, ie] : snap.index)
        if (ie.location) live[ie.location->file] += ie.location->length;
    std::set<ChunkId> rewritten;
    for (const FrozenChunk& f : snap.frozen) rewritten.insert(f.key);
    // The highest generation of each region in the directory listing (a new file goes above it: never over a file the
    // backup commit names)
    std::map<SpaceId, bool> listed;
    std::map<RegionTriple, uint32_t> highest;
    auto listSpace = [&](SpaceId s)
    {
        if (listed[s]) return;
        listed[s] = true;
        std::vector<std::string> names;
        const std::string dir = "regions/" + std::to_string(s);
        if (!fs.exists(dir) || !fs.list(dir, names, nullptr)) return;
        for (const std::string& n : names)
        {
            RegionKey k;
            if (!parseRegionName(s, n, k)) continue;
            uint32_t& h = highest[triple(k)];
            h           = std::max(h, k.generation);
        }
    };
    // Targets: the file each region appends to this commit (and its end); created files get their header
    struct Target
    {
        RegionKey key;
        uint64_t end    = 0;
        bool created    = false;
        bool compacting = false;
    };
    std::map<RegionTriple, Target> targets;
    std::set<std::string> newDirectories;
    bool regionsCreated    = false;
    auto ensureDirectories = [&](SpaceId s)
    {
        if (!fs.exists("regions"))
        {
            if (!fs.makeDirectory("regions", &e)) return false;
            regionsCreated = true;
        }
        const std::string dir = "regions/" + std::to_string(s);
        if (!fs.exists(dir))
        {
            if (!fs.makeDirectory(dir, &e)) return false;
            newDirectories.insert("regions");
        }
        return true;
    };
    auto newFile = [&](const RegionTriple& t, Target& out)
    {
        listSpace(std::get<0>(t));
        uint32_t g = 0;
        if (highest.count(t)) g = highest[t] + 1;
        auto it = current.find(t);
        if (it != current.end()) g = std::max(g, it->second.generation + 1);
        highest[t]  = g;
        out.key     = RegionKey{std::get<0>(t), std::get<1>(t), std::get<2>(t), g};
        out.created = true;
        if (!ensureDirectories(std::get<0>(t))) return false;
        const std::vector<uint8_t> header = encodeRegionHeader(out.key);
        if (!fs.writeNew(out.key.path(), header.data(), header.size(), &e)) return false;
        out.end = header.size();
        newDirectories.insert("regions/" + std::to_string(std::get<0>(t)));
        return true;
    };
    // 0 + 2: the compaction decisions and the truncation of every touched file to its committed end
    std::set<RegionTriple> touched;
    for (const FrozenChunk& f : snap.frozen) touched.insert(triple(regionKeyOf(f.key, 0)));
    for (const auto& [t, key] : current)
    {
        const uint64_t size = snap.files.at(key);
        const uint64_t l    = live.count(key) ? live[key] : 0;
        if (size >= kCompactMinBytes && 2 * l < size)
        {
            Target tg;
            tg.compacting = true;
            targets[t]    = tg;
            touched.insert(t);
        }
    }
    for (const RegionTriple& t : touched)
    {
        auto it = current.find(t);
        if (it == current.end() || (targets.count(t) && targets[t].compacting)) continue;
        Target tg;
        tg.key = it->second;
        tg.end = snap.files.at(it->second);
        // A file that vanished is recreated (never appended to without its header)
        if (!fs.exists(tg.key.path()))
        {
            if (!newFile(t, tg)) return fail("create " + tg.key.path());
        }
        else if (!fs.truncate(tg.key.path(), tg.end, &e))
            return fail("truncate " + tg.key.path());
        targets[t] = tg;
    }
    // 1: the header (before any record that uses a new table)
    if (snap.writeHeader)
    {
        std::vector<uint8_t> header;
        encodeDirHeader(snap.engineVersion, snap.tables, header);
        const std::string tmp = std::string(kDirHeaderName) + ".tmp";
        if (!fs.writeNew(tmp, header.data(), header.size(), &e) || !fs.syncFile(tmp, &e) ||
            !fs.rename(tmp, kDirHeaderName, &e) || !fs.syncDirectory("", &e))
            return fail("write the header");
    }
    // 2: compaction: a new generation with the live records this commit does not rewrite, copied verbatim
    for (auto& [t, tg] : targets)
    {
        if (!tg.compacting) continue;
        const RegionKey old = current.at(t);
        if (!newFile(t, tg)) return fail("create " + tg.key.path());
        for (auto& [key, ie] : result.index)
        {
            if (!ie.location || !(ie.location->file == old) || rewritten.count(key)) continue;
            std::vector<uint8_t> bytes(ie.location->length);
            if (!fs.readAt(old.path(), ie.location->offset, ie.location->length, bytes.data(), &e))
                return fail("compaction: read " + old.path());
            if (!fs.append(tg.key.path(), bytes.data(), bytes.size(), &e)) return fail("append " + tg.key.path());
            ie.location->file   = tg.key;
            ie.location->offset = tg.end;
            tg.end += bytes.size();
            result.bytesWritten += bytes.size();
        }
        result.compactions++;
        result.log.push_back("compacted " + old.path() + " into " + tg.key.path());
    }
    // 3: the records of the frozen chunks
    std::map<std::pair<RegionKey, uint64_t>, std::shared_ptr<RecordContent>> oldRecords;
    std::vector<int32_t> dense;
    for (const FrozenChunk& f : snap.frozen)
    {
        RecordContent rc;
        rc.kind   = f.space ? kRecordKindSpace : kRecordKindChunk;
        rc.key    = f.key;
        rc.commit = snap.commit;
        IndexSummary summary;
        std::vector<uint64_t> ids;
        // The groups: in memory, or read from the old record by their index (an unreadable old record: omitted)
        std::shared_ptr<RecordContent> old;
        bool oldFailed = false;
        std::vector<RecordGroup> groups;
        for (const FrozenGroup& g : f.groups)
        {
            GroupBytes bytes = g.bytes;
            if (!bytes)
            {
                if (!old && !oldFailed && f.oldRecord)
                {
                    const auto okey = std::make_pair(f.oldRecord->file, f.oldRecord->offset);
                    auto it         = oldRecords.find(okey);
                    if (it != oldRecords.end())
                        old = it->second;
                    else
                    {
                        auto content = std::make_shared<RecordContent>();
                        std::string re;
                        if (readRecord(fs, *f.oldRecord, f.key, *content, re))
                        {
                            old              = content;
                            oldRecords[okey] = content;
                        }
                        else
                        {
                            oldFailed = true;
                            result.log.push_back(chunkName(f.key) + ": its old record cannot be read (" + re +
                                                 "): its stored groups are omitted");
                        }
                    }
                }
                if (old && g.oldIndex >= 0 && size_t(g.oldIndex) < old->groups.size())
                    bytes = old->groups[size_t(g.oldIndex)].bytes;
                if (!bytes || fnv1a64Bytes(bytes->data(), bytes->size()) != g.hash)
                {
                    result.failedGroups++;
                    result.omitted.insert(g.id);
                    continue;
                }
            }
            RecordGroup rg;
            rg.schema = g.schema;
            rg.bytes  = bytes;
            groups.push_back(rg);
            summary.groups.push_back({g.schema, g.hash});
            ids.push_back(g.id);
        }
        bool empty = false, summaryOnly = false;
        if (f.space)
        {
            empty = !f.spaceUnloaded;
            if (!groups.empty())
            {
                rc.hasSpaceGroup = true;
                rc.spaceGroup    = groups[0];
            }
        }
        else
        {
            rc.hasTerrain            = f.sequence > 0;
            rc.sequence              = f.sequence;
            rc.hasDeltas             = f.hasDeltas;
            summary.sequence         = f.sequence;
            summary.deltas           = f.hasDeltas;
            summary.deltaHash        = f.deltaHash;
            rc.hasRecord             = f.hasRecord;
            rc.placementVersion      = f.placementVersion;
            rc.stamped               = f.stamped;
            rc.tombstones            = f.tombstones;
            summary.record           = f.hasRecord;
            summary.stamped          = f.stamped;
            summary.placementVersion = f.placementVersion;
            summary.tombstones       = uint32_t(f.tombstones.size());
            rc.groups                = groups;
            empty                    = summary.empty();
            summaryOnly              = !summary.needsRecord() && groups.empty();
            if (f.hasDeltas)
            {
                f.terrain.dense(dense);
                rc.N      = uint16_t(f.terrain.N);
                rc.deltas = dense;
            }
        }
        if (empty)
        {
            result.index.erase(f.key);
            continue;
        }
        IndexEntry ie;
        ie.kind                    = rc.kind;
        ie.key                     = f.key;
        ie.summary                 = summary;
        result.recordGroups[f.key] = ids;
        if (summaryOnly)
        {
            result.index[f.key] = ie;
            continue;
        }
        std::vector<uint8_t> bytes;
        encodeRecord(rc, snap.zstdLevel, bytes);
        const RegionTriple t = triple(regionKeyOf(f.key, 0));
        Target& tg           = targets[t];
        if (tg.end == 0 && !newFile(t, tg)) return fail("create a region file of " + chunkName(f.key));
        if (!fs.append(tg.key.path(), bytes.data(), bytes.size(), &e)) return fail("append " + tg.key.path());
        RecordLocation loc;
        loc.file            = tg.key;
        loc.offset          = tg.end;
        loc.length          = uint32_t(bytes.size());
        loc.commit          = snap.commit;
        ie.location         = loc;
        result.index[f.key] = ie;
        tg.end += bytes.size();
        result.recordsWritten++;
        result.bytesWritten += bytes.size();
    }
    // 4: every file appended to or created is synced, then every directory with a new entry
    for (const auto& [t, tg] : targets)
        if (tg.end > 0 && !fs.syncFile(tg.key.path(), &e)) return fail("sync " + tg.key.path());
    if (regionsCreated && !fs.syncDirectory("", &e)) return fail("sync the directory");
    for (const std::string& d : newDirectories)
        if (!fs.syncDirectory(d, &e)) return fail("sync " + d);
    // 5: the world-state file with the index: the commit point
    ChunkIndex index;
    std::map<RegionKey, uint64_t> used;
    for (const auto& [key, ie] : result.index)
    {
        if (ie.location) used[ie.location->file] = 0;
        index.entries.push_back(ie);
    }
    std::sort(index.entries.begin(), index.entries.end(), [](const IndexEntry& a, const IndexEntry& b)
              { return std::make_tuple(a.kind, a.key) < std::make_tuple(b.kind, b.key); });
    for (auto& [k, end] : used)
    {
        end         = snap.files.count(k) ? snap.files.at(k) : 0;
        auto target = targets.find(triple(k));
        if (target != targets.end() && target->second.key == k) end = target->second.end;
        index.files.push_back({k, end});
    }
    result.files = used;
    WorldFile wf;
    wf.commit       = snap.commit;
    wf.tablesNeeded = uint32_t(snap.tables.size());
    wf.blob         = snap.blob;
    encodeIndex(index, wf.index);
    std::vector<uint8_t> bytes;
    encodeWorldFile(wf, snap.zstdLevel, bytes);
    const std::string name = worldFileName(snap.commit), tmp = name + ".tmp";
    if (!fs.writeNew(tmp, bytes.data(), bytes.size(), &e) || !fs.syncFile(tmp, &e)) return fail("write " + tmp);
    if (!fs.rename(tmp, name, &e)) return fail("rename " + tmp);
    result.bytesWritten += bytes.size();
    result.ok           = true;
    result.pointReached = true;
    // After the commit point: errors are reported, not rolled back
    if (!fs.syncDirectory("", &e)) result.error = "sync the directory after the commit point: " + e;
    // 6: the commit before the backup and the region files no kept commit names
    if (snap.commit >= 3)
    {
        const std::string old = worldFileName(snap.commit - 2);
        if (fs.exists(old) && !fs.remove(old, &e)) result.log.push_back("remove " + old + ": " + e);
    }
    for (const auto& [k, end] : snap.backupFiles)
    {
        if (snap.files.count(k) || result.files.count(k) || !fs.exists(k.path())) continue;
        if (!fs.remove(k.path(), &e)) result.log.push_back("remove " + k.path() + ": " + e);
    }
    result.ms = (nowSeconds() - start) * 1000.0;
}

uint64_t SaveDirectory::commit()
{
    if (inFlightCommit != 0)
    {
        commitRequested = true;
        return 0;
    }
    const auto start     = std::chrono::steady_clock::now();
    const uint64_t c     = committedCommit + 1;
    auto snap            = freeze(c, committedCommit == 0);
    inFlightCommit       = c;
    commitRequested      = false;
    lastCommitStep       = w.stepIndex();
    lastCommitTime       = w.time();
    ChunkStore::Stats& s = w.chunkStore().statsForUpdate();
    s.lastFreezeMs       = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    s.maxFreezeMs        = std::max(s.maxFreezeMs, s.lastFreezeMs);
    auto result          = std::make_shared<Result>();
    const std::shared_ptr<SaveFileSystem> fsp = fs;
    lane.submit(
        -50000, false,
        [snap, fsp, result]()
        {
            try
            {
                writeCommit(*snap, *fsp, *result);
            }
            catch (const std::exception& x)
            {
                result->ok    = result->pointReached;
                result->error = std::string("the commit threw: ") + x.what();
            }
        },
        [this, snap, result]() { complete(*snap, *result); }, synchronousIo());
    return c;
}

void SaveDirectory::complete(const Snapshot& snap, const Result& result)
{
    inFlightCommit    = 0;
    ChunkStore& store = w.chunkStore();
    for (const std::string& m : result.log)
        SAIGA_LOG("save", "commit %llu: %s", (unsigned long long)snap.commit, m.c_str());
    if (snap.storeGeneration != store.generation()) return;
    ChunkStore::Stats& stats = store.statsForUpdate();
    if (!result.ok)
    {
        // Nothing was dropped: the frozen chunks are written by the next commit (the same number)
        for (const FrozenChunk& f : snap.frozen) store.entriesForUpdate()[f.key].dirty = true;
        lastError = result.error;
        stats.failedCommits++;
        SAIGA_LOG("save", "commit %llu failed: %s", (unsigned long long)snap.commit, result.error.c_str());
        return;
    }
    auto& entries = store.entriesForUpdate();
    std::set<ChunkId> keys;
    for (const auto& kv : result.index) keys.insert(kv.first);
    for (const FrozenChunk& f : snap.frozen) keys.insert(f.key);
    for (const auto& kv : snap.index) keys.insert(kv.first);
    for (const ChunkId& key : keys)
    {
        ChunkEntry& e = entries[key];
        auto it       = result.index.find(key);
        if (it != result.index.end())
        {
            e.indexed = true;
            e.record  = it->second.location;
            e.summary = it->second.summary;
        }
        else
        {
            e.indexed = false;
            e.record.reset();
            e.summary = IndexSummary();
        }
        auto ids = result.recordGroups.find(key);
        if (ids != result.recordGroups.end())
            for (StoredGroup& g : e.groups)
            {
                auto pos      = std::find(ids->second.begin(), ids->second.end(), g.id);
                g.recordIndex = pos == ids->second.end() ? -1 : int32_t(pos - ids->second.begin());
            }
        else if (!isSpaceRecordKey(key) &&
                 std::find_if(snap.frozen.begin(), snap.frozen.end(),
                              [&](const FrozenChunk& f) { return f.key == key; }) != snap.frozen.end())
            for (StoredGroup& g : e.groups) g.recordIndex = -1;
        // The groups the record could not carry (an unreadable old record, only on disk) are lost with this commit
        if (!result.omitted.empty())
            e.groups.erase(std::remove_if(e.groups.begin(), e.groups.end(),
                                          [&](const StoredGroup& g) { return !g.bytes && result.omitted.count(g.id); }),
                           e.groups.end());
    }
    // The one rule: the bytes of groups in the record leave memory while the chunk is not requested or resident
    for (const FrozenChunk& f : snap.frozen)
    {
        if (f.space) continue;
        if (!w.streaming() || w.residency().state(f.key) == ChunkState::None) store.dropBytes(f.key);
    }
    for (const ChunkId& key : keys) store.prune(key);
    backupFiles = snap.files;
    files       = result.files;
    regionBytes = 0;
    for (const auto& kv : files) regionBytes += kv.second;
    committedCommit = snap.commit;
    if (snap.writeHeader)
    {
        headerWritten = true;
        headerTables  = snap.tables.size();
    }
    lastError = result.error;
    if (!result.error.empty())
        SAIGA_LOG("save", "commit %llu: %s", (unsigned long long)snap.commit, result.error.c_str());
    stats.commits++;
    stats.recordsWritten += result.recordsWritten;
    stats.bytesWritten += result.bytesWritten;
    stats.compactions += result.compactions;
    stats.failedGroups += result.failedGroups;
    stats.lastCommitWriteMs = result.ms;
}

bool SaveDirectory::waitForCommits()
{
    lane.drainOnCaller();
    return lastError.empty();
}

void SaveDirectory::afterStep()
{
    // The lane's completions (a world without streaming drains its pool nowhere else)
    if (lane.busy() && w.terrain().poolCreated()) w.terrain().drainStreamingPool();
    lane.pump();
    if (inFlightCommit != 0) return;
    if (commitRequested ||
        (options.commitIntervalSeconds > 0 && w.time() - lastCommitTime >= options.commitIntervalSeconds))
        commit();
}

void SaveDirectory::beforeLoad()
{
    lane.drainOnCaller();
    lane.cancelReads();
}

void SaveDirectory::afterLoad()
{
    lastCommitTime = w.time();
    lastCommitStep = w.stepIndex();
}

bool SaveDirectory::close()
{
    lane.drainOnCaller();
    ChunkStore& store = w.chunkStore();
    for (auto& [c, e] : store.entriesForUpdate())
        if (store.needsRead(c)) readStoredSync(c);
    for (auto& [c, e] : store.entriesForUpdate())
    {
        e.indexed = false;
        e.record.reset();
        e.summary = IndexSummary();
        e.dirty   = false;
        for (StoredGroup& g : e.groups) g.recordIndex = -1;
    }
    store.setTracking(false);
    std::vector<ChunkId> keys;
    for (const auto& kv : store.chunks()) keys.push_back(kv.first);
    for (const ChunkId& c : keys) store.prune(c);
    if (locked) fs->unlock();
    locked = false;
    return lastError.empty();
}

// ---- open

SaveOpenResult SaveDirectory::open()
{
    const double start = nowSeconds();
    SaveOpenResult out;
    std::string e;
    auto fail = [&](const std::string& message)
    {
        out.ok    = false;
        out.error = "save directory " + fs->describe() + ": " + message;
        if (locked) fs->unlock();
        locked = false;
        return out;
    };
    if (!fs->lock(&e)) return fail(e);
    locked = true;
    // The names: only the known patterns
    std::vector<std::string> names;
    if (!fs->list("", names, &e)) return fail(e);
    std::vector<uint64_t> worldFiles;
    std::vector<std::string> tmpFiles, corruptFiles;
    bool anyWorld = false, hasHeader = false;
    for (const std::string& n : names)
    {
        uint64_t c = 0;
        int kind   = 0;
        if (n == "lock" || n == "regions") continue;
        if (n == kDirHeaderName)
        {
            hasHeader = true;
            continue;
        }
        if (parseWorldName(n, c, kind))
        {
            if (kind == 0) worldFiles.push_back(c);
            if (kind == 1) tmpFiles.push_back(n);
            if (kind == 2) corruptFiles.push_back(n);
            anyWorld = anyWorld || kind != 1;
            continue;
        }
        if (endsWith(n, ".tmp"))
        {
            tmpFiles.push_back(n);
            continue;
        }
        return fail("the unknown file '" + n + "' (a save directory holds only its own files)");
    }
    // regions/<space>/<rx>.<rz>.<g>.sgr and *.tmp
    std::vector<RegionKey> regionFiles;
    std::vector<std::string> regionTmp;
    if (fs->exists("regions"))
    {
        std::vector<std::string> spaces;
        if (!fs->list("regions", spaces, &e)) return fail(e);
        for (const std::string& s : spaces)
        {
            if (!allDigits(s) || !fs->isDirectory("regions/" + s)) return fail("the unknown entry 'regions/" + s + "'");
            std::vector<std::string> list;
            if (!fs->list("regions/" + s, list, &e)) return fail(e);
            for (const std::string& n : list)
            {
                RegionKey k;
                if (parseRegionName(SpaceId(std::strtoul(s.c_str(), nullptr, 10)), n, k))
                    regionFiles.push_back(k);
                else if (endsWith(n, ".tmp"))
                    regionTmp.push_back("regions/" + s + "/" + n);
                else
                    return fail("the unknown file 'regions/" + s + "/" + n + "'");
            }
        }
    }
    ChunkStore& store = w.chunkStore();
    if (!anyWorld)
    {
        // Adopt (a new directory, or the garbage of a crashed first commit): the current world becomes commit 1
        for (const std::string& t : tmpFiles) fs->remove(t, nullptr);
        for (const std::string& t : regionTmp) fs->remove(t, nullptr);
        for (const RegionKey& k : regionFiles) fs->remove(k.path(), nullptr);
        if (hasHeader) fs->remove(kDirHeaderName, nullptr);
        store.setTracking(true);
        const TerrainEditStore& terrain = w.terrainEdits();
        for (size_t i = 0; i < terrain.spaceCount(); ++i)
            for (const auto& kv : terrain.chunks(SpaceId(i))) store.markDirty(kv.first);
        for (const auto& kv : w.contentRecords().records()) store.markDirty(kv.first);
        for (const ChunkId& c : w.contentRecords().stamped()) store.markDirty(c);
        for (auto& [c, en] : store.entriesForUpdate())
            if (!en.groups.empty()) en.dirty = true;
        for (SpaceId s : store.unloadedSpaces()) store.markDirty(spaceRecordKey(s));
        out.created        = true;
        out.ok             = true;
        out.commit         = commit();
        out.load.committed = true;
        if (synchronousIo() && !lastError.empty()) return fail(lastError);
        out.openSeconds = nowSeconds() - start;
        return out;
    }
    // Existing commits: the header first
    std::vector<uint8_t> headerBytes;
    std::string engineVersionOfDirectory;
    std::vector<std::shared_ptr<const SchemaTable>> tables;
    if (!hasHeader) return fail("the header is missing");
    if (!fs->readAll(kDirHeaderName, headerBytes, &e) ||
        !decodeDirHeader(headerBytes, engineVersionOfDirectory, tables, e))
        return fail(e);
    std::sort(worldFiles.rbegin(), worldFiles.rend());
    std::vector<std::string> corruptCandidates;
    std::string lastCorruption;
    for (size_t ci = 0; ci < worldFiles.size(); ++ci)
    {
        const uint64_t c       = worldFiles[ci];
        const std::string name = worldFileName(c);
        std::vector<uint8_t> bytes;
        WorldFile wf;
        if (!fs->readAll(name, bytes, &e) || !checkTrailer(bytes))
        {
            corruptCandidates.push_back(name);
            lastCorruption = name + ": the checksum failed";
            continue;
        }
        if (!decodeWorldFile(bytes, wf, e))
        {
            corruptCandidates.push_back(name);
            lastCorruption = name + ": " + e;
            continue;
        }
        if (wf.commit != c) return fail(name + ": it names commit " + std::to_string(wf.commit));
        ChunkIndex index;
        if (!decodeIndex(wf.index.data(), wf.index.size(), index, e)) return fail(name + ": " + e);
        if (wf.tablesNeeded > tables.size())
            return fail(name + ": it needs " + std::to_string(wf.tablesNeeded) + " schema tables, the header has " +
                        std::to_string(tables.size()));
        // Each file of the index must exist and reach its committed end
        bool corrupt = false;
        for (const auto& [k, end] : index.files)
        {
            uint64_t size = 0;
            if (!fs->fileSize(k.path(), size, nullptr) || size < end)
            {
                corrupt        = true;
                lastCorruption = name + ": " + k.path() + " is missing or shorter than its committed end";
            }
        }
        // The staged state from the index and the records
        DirectoryLoad dir;
        dir.store.fromDirectory = true;
        dir.store.tables        = tables;
        std::map<SpaceId, TerrainSection::Space> terrainBySpace;
        std::map<SpaceId, ChunkRecordSection::Space> recordsBySpace;
        uint64_t recordsRead = 0;
        for (const IndexEntry& ie : index.entries)
        {
            if (corrupt) break;
            const std::string where = name + ": " + chunkName(ie.key);
            if (ie.key.space >= w.spaceCount()) return fail(where + ": not a space of this world");
            for (const IndexSummary::Group& g : ie.summary.groups)
                if (g.schema >= tables.size())
                    return fail(where + ": a group uses schema table " + std::to_string(g.schema) +
                                ", the header has " + std::to_string(tables.size()));
            ChunkEntry ce;
            ce.indexed = true;
            ce.record  = ie.location;
            ce.summary = ie.summary;
            RecordContent rc;
            const bool needRecord = ie.kind == kRecordKindSpace || ie.summary.needsRecord();
            if (needRecord)
            {
                bool isCorrupt = false;
                if (!readRecord(*fs, *ie.location, ie.key, rc, e, &isCorrupt))
                {
                    if (!isCorrupt) return fail(where + ": " + e);
                    corrupt        = true;
                    lastCorruption = where + ": " + e;
                    break;
                }
                recordsRead++;
            }
            if (ie.kind == kRecordKindSpace)
            {
                dir.store.unloadedSpaces.insert(ie.key.space);
                if (rc.hasSpaceGroup)
                {
                    StoredGroup g;
                    g.bytes       = rc.spaceGroup.bytes;
                    g.schema      = rc.spaceGroup.schema;
                    g.hash        = fnv1a64Bytes(g.bytes->data(), g.bytes->size());
                    g.recordIndex = 0;
                    if (g.schema >= tables.size()) return fail(where + ": the group's schema table is out of range");
                    dir.store.groups[ie.key].push_back(g);
                }
                dir.store.indexEntries[ie.key] = std::move(ce);
                continue;
            }
            const int32_t n = w.spaceTemplate(ie.key.space).sizeChunks;
            if (ie.key.x < 0 || ie.key.z < 0 || ie.key.x >= n || ie.key.z >= n)
                return fail(where + ": outside the space");
            for (size_t k = 0; k < ie.summary.groups.size(); ++k)
            {
                StoredGroup g;
                g.schema      = ie.summary.groups[k].schema;
                g.hash        = ie.summary.groups[k].hash;
                g.recordIndex = int32_t(k);
                ce.groups.push_back(g);
            }
            if (ie.summary.sequence > 0)
            {
                TerrainSection::Space& sp = terrainBySpace[ie.key.space];
                const SpaceTemplate& tpl  = w.spaceTemplate(ie.key.space);
                sp.space                  = ie.key.space;
                sp.spacing                = uint8_t(tpl.baseSpacing);
                sp.N                      = uint16_t(256 / tpl.baseSpacing + 1);
                TerrainSection::Chunk tc;
                tc.x        = ie.key.x;
                tc.z        = ie.key.z;
                tc.sequence = ie.summary.sequence;
                if (ie.summary.deltas)
                {
                    if (!rc.hasTerrain || !rc.hasDeltas || rc.sequence != ie.summary.sequence)
                        return fail(where + ": the record's terrain differs from its index entry");
                    if (rc.N != sp.N)
                        return fail(where + ": saved with N " + std::to_string(rc.N) + ", the world has " +
                                    std::to_string(sp.N));
                    tc.hasDeltas = true;
                    tc.deltas    = std::move(rc.deltas);
                }
                sp.chunks.push_back(std::move(tc));
            }
            if (ie.summary.record)
            {
                ChunkRecordSection::Space& sp = recordsBySpace[ie.key.space];
                sp.space                      = ie.key.space;
                ChunkRecordSection::Record r;
                r.x                = ie.key.x;
                r.z                = ie.key.z;
                r.placementVersion = ie.summary.placementVersion;
                r.stamped          = ie.summary.stamped;
                if (ie.summary.tombstones > 0)
                {
                    if (!rc.hasRecord || rc.tombstones.size() != ie.summary.tombstones)
                        return fail(where + ": the record's tombstones differ from its index entry");
                    r.tombstones = rc.tombstones;
                }
                sp.records.push_back(std::move(r));
            }
            dir.store.indexEntries[ie.key] = std::move(ce);
        }
        if (corrupt)
        {
            corruptCandidates.push_back(name);
            continue;
        }
        if (!terrainBySpace.empty())
        {
            dir.terrain.emplace();
            for (auto& kv : terrainBySpace) dir.terrain->spaces.push_back(std::move(kv.second));
        }
        if (!recordsBySpace.empty())
        {
            dir.records.emplace();
            for (auto& kv : recordsBySpace) dir.records->spaces.push_back(std::move(kv.second));
        }
        // Validate-then-commit through the loader: a semantic error is an open error at once (the newest commit is
        // never discarded for a build mismatch)
        store.setTracking(true);
        out.load = w.loadDirectoryState(wf.blob, dir);
        if (!out.load.committed)
        {
            store.setTracking(false);
            return fail(name + ": " + out.load.error);
        }
        out.recordsRead = recordsRead;
        store.statsForUpdate().recordsRead += recordsRead;
        out.commit      = c;
        out.usedBackup  = ci > 0;
        committedCommit = c;
        headerTables    = tables.size();
        headerWritten   = true;
        // The kept commits' region ends (the loaded one and its predecessor)
        std::map<RegionKey, uint64_t> keptEnds;
        for (const auto& kv : index.files) keptEnds[kv.first] = kv.second;
        files = keptEnds;
        backupFiles.clear();
        std::vector<uint8_t> prevBytes;
        WorldFile prev;
        ChunkIndex prevIndex;
        std::string pe;
        if (fs->exists(worldFileName(c - 1)) && fs->readAll(worldFileName(c - 1), prevBytes, nullptr) &&
            decodeWorldFile(prevBytes, prev, pe) && decodeIndex(prev.index.data(), prev.index.size(), prevIndex, pe))
            for (const auto& kv : prevIndex.files)
            {
                backupFiles[kv.first] = kv.second;
                keptEnds[kv.first]    = std::max(keptEnds[kv.first], kv.second);
            }
        regionBytes = 0;
        for (const auto& kv : files) regionBytes += kv.second;
        if (!corruptCandidates.empty())
            SAIGA_LOG("save", "%s: loaded commit %llu (the newer world-state files are corrupt: %s)",
                      fs->describe().c_str(), (unsigned long long)c, lastCorruption.c_str());
        cleanupAfterOpen(c, keptEnds, corruptCandidates);
        afterLoad();
        out.ok          = true;
        out.openSeconds = nowSeconds() - start;
        return out;
    }
    return fail("no valid commit (" + (lastCorruption.empty() ? std::string("none") : lastCorruption) +
                "): nothing was changed");
}

void SaveDirectory::cleanupAfterOpen(uint64_t loaded, const std::map<RegionKey, uint64_t>& keptEnds,
                                     const std::vector<std::string>& corrupt)
{
    std::string e;
    // Newer invalid world-state files are kept for diagnosis
    for (const std::string& n : corrupt)
    {
        SAIGA_LOG("save", "%s: %s failed its checksum: renamed to %s.corrupt", fs->describe().c_str(), n.c_str(),
                  n.c_str());
        if (!fs->rename(n, n + ".corrupt", &e)) SAIGA_LOG("save", "rename %s: %s", n.c_str(), e.c_str());
    }
    std::vector<std::string> names;
    fs->list("", names, nullptr);
    for (const std::string& n : names)
    {
        uint64_t c       = 0;
        int kind         = 0;
        const bool world = parseWorldName(n, c, kind);
        const bool tmp   = endsWith(n, ".tmp");
        if (tmp || (world && kind == 0 && c != loaded && c + 1 != loaded))
            if (!fs->remove(n, &e)) SAIGA_LOG("save", "remove %s: %s", n.c_str(), e.c_str());
    }
    std::vector<std::string> spaces;
    if (fs->exists("regions") && fs->list("regions", spaces, nullptr))
        for (const std::string& s : spaces)
        {
            std::vector<std::string> list;
            if (!fs->list("regions/" + s, list, nullptr)) continue;
            for (const std::string& n : list)
            {
                const std::string p = "regions/" + s + "/" + n;
                RegionKey k;
                if (!parseRegionName(SpaceId(std::strtoul(s.c_str(), nullptr, 10)), n, k))
                {
                    if (endsWith(n, ".tmp")) fs->remove(p, nullptr);
                    continue;
                }
                auto it = keptEnds.find(k);
                if (it == keptEnds.end())
                {
                    if (!fs->remove(p, &e)) SAIGA_LOG("save", "remove %s: %s", p.c_str(), e.c_str());
                    continue;
                }
                uint64_t size = 0;
                if (fs->fileSize(p, size, nullptr) && size > it->second) fs->truncate(p, it->second, nullptr);
            }
        }
}

}  // namespace Engine
}  // namespace Saiga
