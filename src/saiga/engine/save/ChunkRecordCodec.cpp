/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/save/ChunkRecordCodec.h"

#include "saiga/core/util/assert.h"
#include "saiga/engine/terrain/TerrainDeltaCodec.h"

#include <algorithm>
#include <map>
#include <tuple>
#include <zstd.h>

namespace Saiga
{
namespace Engine
{
namespace
{
std::string chunkName(const ChunkId& c)
{
    if (isSpaceRecordKey(c)) return "the space record of space " + std::to_string(c.space);
    return "chunk (" + std::to_string(c.space) + ", " + std::to_string(c.x) + ", " + std::to_string(c.z) + ")";
}

uint64_t getU64(const uint8_t* p)
{
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= uint64_t(p[i]) << (8 * i);
    return v;
}

void writeBytes(BinaryWriter& w, std::vector<uint8_t>& out, const std::vector<uint8_t>& bytes)
{
    w.write(uint64_t(bytes.size()));
    out.insert(out.end(), bytes.begin(), bytes.end());
}

// A section: tag, version, length, then fill(writer)
template <class F>
void section(std::vector<uint8_t>& out, uint32_t tag, uint32_t version, F&& fill)
{
    BinaryWriter w(out);
    w.write(tag);
    w.write(version);
    const size_t lengthAt = out.size();
    w.write(uint64_t(0));
    const size_t start = out.size();
    fill(w);
    patchU64(out, lengthAt, uint64_t(out.size() - start));
}

bool readGroupBytes(BinaryReader& r, const uint8_t* base, GroupBytes& out, std::string& error)
{
    uint64_t length = 0;
    r.read(length);
    if (!r.ok() || length > r.remaining())
    {
        error = "a group length does not fit";
        return false;
    }
    const size_t at = r.position();
    out             = std::make_shared<const std::vector<uint8_t>>(base + at, base + at + length);
    r.skip(length);
    return true;
}
}  // namespace

void zstdCompress(const uint8_t* data, size_t size, int level, std::vector<uint8_t>& out)
{
    const size_t at = out.size();
    out.resize(at + ZSTD_compressBound(size));
    const size_t n = ZSTD_compress(out.data() + at, out.size() - at, data, size, level);
    SAIGA_ASSERT(!ZSTD_isError(n), "zstd compression failed");
    out.resize(at + n);
}

bool zstdDecompress(const uint8_t* data, size_t size, std::vector<uint8_t>& out, std::string& error)
{
    const unsigned long long len = ZSTD_getFrameContentSize(data, size);
    if (len == ZSTD_CONTENTSIZE_ERROR || len == ZSTD_CONTENTSIZE_UNKNOWN)
    {
        error = "not a zstd frame with a content size";
        return false;
    }
    if (len > kMaxUncompressedPart)
    {
        error = "the frame claims " + std::to_string(len) + " bytes (more than 256 MB)";
        return false;
    }
    out.resize(size_t(len));
    const size_t n = ZSTD_decompress(out.data(), out.size(), data, size);
    if (ZSTD_isError(n) || n != len)
    {
        error = "the zstd frame does not decompress";
        return false;
    }
    return true;
}

// ---- records

bool encodeRecord(const RecordContent& c, int zstdLevel, std::vector<uint8_t>& out)
{
    std::vector<uint8_t> payload;
    if (c.kind == kRecordKindChunk)
    {
        if (c.hasTerrain)
            section(payload, kTagRecTerrain, kRecTerrainVersion,
                    [&](BinaryWriter& w)
                    {
                        w.write(c.sequence);
                        w.write(uint8_t(c.hasDeltas ? 1 : 0));
                        if (!c.hasDeltas) return;
                        w.write(c.N);
                        std::vector<uint8_t> encoded;
                        TerrainDeltaCodec::encode(c.deltas.data(), c.deltas.size(), false, encoded);
                        writeBytes(w, payload, encoded);
                    });
        if (c.hasRecord)
            section(payload, kTagRecRecord, kRecRecordVersion,
                    [&](BinaryWriter& w)
                    {
                        w.write(c.placementVersion);
                        w.write(uint8_t(c.stamped ? 1 : 0));
                        w.write(uint32_t(c.tombstones.size()));
                        for (const InstanceKey& k : c.tombstones)
                        {
                            w.write(uint8_t(k.generatorSlot));
                            w.write(k.counter);
                        }
                    });
        if (!c.groups.empty())
            section(payload, kTagRecGroups, kRecGroupsVersion,
                    [&](BinaryWriter& w)
                    {
                        w.write(uint32_t(c.groups.size()));
                        for (const RecordGroup& g : c.groups)
                        {
                            w.write(g.schema);
                            writeBytes(w, payload, *g.bytes);
                        }
                    });
    }
    else
        section(payload, kTagRecSpace, kRecSpaceVersion,
                [&](BinaryWriter& w)
                {
                    w.write(uint8_t(c.hasSpaceGroup ? 1 : 0));
                    if (!c.hasSpaceGroup) return;
                    w.write(c.spaceGroup.schema);
                    writeBytes(w, payload, *c.spaceGroup.bytes);
                });
    if (payload.size() > kMaxUncompressedPart) return false;
    std::vector<uint8_t> frame;
    zstdCompress(payload.data(), payload.size(), zstdLevel, frame);
    const size_t start = out.size();
    BinaryWriter w(out);
    w.write(kTagRecord);
    w.write(kRecordFormat);
    w.write(c.kind);
    w.write(uint8_t(0));
    w.write(c.key.space);
    w.write(uint16_t(0));
    w.write(c.kind == kRecordKindChunk ? c.key.x : 0);
    w.write(c.kind == kRecordKindChunk ? c.key.z : 0);
    w.write(c.commit);
    w.write(uint32_t(frame.size()));
    out.insert(out.end(), frame.begin(), frame.end());
    const uint64_t h = fnv1a64Bytes(out.data() + start, out.size() - start);
    w.write(h);
    return true;
}

bool decodeRecordHeader(const uint8_t* data, size_t size, RecordHeader& out, std::string& error)
{
    if (size < kRecordHeaderSize)
    {
        error = "a record header does not fit";
        return false;
    }
    BinaryReader r(data, kRecordHeaderSize);
    uint32_t magic  = 0;
    uint16_t format = 0, reserved = 0;
    uint8_t flags = 0;
    int32_t x = 0, z = 0;
    r.read(magic);
    r.read(format);
    r.read(out.kind);
    r.read(flags);
    r.read(out.key.space);
    r.read(reserved);
    r.read(x);
    r.read(z);
    r.read(out.commit);
    r.read(out.payloadLength);
    if (magic != kTagRecord)
    {
        error = "not a record (magic)";
        return false;
    }
    if (format > kRecordFormat || format == 0)
    {
        error = "record format " + std::to_string(format) + " (this build reads " + std::to_string(kRecordFormat) + ")";
        return false;
    }
    if (out.kind > kRecordKindSpace || flags != 0)
    {
        error = "unknown record kind or flags";
        return false;
    }
    if (out.kind == kRecordKindSpace)
        out.key = spaceRecordKey(out.key.space);
    else
    {
        out.key.x = x;
        out.key.z = z;
    }
    return true;
}

bool decodeRecord(const uint8_t* data, size_t size, RecordContent& out, std::string& error)
{
    RecordHeader h;
    if (!decodeRecordHeader(data, size, h, error)) return false;
    const std::string where = chunkName(h.key);
    if (size != kRecordHeaderSize + size_t(h.payloadLength) + 8)
    {
        error = where + ": the record length does not match its payload";
        return false;
    }
    if (fnv1a64Bytes(data, size - 8) != getU64(data + size - 8))
    {
        error = where + ": the record checksum failed";
        return false;
    }
    out        = RecordContent();
    out.kind   = h.kind;
    out.key    = h.key;
    out.commit = h.commit;
    std::vector<uint8_t> payload;
    std::string e;
    if (!zstdDecompress(data + kRecordHeaderSize, h.payloadLength, payload, e))
    {
        error = where + ": " + e;
        return false;
    }
    BinaryReader r(payload.data(), payload.size());
    // The fixed order: RTER, RREC, RGRP (kind 0) or RSPC (kind 1); each at most once
    const uint32_t order[]      = {kTagRecTerrain, kTagRecRecord, kTagRecGroups, kTagRecSpace};
    const uint32_t maxVersion[] = {kRecTerrainVersion, kRecRecordVersion, kRecGroupsVersion, kRecSpaceVersion};
    int next                    = 0;
    while (!r.atEnd())
    {
        uint32_t tag = 0, version = 0;
        uint64_t length = 0;
        r.read(tag);
        r.read(version);
        r.read(length);
        if (!r.ok() || length > r.remaining())
        {
            error = where + ": a section header does not fit";
            return false;
        }
        int slot = -1;
        for (int i = 0; i < 4; ++i)
            if (order[i] == tag) slot = i;
        if (slot < 0)
        {
            error = where + ": the unknown section " + tagName(tag);
            return false;
        }
        const bool kindOk = h.kind == kRecordKindSpace ? slot == 3 : slot < 3;
        if (!kindOk || slot < next)
        {
            error = where + ": the section " + tagName(tag) + " is out of order, repeated or of the other record kind";
            return false;
        }
        if (version == 0 || version > maxVersion[slot])
        {
            error = where + ": the section " + tagName(tag) + " has version " + std::to_string(version) +
                    " (this build reads " + std::to_string(maxVersion[slot]) + ")";
            return false;
        }
        next = slot + 1;
        BinaryReader s(payload.data() + r.position(), size_t(length));
        const uint8_t* sbase = payload.data() + r.position();
        r.skip(length);
        auto sectionFail = [&](const std::string& m)
        {
            error = where + ": " + tagName(tag) + ": " + m;
            return false;
        };
        if (tag == kTagRecTerrain)
        {
            out.hasTerrain = true;
            uint8_t flags  = 0;
            s.read(out.sequence);
            s.read(flags);
            if (!s.ok()) return sectionFail(s.error());
            if (out.sequence == 0 || flags > 1) return sectionFail("sequence 0 or unknown flags");
            out.hasDeltas = flags == 1;
            if (out.hasDeltas)
            {
                uint64_t encoded = 0;
                s.read(out.N);
                s.read(encoded);
                if (!s.ok() || encoded > s.remaining()) return sectionFail("the delta does not fit");
                if (out.N != 129 && out.N != 257)
                    return sectionFail("N " + std::to_string(out.N) + " is not 129 or 257");
                out.deltas.resize(size_t(out.N) * out.N);
                std::string de;
                if (!TerrainDeltaCodec::decode(sbase + s.position(), size_t(encoded), false, out.deltas.size(),
                                               out.deltas.data(), &de))
                    return sectionFail(de);
                s.skip(encoded);
                bool any = false;
                for (int32_t v : out.deltas) any = any || v != 0;
                if (!any) return sectionFail("deltas present without a non-zero value");
            }
        }
        else if (tag == kTagRecRecord)
        {
            out.hasRecord   = true;
            uint8_t stamped = 0;
            uint32_t count  = 0;
            s.read(out.placementVersion);
            s.read(stamped);
            s.read(count);
            if (!s.ok() || stamped > 1 || uint64_t(count) * 5 > s.remaining())
                return sectionFail("malformed record entry");
            out.stamped = stamped == 1;
            out.tombstones.resize(count);
            for (InstanceKey& k : out.tombstones)
            {
                uint8_t slotIndex = 0;
                s.read(slotIndex);
                s.read(k.counter);
                k.generatorSlot = slotIndex;
            }
            if (!s.ok()) return sectionFail(s.error());
        }
        else if (tag == kTagRecGroups)
        {
            uint32_t count = 0;
            s.read(count);
            if (!s.ok() || count == 0 || uint64_t(count) * 12 > s.remaining()) return sectionFail("bad group count");
            out.groups.resize(count);
            for (RecordGroup& g : out.groups)
            {
                s.read(g.schema);
                std::string ge;
                if (!s.ok() || !readGroupBytes(s, sbase, g.bytes, ge)) return sectionFail(ge.empty() ? s.error() : ge);
            }
        }
        else
        {
            uint8_t flag = 0;
            s.read(flag);
            if (!s.ok() || flag > 1) return sectionFail("bad flag");
            out.hasSpaceGroup = flag == 1;
            if (out.hasSpaceGroup)
            {
                s.read(out.spaceGroup.schema);
                std::string ge;
                if (!s.ok() || !readGroupBytes(s, sbase, out.spaceGroup.bytes, ge))
                    return sectionFail(ge.empty() ? s.error() : ge);
            }
        }
        if (!s.atEnd()) return sectionFail("bytes left over");
    }
    if (!r.ok())
    {
        error = where + ": " + r.error();
        return false;
    }
    return true;
}

// ---- the chunk index

void encodeIndex(const ChunkIndex& index, std::vector<uint8_t>& out)
{
    BinaryWriter w(out);
    w.write(kTagChunkIndex);
    w.write(kChunkIndexFormat);
    w.write(uint32_t(index.files.size()));
    // The ordinal of each file in the table (a linear scan per entry was quadratic in the region count).
    std::map<RegionKey, uint32_t> ordinal;
    for (const auto& [k, end] : index.files)
    {
        ordinal.emplace(k, uint32_t(ordinal.size()));
        w.write(k.space);
        w.write(k.rx);
        w.write(k.rz);
        w.write(k.generation);
        w.write(end);
    }
    w.write(uint32_t(index.entries.size()));
    for (const IndexEntry& e : index.entries)
    {
        w.write(e.kind);
        w.write(e.key.space);
        w.write(e.kind == kRecordKindChunk ? e.key.x : 0);
        w.write(e.kind == kRecordKindChunk ? e.key.z : 0);
        uint32_t file = 0xFFFFFFFFu;
        if (e.location)
        {
            auto it = ordinal.find(e.location->file);
            if (it != ordinal.end()) file = it->second;
            SAIGA_ASSERT(file != 0xFFFFFFFFu, "encodeIndex: a location names a file outside the table");
        }
        w.write(file);
        w.write(e.location ? e.location->offset : uint64_t(0));
        w.write(e.location ? e.location->length : uint32_t(0));
        w.write(e.location ? e.location->commit : uint64_t(0));
        const IndexSummary& s = e.summary;
        w.write(s.sequence);
        w.write(uint8_t((s.deltas ? 1 : 0) | (s.record ? 2 : 0) | (s.stamped ? 4 : 0)));
        if (s.deltas) w.write(s.deltaHash);
        if (s.record) w.write(s.placementVersion);
        w.write(s.tombstones);
        w.write(uint32_t(s.groups.size()));
        for (const IndexSummary::Group& g : s.groups)
        {
            w.write(g.schema);
            w.write(g.hash);
        }
    }
}

bool decodeIndex(const uint8_t* data, size_t size, ChunkIndex& out, std::string& error)
{
    out = ChunkIndex();
    BinaryReader r(data, size);
    uint32_t magic = 0, format = 0, files = 0, entries = 0;
    r.read(magic);
    r.read(format);
    if (!r.ok() || magic != kTagChunkIndex)
    {
        error = "CIDX: not a chunk index";
        return false;
    }
    if (format == 0 || format > kChunkIndexFormat)
    {
        error =
            "CIDX: format " + std::to_string(format) + " (this build reads " + std::to_string(kChunkIndexFormat) + ")";
        return false;
    }
    r.read(files);
    if (!r.ok() || uint64_t(files) * 22 > r.remaining())
    {
        error = "CIDX: the file table does not fit";
        return false;
    }
    for (uint32_t i = 0; i < files; ++i)
    {
        RegionKey k;
        uint64_t end = 0;
        r.read(k.space);
        r.read(k.rx);
        r.read(k.rz);
        r.read(k.generation);
        r.read(end);
        if (i > 0 && !(out.files.back().first < k))
        {
            error = "CIDX: the file table is not sorted and unique";
            return false;
        }
        out.files.push_back({k, end});
    }
    r.read(entries);
    if (!r.ok() || uint64_t(entries) * 48 > r.remaining())
    {
        error = "CIDX: the entries do not fit";
        return false;
    }
    out.entries.resize(entries);
    for (uint32_t i = 0; i < entries; ++i)
    {
        IndexEntry& e = out.entries[i];
        int32_t x = 0, z = 0;
        uint32_t file   = 0;
        uint64_t offset = 0, commit = 0;
        uint32_t length = 0;
        uint8_t flags   = 0;
        uint32_t groups = 0;
        r.read(e.kind);
        r.read(e.key.space);
        r.read(x);
        r.read(z);
        r.read(file);
        r.read(offset);
        r.read(length);
        r.read(commit);
        r.read(e.summary.sequence);
        r.read(flags);
        if (!r.ok() || e.kind > kRecordKindSpace || flags > 7)
        {
            error = "CIDX: entry " + std::to_string(i) + " is malformed";
            return false;
        }
        e.summary.deltas  = flags & 1;
        e.summary.record  = flags & 2;
        e.summary.stamped = flags & 4;
        if (e.summary.deltas) r.read(e.summary.deltaHash);
        if (e.summary.record) r.read(e.summary.placementVersion);
        r.read(e.summary.tombstones);
        r.read(groups);
        if (!r.ok() || uint64_t(groups) * 12 > r.remaining())
        {
            error = "CIDX: entry " + std::to_string(i) + ": the groups do not fit";
            return false;
        }
        e.summary.groups.resize(groups);
        for (IndexSummary::Group& g : e.summary.groups)
        {
            r.read(g.schema);
            r.read(g.hash);
        }
        if (e.kind == kRecordKindSpace)
            e.key = spaceRecordKey(e.key.space);
        else
        {
            e.key.x = x;
            e.key.z = z;
        }
        const std::string where = "CIDX: " + chunkName(e.key);
        if (file != 0xFFFFFFFFu)
        {
            if (file >= out.files.size())
            {
                error = where + ": the file index is out of range";
                return false;
            }
            RecordLocation loc;
            loc.file   = out.files[file].first;
            loc.offset = offset;
            loc.length = length;
            loc.commit = commit;
            if (length < kRecordHeaderSize + 8 || offset < kRegionHeaderSize || offset > out.files[file].second ||
                length > out.files[file].second - offset)
            {
                error = where + ": the record lies outside its file's committed end";
                return false;
            }
            if (!loc.file.sameRegion(regionKeyOf(e.key, loc.file.generation)))
            {
                error = where + ": the record lies in another region's file";
                return false;
            }
            e.location = loc;
        }
        else if (e.summary.needsRecord() || !e.summary.groups.empty() || e.kind == kRecordKindSpace)
        {
            error = where + ": the state needs a record, but the entry has none";
            return false;
        }
        if (i > 0)
        {
            const IndexEntry& p = out.entries[i - 1];
            if (std::make_tuple(p.kind, p.key) >= std::make_tuple(e.kind, e.key))
            {
                error = where + ": the entries are not sorted and unique";
                return false;
            }
        }
    }
    if (!r.atEnd())
    {
        error = "CIDX: bytes left over";
        return false;
    }
    return true;
}

// ---- the header

void encodeDirHeader(const std::string& engineVersion, const std::vector<std::shared_ptr<const SchemaTable>>& tables,
                     std::vector<uint8_t>& out)
{
    out.clear();
    BinaryWriter w(out);
    w.write(kTagDirHeader);
    w.write(kDirHeaderFormat);
    w.write(engineVersion);
    w.write(uint32_t(tables.size()));
    for (const auto& t : tables) t->write(w);
    appendTrailer(out);
}

bool decodeDirHeader(const std::vector<uint8_t>& bytes, std::string& engineVersion,
                     std::vector<std::shared_ptr<const SchemaTable>>& tables, std::string& error)
{
    if (!checkTrailer(bytes))
    {
        error = "header.sgh: the checksum failed";
        return false;
    }
    BinaryReader r(bytes.data(), bytes.size() - 8);
    uint32_t magic = 0, format = 0, count = 0;
    r.read(magic);
    r.read(format);
    if (!r.ok() || magic != kTagDirHeader)
    {
        error = "header.sgh: not a save directory header";
        return false;
    }
    if (format == 0 || format > kDirHeaderFormat)
    {
        error = "header.sgh: format " + std::to_string(format) + " (this build reads " +
                std::to_string(kDirHeaderFormat) + ")";
        return false;
    }
    r.read(engineVersion);
    r.read(count);
    if (!r.ok() || count > 65536)
    {
        error = "header.sgh: malformed";
        return false;
    }
    tables.clear();
    for (uint32_t i = 0; i < count; ++i)
    {
        auto t = std::make_shared<SchemaTable>();
        std::string e;
        if (!t->read(r, e))
        {
            error = "header.sgh: schema table " + std::to_string(i) + ": " + e;
            return false;
        }
        tables.push_back(std::move(t));
    }
    if (!r.atEnd())
    {
        error = "header.sgh: bytes left over";
        return false;
    }
    return true;
}

// ---- the world-state file

bool encodeWorldFile(const WorldFile& f, int zstdLevel, std::vector<uint8_t>& out)
{
    out.clear();
    if (f.blob.size() > kMaxUncompressedPart || f.index.size() > kMaxUncompressedPart) return false;
    BinaryWriter w(out);
    w.write(kTagWorldFile);
    w.write(kWorldFileFormat);
    w.write(f.commit);
    w.write(f.tablesNeeded);
    for (const std::vector<uint8_t>* part : {&f.blob, &f.index})
    {
        std::vector<uint8_t> frame;
        zstdCompress(part->data(), part->size(), zstdLevel, frame);
        w.write(uint64_t(part->size()));
        w.write(uint64_t(frame.size()));
        out.insert(out.end(), frame.begin(), frame.end());
    }
    appendTrailer(out);
    return true;
}

bool decodeWorldFile(const std::vector<uint8_t>& bytes, WorldFile& out, std::string& error)
{
    if (!checkTrailer(bytes))
    {
        error = "the checksum failed";
        return false;
    }
    BinaryReader r(bytes.data(), bytes.size() - 8);
    uint32_t magic = 0, format = 0;
    r.read(magic);
    r.read(format);
    r.read(out.commit);
    r.read(out.tablesNeeded);
    if (!r.ok() || magic != kTagWorldFile || format == 0 || format > kWorldFileFormat)
    {
        error = "not a world-state file of a known format";
        return false;
    }
    for (std::vector<uint8_t>* part : {&out.blob, &out.index})
    {
        uint64_t raw = 0, compressed = 0;
        r.read(raw);
        r.read(compressed);
        if (!r.ok() || compressed > r.remaining())
        {
            error = "a framed part does not fit";
            return false;
        }
        std::string e;
        if (!zstdDecompress(bytes.data() + r.position(), size_t(compressed), *part, e) || part->size() != raw)
        {
            error = "a framed part: " + (e.empty() ? std::string("the length differs") : e);
            return false;
        }
        r.skip(compressed);
    }
    if (!r.atEnd())
    {
        error = "bytes left over";
        return false;
    }
    return true;
}

// ---- region files

std::vector<uint8_t> encodeRegionHeader(const RegionKey& key)
{
    std::vector<uint8_t> out;
    BinaryWriter w(out);
    w.write(kTagRegionFile);
    w.write(kRegionFormat);
    w.write(key.space);
    w.write(uint16_t(0));
    w.write(key.rx);
    w.write(key.rz);
    w.write(key.generation);
    return out;
}

bool checkRegionHeader(const uint8_t* data, size_t size, const RegionKey& key, std::string& error)
{
    if (size < kRegionHeaderSize)
    {
        error = key.path() + ": the header does not fit";
        return false;
    }
    std::vector<uint8_t> expected = encodeRegionHeader(key);
    if (!std::equal(expected.begin(), expected.end(), data))
    {
        error = key.path() + ": the header does not match";
        return false;
    }
    return true;
}

}  // namespace Engine
}  // namespace Saiga
