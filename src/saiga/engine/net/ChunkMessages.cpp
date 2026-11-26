/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/net/ChunkMessages.h"

#include "saiga/engine/content/ChunkRecords.h"
#include "saiga/engine/net/WireFormat.h"
#include "saiga/engine/terrain/TerrainDeltaCodec.h"

#include <zstd.h>

namespace Saiga
{
namespace Engine
{
namespace
{
constexpr size_t kZstdBodyBytes  = 1024;
constexpr uint8_t kSnapshotFlags = kSnapshotZstd | kSnapshotDeltas | kSnapshotRecord;

uint64_t keyValue(const InstanceKey& k)
{
    return (uint64_t(k.generatorSlot) << 24) | uint64_t(k.counter);
}
}  // namespace

void encodeChunkInterest(const ChunkInterestMessage& m, std::vector<uint8_t>& out)
{
    putU8(out, m.all ? kInterestAll : 0);
    putU32(out, m.epoch);
    putU16(out, m.space);
    if (m.all) return;
    putU16(out, uint16_t(m.removes.size()));
    putU16(out, uint16_t(m.adds.size()));
    for (const auto& r : m.removes)
    {
        putI32(out, r.first);
        putI32(out, r.second);
    }
    for (const ChunkInterestMessage::Add& a : m.adds)
    {
        putI32(out, a.x);
        putI32(out, a.z);
        putU32(out, a.ticket);
    }
}

bool decodeChunkInterest(const uint8_t* data, size_t size, ChunkInterestMessage& out, std::string& error)
{
    out = ChunkInterestMessage();
    WireReader r(data, size);
    const uint8_t flags = r.u8();
    out.epoch           = r.u32();
    out.space           = r.u16();
    if (!r.ok) return (error = "ChunkInterest: truncated", false);
    if (flags & ~kInterestAll) return (error = "ChunkInterest: unknown flags", false);
    out.all = (flags & kInterestAll) != 0;
    if (out.all)
    {
        if (!r.atEnd()) return (error = "ChunkInterest: entries with the all flag", false);
        return true;
    }
    const uint16_t removes = r.u16();
    const uint16_t adds    = r.u16();
    if (!r.ok) return (error = "ChunkInterest: truncated", false);
    const size_t total = size_t(removes) + adds;
    if (total == 0 || total > kMaxInterestEntries)
        return (error = "ChunkInterest: " + std::to_string(total) + " entries (1..4096)", false);
    if (size_t(removes) * 8 + size_t(adds) * 12 != r.remaining())
        return (error = "ChunkInterest: the entries do not match the payload", false);
    out.removes.resize(removes);
    for (auto& e : out.removes)
    {
        e.first  = r.i32();
        e.second = r.i32();
    }
    out.adds.resize(adds);
    for (ChunkInterestMessage::Add& a : out.adds)
    {
        a.x      = r.i32();
        a.z      = r.i32();
        a.ticket = r.u32();
    }
    if (!r.atEnd()) return (error = "ChunkInterest: bytes left over", false);
    return true;
}

void encodeChunkIndex(const ChunkIndexMessage& m, std::vector<uint8_t>& out)
{
    putU32(out, m.epoch);
    putU16(out, m.space);
    writeVarint(out, m.chunks.size());
    for (const auto& c : m.chunks)
    {
        putI32(out, c.first);
        putI32(out, c.second);
    }
}

bool decodeChunkIndex(const uint8_t* data, size_t size, ChunkIndexMessage& out, std::string& error)
{
    out = ChunkIndexMessage();
    WireReader r(data, size);
    out.epoch            = r.u32();
    out.space            = r.u16();
    const uint64_t count = r.varint();
    if (!r.ok) return (error = "ChunkIndex: truncated", false);
    if (count > kMaxIndexChunks || count * 8 != r.remaining())
        return (error = "ChunkIndex: the count does not match the payload", false);
    out.chunks.resize(size_t(count));
    for (size_t i = 0; i < out.chunks.size(); ++i)
    {
        out.chunks[i].first  = r.i32();
        out.chunks[i].second = r.i32();
        if (i > 0 && !(out.chunks[i - 1] < out.chunks[i]))
            return (error = "ChunkIndex: the chunks are not ascending and unique", false);
    }
    if (!r.atEnd()) return (error = "ChunkIndex: bytes left over", false);
    return true;
}

void encodeChunkSnapshot(const ChunkSnapshotMessage& m, std::vector<uint8_t>& out)
{
    std::vector<uint8_t> body;
    if (m.hasRecord)
    {
        putU16(body, m.placementVersion);
        writeVarint(body, m.keys.size());
        uint64_t previous = 0;
        for (size_t i = 0; i < m.keys.size(); ++i)
        {
            const uint64_t v = keyValue(m.keys[i]);
            writeVarint(body, i == 0 ? v : v - previous);
            previous = v;
        }
    }
    if (m.hasDeltas)
    {
        std::vector<uint8_t> e;
        TerrainDeltaCodec::encode(m.deltas.data(), m.deltas.size(), false, e);
        writeVarint(body, e.size());
        body.insert(body.end(), e.begin(), e.end());
    }
    bool zstd = false;
    std::vector<uint8_t> packed;
    if (body.size() >= kZstdBodyBytes)
    {
        packed.resize(ZSTD_compressBound(body.size()));
        const size_t n = ZSTD_compress(packed.data(), packed.size(), body.data(), body.size(), kTerrainDeltaZstdLevel);
        if (!ZSTD_isError(n) && n < body.size())
        {
            packed.resize(n);
            zstd = true;
        }
    }
    putU8(out, uint8_t((zstd ? kSnapshotZstd : 0) | (m.hasDeltas ? kSnapshotDeltas : 0) |
                       (m.hasRecord ? kSnapshotRecord : 0)));
    putU32(out, m.epoch);
    putU32(out, m.ticket);
    putU16(out, m.chunk.space);
    putI32(out, m.chunk.x);
    putI32(out, m.chunk.z);
    putU32(out, m.sequence);
    putU16(out, m.N);
    const std::vector<uint8_t>& b = zstd ? packed : body;
    out.insert(out.end(), b.begin(), b.end());
}

bool decodeChunkSnapshot(const uint8_t* data, size_t size, ChunkSnapshotMessage& out, std::string& error)
{
    out = ChunkSnapshotMessage();
    WireReader r(data, size);
    const uint8_t flags = r.u8();
    out.epoch           = r.u32();
    out.ticket          = r.u32();
    out.chunk.space     = r.u16();
    out.chunk.x         = r.i32();
    out.chunk.z         = r.i32();
    out.sequence        = r.u32();
    out.N               = r.u16();
    if (!r.ok) return (error = "ChunkSnapshot: truncated header", false);
    if (flags & ~kSnapshotFlags) return (error = "ChunkSnapshot: unknown flags", false);
    out.hasDeltas = (flags & kSnapshotDeltas) != 0;
    out.hasRecord = (flags & kSnapshotRecord) != 0;
    if (out.N > 257) return (error = "ChunkSnapshot: N above 257", false);
    if (out.hasDeltas && out.N == 0) return (error = "ChunkSnapshot: deltas without a sample grid", false);
    // The body
    std::vector<uint8_t> unpacked;
    const uint8_t* body = data + r.pos;
    size_t bodySize     = r.remaining();
    if (flags & kSnapshotZstd)
    {
        const unsigned long long content = ZSTD_getFrameContentSize(body, bodySize);
        if (content == ZSTD_CONTENTSIZE_ERROR || content == ZSTD_CONTENTSIZE_UNKNOWN || content > kMaxChunkSnapshotBody)
            return (error = "ChunkSnapshot: a compressed body of unknown size or above 8 MB", false);
        unpacked.resize(size_t(content));
        const size_t n = ZSTD_decompress(unpacked.data(), unpacked.size(), body, bodySize);
        if (ZSTD_isError(n) || n != unpacked.size()) return (error = "ChunkSnapshot: a corrupt compressed body", false);
        body     = unpacked.data();
        bodySize = unpacked.size();
    }
    WireReader b(body, bodySize);
    if (out.hasRecord)
    {
        out.placementVersion = b.u16();
        const uint64_t count = b.varint();
        if (!b.ok) return (error = "ChunkSnapshot: a truncated record", false);
        if (count > kMaxChunkTombstones || count > b.remaining())
            return (error = "ChunkSnapshot: more keys than fit (at most 2^20)", false);
        out.keys.resize(size_t(count));
        uint64_t previous = 0;
        for (size_t i = 0; i < out.keys.size(); ++i)
        {
            const uint64_t d = b.varint();
            if (!b.ok) return (error = "ChunkSnapshot: a truncated key", false);
            if (i > 0 && d == 0) return (error = "ChunkSnapshot: the keys are not strictly ascending", false);
            const uint64_t v = i == 0 ? d : previous + d;
            if (v < previous || v >= (uint64_t(256) << 24))
                return (error = "ChunkSnapshot: a key above slot 255", false);
            out.keys[i].generatorSlot = uint32_t(v >> 24);
            out.keys[i].counter       = uint32_t(v & 0xFFFFFF);
            previous                  = v;
        }
    }
    if (out.hasDeltas)
    {
        const uint64_t len = b.varint();
        const uint8_t* e   = b.bytes(len);
        if (!b.ok) return (error = "ChunkSnapshot: truncated deltas", false);
        out.deltas.resize(size_t(out.N) * out.N);
        std::string codecError;
        if (!TerrainDeltaCodec::decode(e, size_t(len), false, out.deltas.size(), out.deltas.data(), &codecError))
            return (error = "ChunkSnapshot: " + codecError, false);
    }
    if (!b.atEnd()) return (error = "ChunkSnapshot: bytes left over", false);
    return true;
}

}  // namespace Engine
}  // namespace Saiga
