/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/terrain/TerrainMessages.h"

#include "saiga/engine/net/Varint.h"
#include "saiga/engine/terrain/TerrainDeltaCodec.h"

namespace Saiga
{
namespace Engine
{
namespace
{
void putU8(std::vector<uint8_t>& out, uint8_t v)
{
    out.push_back(v);
}
void putU16(std::vector<uint8_t>& out, uint16_t v)
{
    out.push_back(uint8_t(v));
    out.push_back(uint8_t(v >> 8));
}
void putU32(std::vector<uint8_t>& out, uint32_t v)
{
    for (int i = 0; i < 4; ++i) out.push_back(uint8_t(v >> (8 * i)));
}

// Untrusted input: every error is remembered, never asserted
struct Reader
{
    const uint8_t* d;
    size_t n;
    size_t pos = 0;
    bool ok    = true;
    uint32_t le(int bytes)
    {
        if (!ok || n - pos < size_t(bytes))
        {
            ok = false;
            return 0;
        }
        uint32_t v = 0;
        for (int i = 0; i < bytes; ++i) v |= uint32_t(d[pos++]) << (8 * i);
        return v;
    }
    uint8_t u8() { return uint8_t(le(1)); }
    uint16_t u16() { return uint16_t(le(2)); }
    uint32_t u32() { return le(4); }
    int32_t i32() { return int32_t(le(4)); }
    uint64_t varint()
    {
        uint64_t v = 0;
        if (ok && !readVarint(d, n, pos, v)) ok = false;
        return ok ? v : 0;
    }
    const uint8_t* bytes(uint64_t len)
    {
        if (!ok || len > n - pos)
        {
            ok = false;
            return nullptr;
        }
        const uint8_t* p = d + pos;
        pos += size_t(len);
        return p;
    }
};

// The encoded deltas with their varint length
void putDeltas(std::vector<uint8_t>& out, const int32_t* values, size_t count, bool zstd)
{
    std::vector<uint8_t> e;
    TerrainDeltaCodec::encode(values, count, zstd, e);
    writeVarint(out, e.size());
    out.insert(out.end(), e.begin(), e.end());
}

// zstd pays off from about this many values
constexpr size_t kZstdValues = 256;
}  // namespace

void encodeTerrainModification(uint32_t epoch, SpaceId space, const std::vector<TerrainPart>& parts,
                               std::vector<uint8_t>& out)
{
    size_t values = 0;
    for (const TerrainPart& p : parts) values += p.deltas.size();
    const bool zstd = values >= kZstdValues;
    putU8(out, zstd ? 1 : 0);
    putU32(out, epoch);
    putU16(out, space);
    putU16(out, uint16_t(parts.size()));
    for (const TerrainPart& p : parts)
    {
        putU32(out, uint32_t(p.chunk.x));
        putU32(out, uint32_t(p.chunk.z));
        putU32(out, p.sequence);
        putU16(out, uint16_t(p.x0));
        putU16(out, uint16_t(p.z0));
        putU16(out, uint16_t(p.w));
        putU16(out, uint16_t(p.h));
        putDeltas(out, p.deltas.data(), p.deltas.size(), zstd);
    }
}

bool decodeTerrainModification(const uint8_t* data, size_t size, TerrainModificationMessage& out, std::string& error)
{
    Reader r{data, size};
    const uint8_t flags = r.u8();
    out.epoch           = r.u32();
    out.space           = r.u16();
    const uint16_t n    = r.u16();
    if (!r.ok) return (error = "TerrainModification: truncated header", false);
    if (flags & ~uint8_t(1)) return (error = "TerrainModification: unknown flags", false);
    if (n == 0) return (error = "TerrainModification: no parts", false);
    out.parts.clear();
    out.parts.resize(n);
    for (TerrainPart& p : out.parts)
    {
        p.chunk.space      = out.space;
        p.chunk.x          = r.i32();
        p.chunk.z          = r.i32();
        p.sequence         = r.u32();
        p.x0               = r.u16();
        p.z0               = r.u16();
        p.w                = r.u16();
        p.h                = r.u16();
        const uint64_t len = r.varint();
        const uint8_t* b   = r.bytes(len);
        if (!r.ok) return (error = "TerrainModification: truncated part", false);
        // At most one 257 x 257 tile per part
        if (p.w < 1 || p.h < 1 || p.w > 257 || p.h > 257) return (error = "TerrainModification: bad box size", false);
        p.deltas.resize(size_t(p.w) * size_t(p.h));
        std::string e;
        if (!TerrainDeltaCodec::decode(b, size_t(len), flags & 1, p.deltas.size(), p.deltas.data(), &e))
            return (error = "TerrainModification: " + e, false);
    }
    if (r.pos != size) return (error = "TerrainModification: bytes left over", false);
    return true;
}

}  // namespace Engine
}  // namespace Saiga
