/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The wire primitives of the S26 messages (docs/engine_design.md §13.2, §13.6): little-endian fixed-width integers,
// varints of Varint.h and strings as varint length + bytes. WireReader treats its input as untrusted: every error is
// remembered (ok false), never asserted, and nothing is read past the end.

#pragma once

#include "saiga/engine/net/Varint.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace Saiga
{
namespace Engine
{
inline void putU8(std::vector<uint8_t>& out, uint8_t v)
{
    out.push_back(v);
}
inline void putU16(std::vector<uint8_t>& out, uint16_t v)
{
    out.push_back(uint8_t(v));
    out.push_back(uint8_t(v >> 8));
}
inline void putU32(std::vector<uint8_t>& out, uint32_t v)
{
    for (int i = 0; i < 4; ++i) out.push_back(uint8_t(v >> (8 * i)));
}
inline void putU64(std::vector<uint8_t>& out, uint64_t v)
{
    for (int i = 0; i < 8; ++i) out.push_back(uint8_t(v >> (8 * i)));
}
inline void putI32(std::vector<uint8_t>& out, int32_t v)
{
    putU32(out, uint32_t(v));
}
inline void putString(std::vector<uint8_t>& out, const std::string& s)
{
    writeVarint(out, s.size());
    out.insert(out.end(), s.begin(), s.end());
}

struct WireReader
{
    const uint8_t* d = nullptr;
    size_t n         = 0;
    size_t pos       = 0;
    bool ok          = true;
    WireReader(const uint8_t* data, size_t size) : d(data), n(size) {}
    uint64_t le(int bytes)
    {
        if (!ok || n - pos < size_t(bytes))
        {
            ok = false;
            return 0;
        }
        uint64_t v = 0;
        for (int i = 0; i < bytes; ++i) v |= uint64_t(d[pos++]) << (8 * i);
        return v;
    }
    uint8_t u8() { return uint8_t(le(1)); }
    uint16_t u16() { return uint16_t(le(2)); }
    uint32_t u32() { return uint32_t(le(4)); }
    uint64_t u64() { return le(8); }
    int32_t i32() { return int32_t(u32()); }
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
    // A string of at most maxBytes (longer: an error)
    std::string string(size_t maxBytes)
    {
        const uint64_t len = varint();
        if (!ok || len > maxBytes)
        {
            ok = false;
            return {};
        }
        const uint8_t* p = bytes(len);
        return p ? std::string(reinterpret_cast<const char*>(p), size_t(len)) : std::string();
    }
    size_t remaining() const { return ok ? n - pos : 0; }
    bool atEnd() const { return ok && pos == n; }
};

}  // namespace Engine
}  // namespace Saiga
