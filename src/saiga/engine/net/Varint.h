/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Unsigned LEB128 varints (at most 10 bytes for 64 bits) and zigzag for signed values (S15; S16's quantization). The
// reader treats its input as untrusted: over-long encodings and overflow bits are errors, never asserts.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace Saiga
{
namespace Engine
{
inline void writeVarint(std::vector<uint8_t>& out, uint64_t v)
{
    while (v >= 0x80)
    {
        out.push_back(uint8_t(v | 0x80));
        v >>= 7;
    }
    out.push_back(uint8_t(v));
}

inline size_t varintSize(uint64_t v)
{
    size_t n = 1;
    while (v >= 0x80)
    {
        v >>= 7;
        n++;
    }
    return n;
}

// Reads a varint at data[pos] (pos advances); false on truncated, over-long (more than 10 bytes) or overflowing input
inline bool readVarint(const uint8_t* data, size_t size, size_t& pos, uint64_t& v)
{
    v = 0;
    for (int i = 0; i < 10; ++i)
    {
        if (pos >= size) return false;
        const uint8_t b = data[pos++];
        // The 10th byte holds only the top bit of 64
        if (i == 9 && (b & 0xFE) != 0) return false;
        v |= uint64_t(b & 0x7F) << (7 * i);
        if ((b & 0x80) == 0) return true;
    }
    return false;
}

inline uint64_t zigzag(int64_t v)
{
    return (uint64_t(v) << 1) ^ uint64_t(v >> 63);
}
inline int64_t unzigzag(uint64_t v)
{
    return int64_t(v >> 1) ^ -int64_t(v & 1);
}

}  // namespace Engine
}  // namespace Saiga
