/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#include <string_view>

namespace Saiga
{
namespace Engine
{
// FNV-1a 64 (docs/engine_design.md §7.10): seeds, content hashes and world.hash(). Never std::hash (its values differ
// between standard libraries).
constexpr uint64_t kFnvOffset = 14695981039346656037ULL;
constexpr uint64_t kFnvPrime  = 1099511628211ULL;

// Bytes. A separate name: with one overload set, fnv1a64(str.c_str(), seed) would pick the (pointer, size) overload and
// read seed bytes
inline uint64_t fnv1a64Bytes(const void* data, size_t size, uint64_t h = kFnvOffset)
{
    auto p = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < size; ++i)
    {
        h ^= p[i];
        h *= kFnvPrime;
    }
    return h;
}

constexpr uint64_t fnv1a64(std::string_view s, uint64_t h = kFnvOffset)
{
    for (char c : s)
    {
        h ^= uint64_t(static_cast<unsigned char>(c));
        h *= kFnvPrime;
    }
    return h;
}

/**
 * Incremental FNV-1a 64. Integers are added as little-endian bytes and doubles as their IEEE bit pattern (-0.0 and
 * NaN payloads are kept), so a hash does not depend on the host byte order.
 */
class Hasher
{
   public:
    explicit Hasher(uint64_t seed = kFnvOffset) : h(seed) {}

    Hasher& add(const void* data, size_t size)
    {
        h = fnv1a64Bytes(data, size, h);
        return *this;
    }
    Hasher& addU8(uint8_t v) { return add(&v, 1); }
    Hasher& addU32(uint32_t v) { return addLittleEndian(v, 4); }
    Hasher& addU64(uint64_t v) { return addLittleEndian(v, 8); }
    Hasher& addI32(int32_t v) { return addU32(uint32_t(v)); }
    Hasher& addI64(int64_t v) { return addU64(uint64_t(v)); }
    Hasher& addDouble(double v)
    {
        uint64_t bits;
        static_assert(sizeof(bits) == sizeof(v), "double must be 64 bit");
        std::memcpy(&bits, &v, sizeof(bits));
        return addU64(bits);
    }
    // The length (u64) first, so "ab" + "c" differs from "a" + "bc"
    Hasher& addString(std::string_view s)
    {
        addU64(uint64_t(s.size()));
        return add(s.data(), s.size());
    }

    uint64_t value() const { return h; }

   private:
    Hasher& addLittleEndian(uint64_t v, int bytes)
    {
        unsigned char b[8];
        for (int i = 0; i < bytes; ++i) b[i] = static_cast<unsigned char>(v >> (8 * i));
        return add(b, size_t(bytes));
    }

    uint64_t h;
};

// A derived seed or combined hash: FNV-1a over the 16 little-endian bytes of (seed, value); order dependent
inline uint64_t hashCombine(uint64_t seed, uint64_t value)
{
    return Hasher().addU64(seed).addU64(value).value();
}

}  // namespace Engine
}  // namespace Saiga
